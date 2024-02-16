import os
import gc
import yaml
import torch
import scipy.sparse as sps
import numpy as np
import pandas as pd

from time import time as tt
from tqdm import tqdm
import time, rmm, pprint
import logging

from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier import InteractionGNN, InteractionGNN2
from acorn.stages.graph_construction.models.py_module_map import PyModuleMap
from acorn.stages.graph_construction.models.utils import graph_intersection, build_edges
from acorn.stages.track_building import utils 
from torch_geometric.utils import to_scipy_sparse_matrix
from acorn.utils.version_utils import get_pyg_data_keys

from acorn.utils import handle_hard_cuts, handle_edge_features
pd.set_option('display.float_format', '{:10.4f}'.format)

from pynvml import *
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()

import profile

def make_result_summary(
    n_reconstructed_particles,
    n_particles,
    n_matched_tracks,
    n_tracks,
    n_dup_reconstructed_particles,
    eff,
    fake_rate,
    dup_rate,
):
    summary = f"Number of reconstructed particles: {n_reconstructed_particles}\n"
    summary += f"Number of particles: {n_particles}\n"
    summary += f"Number of matched tracks: {n_matched_tracks}\n"
    summary += f"Number of tracks: {n_tracks}\n"
    summary += (
        "Number of duplicate reconstructed particles:"
        f" {n_dup_reconstructed_particles}\n"
    )
    summary += f"Efficiency: {eff:.3f}\n"
    summary += f"Fake rate: {fake_rate:.3f}\n"
    summary += f"Duplication rate: {dup_rate:.3f}\n"

    return summary


def tracking_efficiency(dataset, config):
    """
    Plot the track efficiency vs. pT of the edge.
    """
    all_y_truth, all_pt = [], []
    #dataset = getattr(self, config["dataset"])

    evaluated_events = []
    for event in tqdm(dataset):
        evaluated_events.append(
            utils.evaluate_labelled_graph(
                event,
                matching_fraction=config["matching_fraction"],
                matching_style=config["matching_style"],
                sel_conf=config["target_tracks"],
                min_track_length=config["min_track_length"],
            )
        )

    evaluated_events = pd.concat(evaluated_events)
    #print("Debug: ", evaluated_events)
    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[
        particles["is_reconstructed"] & particles["is_matchable"]
    ]
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=["event_id", "particle_id"]))
    n_reconstructed_particles = len(
        reconstructed_particles.drop_duplicates(subset=["event_id", "particle_id"])
    )

    n_tracks = len(tracks.drop_duplicates(subset=["event_id", "track_id"]))
    n_matched_tracks = len(
        matched_tracks.drop_duplicates(subset=["event_id", "track_id"])
    )

    n_dup_reconstructed_particles = (
        len(reconstructed_particles) - n_reconstructed_particles
    )

    eff = n_reconstructed_particles / n_particles
    fake_rate = 1 - (n_matched_tracks / n_tracks)
    dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles

    result_summary = make_result_summary(
        n_reconstructed_particles,
        n_particles,
        n_matched_tracks,
        n_tracks,
        n_dup_reconstructed_particles,
        eff,
        fake_rate,
        dup_rate,
    )

    print(f"Number of reconstructed particles: {n_reconstructed_particles}")
    print(f"Number of particles: {n_particles}")
    print(f"Number of matched tracks: {n_matched_tracks}")
    print(f"Number of tracks: {n_tracks}")
    print(f"Number of duplicate reconstructed particles: {n_dup_reconstructed_particles}")   
    print(f"Efficiency: {eff:.3f}")
    print(f"Fake rate: {fake_rate:.3f}")
    print(f"Duplication rate: {dup_rate:.3f}")
    
    log = logging.getLogger("Track Building Results")
    log.info("Result Summary :\n\n" + result_summary)
    res_fname = os.path.join(
        config["stage_dir"],
        f"results_summary_{config['matching_style']}.txt",
    )

    with open(res_fname, "w") as f:
        f.write(result_summary)

    # First get the list of particles without duplicates
    grouped_reco_particles = particles.groupby("particle_id")[
        "is_reconstructed"
    ].any()
    # particles["is_reconstructed"] = particles["particle_id"].isin(grouped_reco_particles[grouped_reco_particles].index.values)
    particles.loc[
        particles["particle_id"].isin(
            grouped_reco_particles[grouped_reco_particles].index.values
        ),
        "is_reconstructed",
    ] = True
    particles = particles.drop_duplicates(subset=["particle_id"])

    # Plot the results across pT and eta (if provided in conf file)
    os.makedirs(config["stage_dir"], exist_ok=True)

    plot_config = config["plots"]['tracking_efficiency']
    for var, varconf in plot_config['variables'].items():
        utils.plot_eff(
            particles,
            var,
            varconf,
            save_path=os.path.join(
                config["stage_dir"],
                f"track_reconstruction_eff_vs_{var}_{config['matching_style']}.png",
            ),
        )
    
def scale_features(event, config):
    """
    Handle feature scaling for the event
    """

    if (
        config is not None
        and "node_scales" in config.keys()
        and "node_features" in config.keys()
    ):
        assert isinstance(
            config["node_scales"], list
        ), "Feature scaling must be a list of ints or floats"
        for i, feature in enumerate(config["node_features"]):
            assert feature in get_pyg_data_keys(
                event
            ), f"Feature {feature} not found in event"
            event[feature] = event[feature] / config["node_scales"][i]

    return event

def add_edge_features(event, config):
    if "edge_features" in config.keys():
        assert isinstance(
            config["edge_features"], list
        ), "Edge features must be a list of strings"
        handle_edge_features(event, config["edge_features"])
    return event

def process_event(batch, truth, config, config_gnn, device, stats_pool_memory_resource, model_gnn, model_map, model_fil=None):
    running_time = []
    if config['debug']==True:
        print(batch.event_id)

    running_time.append(batch.event_id)
    if config['debug']==True:
        print(batch.x.size(0))
    running_time.append(batch.x.size(0))
    if config['graph_construction'] == 'ModuleMap':
        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        start_event = start
        if device == 'cuda':        
            starter.record() #gpu
            starter_event = starter
        batch, mm_time_ind = model_map.build_graph(batch, truth)
        mm_time = 0
        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender)/1000.0
        end = time.time()
        running_time.extend(((end - start), gpu_time))
        if config['debug']==True:
            print("MM: ", ((end - start), gpu_time))
        #print(nvsmi.DeviceQuery('memory.free, memory.used'))

        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        if device == 'cuda':        
            starter.record() #gpu   
        #handle_hard_cuts(batch, model_gnn.hparams["hard_cuts"])
        batch = scale_features(batch, model_gnn.hparams)

        if model_gnn.hparams["edge_features"] is not None:
            batch = add_edge_features(
            batch, model_gnn.hparams
        )  # scaling must be done before adding features
        batch = batch.to(device)
        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender)/1000.0
        end = time.time()
        running_time.extend(((end - start), gpu_time))
        if config['debug']==True:
            print("Preprocess GNN: ", ((end - start), gpu_time))

            print(batch.edge_index.size(1))
        running_time.append(batch.edge_index.size(1))
    else:
        with torch.no_grad():
            if device == 'cuda':
                #with torch.cuda.amp.autocast():
                    embedding = model_ml.apply_embedding(batch)
        
        batch.edge_index = build_edges(
            query=embedding, database=embedding, indices=None, r_max=0.1, k_max=10, backend="FRNN"
        )
        R = batch.r**2 + batch.z**2
        flip_edge_mask = R[batch.edge_index[0]] > R[batch.edge_index[1]]
        batch.edge_index[:, flip_edge_mask] = batch.edge_index[:, flip_edge_mask].flip(0)
        with torch.no_grad():
            if device == 'cuda':
                #with torch.cuda.amp.autocast():
                    out = model_fil(batch)   
        preds = torch.sigmoid(out)
        batch.edge_index = batch.edge_index[:, preds > model_fil.hparams['edge_cut']]            
        
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
    if device == 'cuda':        
        starter.record() #gpu   

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            batch.scores = torch.sigmoid(model_gnn(batch))
    #max_memory_py = torch.cuda.max_memory_allocated(device) / 1024**3
    #max_memory_py = stats_pool_memory_resource.allocation_counts['peak_bytes'] / 1024**3
    max_memory_py = 0
    if config['debug']==True:
        print(f"Maximum memory allocated on {device}: {max_memory_py} GB")

    edge_mask = batch.scores > config_tbi['score_cut_cc']
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0
    end = time.time()
    running_time.extend(((end - start), gpu_time))
    if config['debug']==True:
        print("GNN: ", ((end - start), gpu_time))
    
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()

    if device == 'cuda':        
        starter.record() #gpu          
    # Get number of nodes
    if hasattr(batch, "num_nodes"):
        num_nodes = batch.num_nodes
    elif hasattr(batch, "x"):
        num_nodes = batch.x.size(0)
    elif hasattr(batch, "x_x"):
        num_nodes = batch.x_x.size(0)
    else:
        num_nodes = batch.edge_index.max().item() + 1
    # Convert to sparse scipy array
    sparse_edges = to_scipy_sparse_matrix(
        batch.edge_index[:, edge_mask], num_nodes=num_nodes
    )
    # Run connected components
    _, candidate_labels = sps.csgraph.connected_components(
        sparse_edges, directed=False, return_labels=True
    )
    batch.labels = torch.from_numpy(candidate_labels).long()

    #tracking_efficiency(graphs, config_tbe)
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0
        gpu_time_event = starter_event.elapsed_time(ender)/1000.0
    end = time.time()

    running_time.extend(((end - start), gpu_time))
    if config['debug']==True:
        print("Tracking: ", ((end - start), gpu_time))
    
    running_time.extend(((end - start_event), gpu_time_event))
    if config['debug']==True:
        print("Total per event: ", ((end - start_event), gpu_time_event))  
    #max_memory_py = stats_pool_memory_resource.allocation_counts['peak_bytes'] / 1024**3
    max_memory_py = 0
    running_time.extend([max_memory_py])    
    gc.collect()
    torch.cuda.empty_cache()    
    return batch, running_time, mm_time_ind

def inference(config, device, model_gnn, model_map, model_fil=None):
    mr = rmm.mr.get_current_device_resource()
    stats_pool_memory_resource = rmm.mr.StatisticsResourceAdaptor(mr)
    rmm.mr.set_current_device_resource(stats_pool_memory_resource)
    resultsDF = pd.DataFrame()
    
    graphs = []
    all_events_time = []
    all_mm_time = []
    if config['graph_construction'] != 'ModuleMap':
        model_ml = model_map
    wmode ='w'
    header = True
    if len(model_map.testset) < 100:
        debug = True
    # Warm up 
    #process_event(model_map.testset[0][0], model_map.testset[0][2], config, config_gnn, device, stats_pool_memory_resource,\
    #        model_gnn, model_map, model_fil)
    # Inference
    for batch_idx, (graph, _, truth) in enumerate(model_mm.testset):
        print("Event Id:", graph.event_id)
        batch, running_time, mm_time_ind = process_event(graph, truth, config, config_gnn, device, stats_pool_memory_resource,\
            model_gnn, model_map, model_fil)
        graphs.append(batch.to('cpu'))
        all_events_time.append(running_time)
        all_mm_time.append(mm_time_ind)
        # if batch_idx % 100 == 0 and batch_idx != 0:
        #     print("Batch: ", batch_idx)
        #     resultsDF = save_results(all_events_time, all_mm_time, resultsDF)
        #     if batch_idx >= 100:
        #         wmode ='a'
        #         header = False
        #     print(resultsDF)
        #     resultsDF.to_csv(config['stage_dir']+'results.csv', mode=wmode, header=header)
        #     all_events_time = []
        #     all_mm_time = []
    resultsDF = save_results(all_events_time, all_mm_time, resultsDF)
    resultsDF.set_index('event_id', inplace=True)
    resultsDF.loc['mean'] = resultsDF.mean()
    resultsDF.loc['std'] = resultsDF.std()
    resultsDF['#nodes'] = resultsDF['#nodes'].astype(int)
    resultsDF['#edges'] = resultsDF['#edges'].astype(int)
    print(resultsDF)
    resultsDF.to_csv(config['stage_dir']+'results_final.csv', mode=wmode, header=header)
    return graphs
 
def save_results(list1, list2, resultsDF):
    """
    Save results to a file
    """
    df1 = pd.DataFrame(list1, columns=['event_id','#nodes','MM','MM_gpu','Pre','Pre_gpu',\
        '#edges','GNN', 'GNN_gpu','Track','Track_gpu','Total','Total_gpu', 'MemoryPyT'])
    df2 = pd.DataFrame(list2, columns=['merge',"doublet","doublet2","triplet","concat","get y","max_memory"])
    resultsIter = pd.concat([df1 , df2], axis=1)
    resultsDF = pd.concat([resultsDF, resultsIter], axis=0)
    return resultsDF

    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Inference args.')
    # parser.add_argument('--option', default='map', help='map for module map or metric for metric learning')
    # parser.add_argument('--example', default='CTD_2023', help='Directory of the example (config files) to run')
    # parser.add_argument('--debug', default=False, help='Print infor per event')
    # args = parser.parse_args()
    # print(args.option)
    
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(device)
    #use Rapids Memory Manager
    from rmm.allocators.torch import rmm_torch_allocator
    #torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
    
    #torch.cuda.memory._record_memory_history()
    #torch.cuda.reset_max_memory_allocated() 
    #########################################
    ### 2 Initializing the Model
    #########################################
    example = 'CTD_2023'
    exPath = os.environ['HOME']+'/acorn/examples/'+ example +'/'

    configPipe = exPath + 'pipe_infer.yaml'
    config_pipe = yaml.load(open(configPipe), Loader=yaml.FullLoader)
    print(config_pipe)
 
    configDr = exPath + config_pipe['data_reader']+'.yaml'
    configGnn_infer = exPath + config_pipe['gnn_infer']+'.yaml'
    configTbi = exPath + config_pipe['track_building']+'.yaml'
    configTbe = exPath + config_pipe['track_evaluation']+'.yaml'
    
    if config_pipe['graph_construction'] == 'ModuleMap': 
        configMm = exPath + 'module_map_infer.yaml'
    else:
        configMl = exPath + 'metric_learning_infer.yaml'
        configFil = exPath + 'filter_infer.yaml'
    dataPath = config_pipe['input_dir']
    
    if config_pipe['graph_construction'] == 'ModuleMap':    
        config_mm = yaml.load(open(configMm), Loader=yaml.FullLoader)
        print("Loading Module Map", configMm)
        model_mm = PyModuleMap(config_mm)
        model_mm.load_module_map()
        model_mm.load_data(dataPath)
        model_mm = model_mm.to(device)
        print("Loaded Module Map")
    else:
        config_ml = yaml.load(open(configMl), Loader=yaml.FullLoader)
        print("Loading Metric Learning")
        model_ml = MetricLearning(config_ml)
        model_ml.load_from_checkpoint(config_ml['stage_dir']+'artifacts/last-v2-180.ckpt') #last--v1.ckpt
        model_ml.load_data(dataPath)
        model_ml = model_ml.to(device)
        print("Loaded Metric Learning")
        
        config_fil = yaml.load(open(configFil), Loader=yaml.FullLoader)
        print("Loading Filtering")
        model_fil = Filter(config_fil)
        model_fil.load_from_checkpoint(config_fil['stage_dir']+'artifacts/last--v1.ckpt') 
        model_fil = model_fil.to(device)
        print("Loaded Filtering")
    
    config_gnn = yaml.load(open(configGnn_infer), Loader=yaml.FullLoader)
    config_tbi = yaml.load(open(configTbi), Loader=yaml.FullLoader)   
    config_tbe = yaml.safe_load(open(configTbe, "r"))

    model_gnn = InteractionGNN2.load_from_checkpoint(config_gnn['stage_dir']+'artifacts/last-v2-180.ckpt',\
        map_location=torch.device(device))    
    #model_gnn.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    #model_gnn.hparams['input_dir'] = mmPath
    #model_gnn.setup('predict')
    print(model_gnn.hparams)
    model_gnn = model_gnn.to(device)
 
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
 
    if device == 'cuda':
        starter.record()   
    #profile.run('inference(model_mm, model_gnn, device)')
    
    if config_pipe['graph_construction'] == 'ModuleMap':
        graphs = inference(config_pipe, device, model_gnn, model_mm)
    else:
        graphs = inference(config_pipe, device, model_gnn, model_ml, model_fil)
    end = time.time()
    end_cpu = time.process_time()
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0    
    print("Total: ",((end - start),gpu_time))  
    tracking_efficiency(graphs, config_tbe)
    #torch.cuda.memory._dump_snapshot("my_mem_snapshot.pickle")
    #print(torch.cuda.memory_summary(device))



