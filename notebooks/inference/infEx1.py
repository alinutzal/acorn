import os
import gc
import yaml
import torch
import scipy.sparse as sps
import numpy as np
import pandas as pd

from time import time as tt
from tqdm import tqdm
import time

from acorn.stages.graph_construction.models.metric_learning import MetricLearning
from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier import InteractionGNN
from acorn.stages.graph_construction.models.py_module_map import PyModuleMap

from acorn.stages.track_building import utils 
from torch_geometric.utils import to_scipy_sparse_matrix
from acorn.utils.version_utils import get_pyg_data_keys

from acorn.utils import handle_hard_cuts

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


def tracking_efficiency(dataset, config): #plot_config,
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

    #self.log.info("Result Summary :\n\n" + result_summary)

    # res_fname = os.path.join(
    #     self.hparams["stage_dir"],
    #     f"results_summary_{self.hparams['matching_style']}.txt",
    # )

    # with open(res_fname, "w") as f:
    #     f.write(result_summary)

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
    #os.makedirs(self.hparams["stage_dir"], exist_ok=True)

    # for var, varconf in plot_config["variables"].items():
    #     utils.plot_eff(
    #         particles,
    #         var,
    #         varconf,
    #         save_path=os.path.join(
    #             self.hparams["stage_dir"],
    #             f"track_reconstruction_eff_vs_{var}_{self.hparams['matching_style']}.png",
    #         ),
    #     )
    
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

def inference(mode, model_map, model_metric, model_filer, model_gnn, device):
    graphs = []
    all_events_time = []
    all_mm_time = []
    if mode == 'map':
        model_mm = model_map
    else:
        model_mm = model_metric
    
    print(len(model_mm.valset))
    for batch_idx, (graph, _, truth) in enumerate(model_mm.valset):
        running_time = []
        print(graph.event_id)
        running_time.append(graph.event_id)
        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        start_event = start
        if device == 'cuda':        
            starter.record() #gpu
            starter_event = starter
        batch, mm_time_ind = model_map.build_graph(graph, truth)
        mm_time = 0
        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender)/1000.0
        end = time.time()
        running_time.extend(((end - start), gpu_time))
        print("MM: ", ((end - start), gpu_time))
        #print(nvsmi.DeviceQuery('memory.free, memory.used'))
        
        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        if device == 'cuda':        
            starter.record() #gpu   
        handle_hard_cuts(batch, model_gnn.hparams["hard_cuts"])
        batch = scale_features(batch, model_gnn.hparams)
        if config_gnn.get("edge_features") is not None:
            event = add_edge_features(
            event
        )  # scaling must be done before adding features

        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender)/1000.0
        end = time.time()
        running_time.extend(((end - start), gpu_time))
        print("Preprocess GNN: ", ((end - start), gpu_time))

        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        if device == 'cuda':        
            starter.record() #gpu   

        with torch.no_grad():
            batch.scores = torch.sigmoid(model_gnn(batch))

        edge_mask = batch.scores > config_tbi['score_cut']
        if device == 'cuda':
            ender.record()
            torch.cuda.synchronize()
            gpu_time = starter.elapsed_time(ender)/1000.0
        end = time.time()
        running_time.extend(((end - start), gpu_time))
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
        print("Tracking: ", ((end - start), gpu_time))
        
        running_time.extend(((end - start_event), gpu_time_event))
        print("Total per event: ", ((end - start_event), gpu_time_event))      
         
        graphs.append(batch.to('cpu'))
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        all_events_time.append(running_time)
        all_mm_time.append(mm_time_ind)
    tracking_efficiency(graphs, config_tbe)

    return (all_events_time, all_mm_time)
    
if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"
    print(device)
    #########################################
    ### 2 Initializing the Model
    #########################################
    exPath = os.environ['HOME']+'/acorn/examples/Example_1/'
    dataPath = '/pscratch/sd/a/alazar/cf/Example_1/feature_store/'
    mmPath = '/pscratch/sd/a/alazar/cf/Example_1/module_map/'
    confidr = exPath + 'data_reader.yaml'
    configMm = exPath + 'module_map_infer.yaml'
    configGnn = exPath + 'gnn_infer.yaml'
    configGnn_eval = exPath + 'gnn_eval.yaml'
    configTbi = exPath + 'track_building_infer.yaml'
    configTbe = exPath + 'track_building_eval.yaml'
    
    config_mm = yaml.load(open(configMm), Loader=yaml.FullLoader)
    print("Loading Module Map")
    model_mm = PyModuleMap(config_mm)
    model_mm.load_module_map()
    model_mm.load_data(dataPath)
    print("Loaded Module Map")
    
    config_gnn = yaml.load(open(configGnn), Loader=yaml.FullLoader)
    config_tbi = yaml.load(open(configTbi), Loader=yaml.FullLoader)   
    config_tbe = yaml.safe_load(open(configTbe, "r"))
    print("Loading GNN")
    model_gnn = InteractionGNN.load_from_checkpoint(config_gnn['stage_dir']+'artifacts/last--v1.ckpt')    

    model_gnn.hparams['input_dir'] = mmPath
    model_gnn.setup('predict')
    print("Loaded GNN")
    #print(model_gnn.valset)
    
    print("Put models on GPU")
    model_mm = model_mm.to(device)
    model_gnn = model_gnn.to(device)
    print("Models loaded to GPU")
    #print(len(model_mm.valset))
 
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
 
    if device == 'cuda':
        starter.record()   
    #profile.run('inference(model_mm, model_gnn, device)')
    list1, list2 = inference('map', model_mm, model_mm, model_gnn, model_gnn, device)
    end = time.time()
    end_cpu = time.process_time()
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0    
    print("Total: ",((end - start),gpu_time))  

    df1 = pd.DataFrame(list1, columns=['event_id','MM','MM_gpu','Preprocess','Preprocess_gpu',\
        'GNN', 'GNN_gpu','Tracking','Tracking_gpu','Total','Total_gpu'])
    df2 = pd.DataFrame(list2, columns=['merge',"doublet edges","doublet edges 2","triplet edges","concat","get y"])
    resultsDF = pd.concat([df1 , df2], axis=1)
    resultsDF.set_index('event_id', inplace=True)
    resultsDF.loc['mean'] = resultsDF.mean()
    resultsDF.loc['std'] = resultsDF.std()
    print(resultsDF)
    resultsDF.to_csv("resuts.csv")

