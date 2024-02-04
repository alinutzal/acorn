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

def inference(model_mm, model_gnn, device):
    graphs = []
    all_events_time = []
    all_mm_time = []
    print(len(model_mm.valset))
    for batch_idx, (graph, _, truth) in enumerate(model_mm.valset):
    #for batch_idx, batch in enumerate(model_gnn.valset):
        #print(device)
        #graph.to(device)
        #if batch.event_id != '000000123': continue

        running_time = []
        
        gpu_time = 0
        if device == 'cuda':
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        if device == 'cuda':        
            starter.record() #gpu
        batch= model_mm.build_graph(graph, truth)
        mm_time = 0

        # want bypass saving to disk
        # Initiate a graph dataset instance from 
        print(batch.event_id)
        running_time.extend((batch.event_id))


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
        batch = model_gnn.valset.preprocess_event(batch.to('cpu')) 
        batch.to(device)

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
            gnn = model_gnn.shared_evaluation(batch,batch_idx)
            batch = gnn['batch']

        #batch.scores = torch.sigmoid(gnn['output'])

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
        end = time.time()

        running_time.extend(((end - start), gpu_time))
        print("Tracking: ", ((end - start), gpu_time))
        
        graphs.append(batch.to('cpu'))
        del batch, gnn 
        gc.collect()
        torch.cuda.empty_cache()
        all_events_time.append(running_time)
        all_mm_time.append(mm_time)
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
    model_mm = PyModuleMap(config_mm)
    model_mm.load_module_map()
    model_mm.load_data(dataPath)
    
    config_gnn = yaml.load(open(configGnn), Loader=yaml.FullLoader)
    config_tbi = yaml.load(open(configTbi), Loader=yaml.FullLoader)   
    config_tbe = yaml.safe_load(open(configTbe, "r"))
    model_gnn = InteractionGNN.load_from_checkpoint(config_gnn['stage_dir']+'artifacts/last--v1.ckpt')    

    model_gnn.hparams['input_dir'] = mmPath
    print(model_gnn.hparams['input_dir'])
    model_gnn.setup('predict')
    #print(model_gnn.valset)
    
    model_mm = model_mm.to(device)
    model_gnn = model_gnn.to(device)
    #print(len(model_mm.valset))
 
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
 
    if device == 'cuda':
        starter.record()   
    #profile.run('inference(model_mm, model_gnn, device)')
    list1, list2 = inference(model_mm, model_gnn, device)
    end = time.time()
    end_cpu = time.process_time()
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0     
    print(list1, list2)
    df1 = pd.DataFrame(list1, columns=['col_1','col_2','col_3','col_4','col_5','col_6','col_7','col_8','col_9','col_10',\
        'col_11','col_12','col_13','col_14','col_15','col_16','col_17'])
    #df2 = pd.DataFrame(list2, columns=['merge',"doublet edges","doublet edges 2","triplet edges","concat","get y"])
    df1.to_csv("resuts1.csv")
    #df2.to_csv("resuts2.csv")
    print("Total: ",((end - start),gpu_time)) 