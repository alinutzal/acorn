import os

import yaml
import torch
import scipy.sparse as sps
import numpy as np
import pandas as pd

from time import time as tt
from tqdm import tqdm

from gnn4itk_cf.stages.graph_construction.models.metric_learning import MetricLearning
from gnn4itk_cf.stages.edge_classifier.models.filter import Filter
from gnn4itk_cf.stages.edge_classifier import InteractionGNN

from gnn4itk_cf.stages.track_building import utils 
from torch_geometric.utils import to_scipy_sparse_matrix

def evaluate_labelled_graphs(graphset, config):
    all_y_truth, all_pt  = [], []
    evaluated_events = [
        utils.evaluate_labelled_graph(
            event,
            matching_fraction=config["matching_fraction"],
            matching_style=config["matching_style"],
            min_track_length=config["min_track_length"],
            min_particle_length=config["min_particle_length"],
        )
        for event in tqdm(graphset)
    ]
    evaluated_events = pd.concat(evaluated_events)

    particles = evaluated_events[evaluated_events["is_reconstructable"]]
    reconstructed_particles = particles[particles["is_reconstructed"] & particles["is_matchable"]]
    tracks = evaluated_events[evaluated_events["is_matchable"]]
    matched_tracks = tracks[tracks["is_matched"]]

    n_particles = len(particles.drop_duplicates(subset=['event_id', 'particle_id']))
    n_reconstructed_particles = len(reconstructed_particles.drop_duplicates(subset=['event_id', 'particle_id']))

    n_tracks = len(tracks.drop_duplicates(subset=['event_id', 'track_id']))
    n_matched_tracks = len(matched_tracks.drop_duplicates(subset=['event_id', 'track_id']))

    n_dup_reconstructed_particles = len(reconstructed_particles) - n_reconstructed_particles

    print(f"Number of reconstructed particles: {n_reconstructed_particles}")
    print(f"Number of particles: {n_particles}")
    print(f"Number of matched tracks: {n_matched_tracks}")
    print(f"Number of tracks: {n_tracks}")
    print(f"Number of duplicate reconstructed particles: {n_dup_reconstructed_particles}")   

    # Plot the results across pT and eta
    eff = n_reconstructed_particles / n_particles
    fake_rate = 1 - (n_matched_tracks / n_tracks)
    dup_rate = n_dup_reconstructed_particles / n_reconstructed_particles

    # logging.info(f"Efficiency: {eff:.3f}")
    # logging.info(f"Fake rate: {fake_rate:.3f}")
    # logging.info(f"Duplication rate: {dup_rate:.3f}")
    print(f"Efficiency: {eff:.3f}")
    print(f"Fake rate: {fake_rate:.3f}")
    print(f"Duplication rate: {dup_rate:.3f}")

def inference(model, dataloader, device):
    graphs = []
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)    
        #out = model_mm.infer(batch)
        gnn = model_gnn.shared_evaluation(batch,batch_idx)
        #print(gnn['output'].max(),gnn['output'].min())
        batch = gnn['batch']
        # with torch.no_grad():
        #     if device == 'cuda':
        #         with torch.cuda.amp.autocast():
        #             out = model_gnn(batch)
        # batch.scores = torch.sigmoid(out)
        #model_gnn.log_metrics(gnn['output'],gnn['all_truth'],gnn['target_truth'],gnn['loss'])
        edge_mask = gnn['output'] >= 0.8 #model_gnn.hparams['edge_cut'] # score_cut for evaluation
        
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
        candidate_labels = sps.csgraph.connected_components(
            sparse_edges, directed=False, return_labels=True
        )
        batch.labels = torch.from_numpy(candidate_labels[1]).long()
        graphs.append(batch.to('cpu'))

    evaluate_labelled_graphs(graphs, config_tbe)
    
if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    #########################################
    ### 2 Initializing the Model
    #########################################
    config_gnn = yaml.load(open(os.environ['HOME']+'/acorn/examples/Example_1/gnn_train.yaml'), Loader=yaml.FullLoader)
    config_tbe = yaml.safe_load(open(os.environ['HOME']+"/acorn/examples/Example_1/track_building_eval.yaml", "r"))
    model_gnn = InteractionGNN.load_from_checkpoint(config_gnn['stage_dir']+'artifacts/best-v3.ckpt')    
    model_gnn.setup('predict')
    model_gnn.to(device)
    dataloaders = model_gnn.predict_dataloader()
    
    inference(model_gnn, dataloaders[2], device)
    