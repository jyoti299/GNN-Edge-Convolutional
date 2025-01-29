#!/usr/bin/env python3

import time
import glob
import uproot
import numpy as np
import awkward as ak
import pickle
from tqdm import tqdm

import torch
from torch_geometric.data import Data
THRESHOLD_Z = 0.3
THRESHOLD_VERTEX_Z = 0.3
THRESHOLD_VERTEX_T = 0.015

def mkdir_p(mypath):
    '''Function to create a new directory, if it not already exist.
       Args:
           mypath : directory path
    '''
    from errno import EEXIST
    from os import makedirs, path
    try:
        makedirs(mypath)
    except OSError as exc:
        if not (exc.errno == EEXIST and path.isdir(mypath)):
            raise

def computeEdgeAndLabels(track_data, truth_data, edges, edges_labels, edge_features):
    '''Compute the truth graph'''
    print("Building the graph")

    n_tracks = len(ak.flatten(track_data["gnn_pt"]))

    print('n_tracks', n_tracks)
    z_pca = ak.to_numpy(ak.flatten(track_data["gnn_z_pca"]))
    sim_vertex_ID = ak.to_numpy(ak.flatten(track_data.gnn_sim_vertex_evID))
    sim_vertex_BX = ak.to_numpy(ak.flatten(truth_data.gnn_sim_vertex_BX))
    sim_vertex_z = ak.to_numpy(ak.flatten(truth_data.gnn_sim_vertex_z))
    sim_vertex_t = ak.to_numpy(ak.flatten(truth_data.gnn_sim_vertex_t))
    sim_vertex_LV = ak.to_numpy(ak.flatten(truth_data.gnn_sim_vertex_isLV))
    sim_vertex_index = ak.to_numpy(ak.flatten(truth_data.gnn_sim_vertex_index))
    gnn_pt = ak.to_numpy(ak.flatten(track_data.gnn_pt))
    t_pi = ak.to_numpy(ak.flatten(track_data["gnn_t_Pi"]))
    t_k = ak.to_numpy(ak.flatten(track_data["gnn_t_K"]))
    t_p = ak.to_numpy(ak.flatten(track_data["gnn_t_P"]))
    dz = ak.to_numpy(ak.flatten(track_data["gnn_dz"]))
    #t_pi = np.array(track_data.gnn_t_Pi)
    #t_k = np.array(track_data.gnn_t_K)
    #t_p = np.array(track_data.gnn_t_P)
    #dz = np.array(track_data.gnn_dz)
    print("Sizes:", len(sim_vertex_ID), len(sim_vertex_LV), len(sim_vertex_BX))
    vertex_id_size = len(ak.flatten(track_data.gnn_sim_vertex_evID))
    if vertex_id_size > 1:
      for i in tqdm(range(vertex_id_size)):
        if i == vertex_id_size -1:
           break 
        
        z_diff = np.abs(z_pca[i+1:vertex_id_size] - z_pca[i])
        ids = np.where(z_diff < THRESHOLD_Z)[0] 
        valid_indices = ids + i + 1
        edges.extend([(j, i) for j in valid_indices])
        edges.extend([(i, j) for j in valid_indices])
        
        labels = ((sim_vertex_ID[i] == sim_vertex_ID[valid_indices]) & (sim_vertex_LV[i] == 1) & (sim_vertex_LV[valid_indices] == 1) & (sim_vertex_BX[i] == 0) &  (sim_vertex_BX[valid_indices] == 0)).astype(int)
        
        edges_labels.extend(labels)
        edges_labels.extend(labels) 
        pt_diff = np.abs(gnn_pt[valid_indices] - gnn_pt[i])
        time_comps = np.array([np.abs(t_pi[valid_indices] - t_pi[i]),
                      np.abs(t_k[valid_indices] - t_k[i]),
                      np.abs(t_p[valid_indices] - t_p[i]),
                      
                      np.abs(t_pi[valid_indices] - t_k[i]),
                      np.abs(t_pi[valid_indices] - t_p[i]),
                      
                      np.abs(t_k[valid_indices] - t_pi[i]),
                      np.abs(t_k[valid_indices] - t_p[i]),
                      
                      np.abs(t_p[valid_indices] - t_k[i]),
                      np.abs(t_p[valid_indices] - t_pi[i])])
        z_diff = z_diff[ids]

        for k, ind in enumerate(valid_indices):
            dz_significance = (z_pca[i] - z_pca[ind]) / np.sqrt(dz[i]**2 + dz[ind]**2)
            #print('k ', k, 'pt_diff[k]', pt_diff[k], 'z_diff[k]', z_diff[k], 'dz_significance ', dz_significance, 'time ',  list(time_comps[:, k]))
            edge_features.append([z_diff[k], dz_significance])
    else:
        print("Number of tracks is less than or equal to 1. No processing needed.")
def set_small_to_zero(a, eps=1e-8):
    a[np.abs(a) < eps] = 0
    return a

def remap_PIDs(pids):
    """
    Remaps particle IDs to a simplified classification scheme.
    
    Args:
    - pids (list): List of particle IDs to be remapped.
    
    Returns:
    - remapped_pids (list): List of remapped particle IDs.
    """
    # Mapping of particle IDs to simplified classification
    pid_map = {11: 0, 13: 0, 211: 0, 321: 1, 2212: 2, 3112: 2}
    
    # Remap PIDs using the pid_map dictionary
    remapped_pids = [pid_map.get(pid, -1) for pid in pids]
    
    return remapped_pids

def process_files(input_folder, output_folder, n_files=100000, offset=0):
    files = glob.glob(f"{input_folder}/*.root")
    print(f"Number of files: {len(files)}")
    
    print('input folder', input_folder, ' space ', output_folder)

    X, Edges, Edges_labels, Edge_features = [], [], [], []
    PIDs_truth, Times_truth = [], []

    mkdir_p(output_folder)
    print('input folder', input_folder, ' space ', output_folder)
    for i_file, file in enumerate(files[offset:offset+n_files]):
        i_file += offset
        print('ifile', file)

        print('\nProcessing file {} '.format(file))
        try:
            with uproot.open(file) as f:
                tree = f["mvaTrainingNtuple"] 
                for ev, key in enumerate(tree):
                    print('key', key)
                    t = tree[key]
                    track_data = t.arrays(["gnn_pt", "gnn_eta", "gnn_phi", "gnn_z_pca",
                                            "gnn_t_Pi", "gnn_t_K", "gnn_t_P", "gnn_mva_qual", 'gnn_btlMatchChi2',
                                            'gnn_btlMatchTimeChi2', 'gnn_etlMatchChi2', 'gnn_sim_vertex_evID',
                                            'gnn_etlMatchTimeChi2', 'gnn_pathLength', 'gnn_npixBarrel', 'gnn_npixEndcap','gnn_mtdTime', 'gnn_is_matched_tp', 'gnn_dz', 'gnn_sigma_tmtd', 'gnn_sigma_tof_P', 'gnn_trk_ndof', 'gnn_trk_chi2']) #, 'gnn_sigma_tof_pi', 'gnn_sigma_tof_k', 'gnn_sigma_tof_p'])
                    truth_data = t.arrays(['gnn_sim_vertex_z', 'gnn_sim_vertex_t', 'gnn_tp_tEst', 'gnn_tp_pdgId','gnn_sim_vertex_evID','gnn_sim_vertex_BX', 'gnn_sim_vertex_index', 'gnn_sim_vertex_isLV'])
                    print('track_data', track_data)
                    number_of_tracks = ak.num(track_data["gnn_pt"])
                    sim_vertex_z = truth_data.gnn_sim_vertex_z
                    sim_vertex_LV = truth_data.gnn_sim_vertex_isLV
                    print(f"{i_file}_{ev} : Have {number_of_tracks} tracks in the file")
                    sim_vertex_t = truth_data.gnn_sim_vertex_t
                    sim_tp_pdg = truth_data.gnn_tp_pdgId
                    print('number of tracks', number_of_tracks)
                    start = time.time()
                    x_ev = ak.to_numpy(ak.flatten([track_data.gnn_z_pca,track_data.gnn_dz]))
                    #x_ev = np.array([track_data.gnn_z_pca,
                    #                 track_data.gnn_dz],
                    #                dtype=np.float32)

                    x_ev = set_small_to_zero(x_ev, eps=1e-5)

                    print(f"{i_file}_{ev} : Got the track properties")

                    X.append(x_ev)

                    edges, edges_labels, edge_features = [], [], []
                    pids, times = truth_data.gnn_tp_pdgId, truth_data.gnn_tp_tEst

                    # Call the function to compute edges and labels
                    computeEdgeAndLabels(track_data, truth_data, edges, edges_labels, edge_features)
                    #print('edges_labels', len(edges_labels), 'and', edges_labels) 
                    #print(len(edge_features), len(edge_features[0]))
                    print(len(X))

                   # edges, edges_labels, edge_features = prepare_data_with_edge_dropping(
                    #    edges, edges_labels, edge_features, drop_fraction=0.70
                    #)
                    Edges.append(np.array(edges).T)
                    PIDs_truth.append(np.array(pids, dtype=np.int64))
                    Times_truth.append(np.array(times, dtype=np.float32))
                    Edges_labels.append(np.array(edges_labels))
                    Edge_features.append(np.array(edge_features))
                    
                    
                    if (ev % 10 == 0 and ev != 0) or ev == len(tree.keys())-1:
                        stop = time.time()
                        print(f"t = {stop - start} ... Saving the pickle data for {i_file}_{ev}")

                        # Save the processed data into pickle files
                        print('loading error 1')
                        with open(f"{output_folder}{i_file}_{ev}_node_features.pkl", "wb") as fp:
                            pickle.dump(X, fp)
                        print('loading error 2')
                        with open(f"{output_folder}{i_file}_{ev}_edges.pkl", "wb") as fp:
                            pickle.dump(Edges, fp)
                        with open(f"{output_folder}{i_file}_{ev}_edges_labels.pkl", "wb") as fp:
                            pickle.dump(Edges_labels, fp)
                        with open(f"{output_folder}{i_file}_{ev}_edge_features.pkl", "wb") as fp:
                            pickle.dump(Edge_features, fp)
                        with open(f"{output_folder}{i_file}_{ev}_times_truth.pkl", "wb") as fp:
                            pickle.dump(Times_truth, fp)
                        with open(f"{output_folder}{i_file}_{ev}_PID_truth.pkl", "wb") as fp:
                            pickle.dump(PIDs_truth, fp)
                        print('filennaame ', f"{output_folder}{i_file}_{ev}_edge_features.pkl")     
                        X, Edges, Edges_labels, Edge_features = [], [], [], []
                        PIDs_truth, Times_truth = [], []
                        start = time.time()


        except Exception as e:
            print(f"Error: {e}")
            continue


def loadData(path, num_files = -1):
    """
    Loads pickle files of the graph data for network training.
    """
    f_edges_label = glob.glob(f"{path}*edges_labels.pkl")
    f_edges_features = glob.glob(f"{path}*edge_features.pkl")
    f_edges = glob.glob(f"{path}*edges.pkl" )
    f_nodes_features = glob.glob(f"{path}*node_features.pkl")
    f_PID = glob.glob(f"{path}*PID_truth.pkl")
    f_times = glob.glob(f"{path}*times_truth.pkl")
    
    edges_label, edges, nodes_features, edges_features, PID_truth, times_truth = [], [], [], [], [], []
    n = len(f_edges_label) if num_files == -1 else num_files
    for i_f, _ in enumerate(tqdm(f_edges_label)):
        # Load the data
        if (i_f <= n):
            f = f_edges_label[i_f]
            with open(f, 'rb') as fb:
                edges_label.append(pickle.load(fb))
                
            f = f_edges_features[i_f]
            with open(f, 'rb') as fb:
                edges_features.append(pickle.load(fb))
                
            f = f_edges[i_f]
            with open(f, 'rb') as fb:
                edges.append(pickle.load(fb))
                
            f = f_nodes_features[i_f]
            with open(f, 'rb') as fb:
                nodes_features.append(pickle.load(fb))
                
            f = f_PID[i_f]
            with open(f, 'rb') as fb:
                PID_truth.append(pickle.load(fb))
                
            f = f_times[i_f]
            with open(f, 'rb') as fb:
                times_truth.append(pickle.load(fb))
                
        else:
            break
            
    return edges_label, edges, nodes_features, edges_features, PID_truth, times_truth


def prepare_test_data(data_list, ev):
    """
    Function to prepare (and possibly standardize) the test data
    """
    x_np, edge_label, edge_index, edge_features = data_list[ev]
    #x_norm, mean, std = standardize_data(x_np)

    # Create torch vectors from the numpy arrays
    x = torch.from_numpy(x_np)
    x = torch.nan_to_num(x, nan=0.0)
    
    e_label = torch.from_numpy(edge_label)
    edge_index = torch.from_numpy(edge_index)
    e_features = torch.from_numpy(edge_features)
    
    data = Data(x=x, num_nodes=torch.tensor(x.shape[0]), edge_index=edge_index, edge_label=edge_label,
               edge_features=edge_features)
    return data


def flatten_lists(el, ed, nd, ef, pid, times):
    edge_label, edges, node_data, edge_features, PID_truth, times_truth = [], [], [], [], [], []
    print('lenght of nodes', len(nd))
    for i, X in enumerate(nd):
        for ev in range(len(X)):
                  
            if len(ed[i][ev]) == 0:
                print(f"Event {i}:{ev} has NO edges. Skipping.")
                continue # skip events with no edges
                
            elif X[ev].shape[1] <= 1:
                print(f"Event {i}:{ev} has {X[ev].shape[1]} nodes. Skipping.")
                continue
            else:
                edges.append(ed[i][ev])
                edge_label.append(el[i][ev])
                node_data.append(X[ev])
                edge_features.append(ef[i][ev])
                PID_truth.append(pid[i][ev])
                times_truth.append(times[i][ev])
    return edge_label, edges, node_data, edge_features, PID_truth, times_truth


def save_dataset(pickle_data, output_location, trainRatio = 0.8, valRatio = 0.1, testRatio = 0.1, num_files=-1):
    
    print("Loading Pickle Files...")
    # obtain edges_label, edges, nodes_features... from all the pickle files
    el, ed, nd, ef, pid, times = loadData(pickle_data, num_files = num_files)
    print("Loaded.")

    edge_label, edge_data, node_data, edge_features, PIDs, Times = flatten_lists(el, ed, nd, ef, pid, times)

    data_list = []
    print(f"{len(node_data)} total events in dataset.")

    nSamples = len(node_data)
    nTrain = int(trainRatio * nSamples)
    nVal = int(valRatio * nSamples)

    print("Preparing training and validation split")
    for ev in tqdm(range(len(node_data[:nTrain+nVal]))):
                
        x_np = node_data[ev].T
        #x_norm, _, _ = standardize_data(x_np)
        
        # Create torch vectors from the numpy arrays
        x = torch.from_numpy(x_np)
        x = torch.nan_to_num(x, nan=0.0)
        
        e_label = torch.from_numpy(edge_label[ev])
        edge_index = torch.from_numpy(edge_data[ev])
        e_features = torch.from_numpy(edge_features[ev])
        e_PIDs = torch.from_numpy(PIDs[ev])
        e_times = torch.from_numpy(Times[ev])

        data = Data(x=x, num_nodes=torch.tensor(x.shape[0]),
                    edge_index=edge_index, edge_label=e_label, 
                    edge_features=e_features)
        
        # This graph is directed.
        #print(f"data is directed: {data.is_directed()}")
        data_list.append(data)

    # The test split is not normalized and is stored as a list
    test_data_list = []
    
    print("Preparing test split (data not preprocessed)")
    for ev in tqdm(range(len(node_data[nTrain+nVal:]))):

        x_np = node_data[ev].T
        # Do not pre-process the test split
        data = [x_np, edge_label[ev], edge_data[ev], edge_features[ev]]
        test_data_list.append(data)


    trainDataset = data_list[:nTrain] # training dataset
    valDataset = data_list[nTrain:]   # validation dataset
    
    # Saves the dataset objects to disk.
    mkdir_p(f'{output_location}')
    torch.save(trainDataset, f'{output_location}/dataTraining.pt')
    torch.save(valDataset, f'{output_location}/dataVal.pt')
    torch.save(test_data_list, f'{output_location}/dataTest.pt')
    print("Done: Saved the training datasets.")


def prepare_data_with_edge_dropping(edges, edges_labels, edge_features, drop_fraction=0.9):
    """
    Prepare data by dropping a fraction of negative edges to handle class imbalance.

    Parameters:
    - edges: List of edges in the graph.
    - edges_labels: List of labels corresponding to the edges.
                    0 for negative edges, 1 for positive edges.
    - edge_features: List of features corresponding to each edge.
    - drop_fraction: Fraction of negative edges to drop. Value between 0 and 1.

    Returns:
    - Filtered edges, labels, and features after dropping a fraction of negative edges.
    """

    # Convert inputs to numpy arrays for easier manipulation
    edges = np.array(edges)
    edges_labels = np.array(edges_labels)
    edge_features = np.array(edge_features)

    # Separate positive and negative edges based on labels
    positive_indices = np.where(edges_labels == 1)[0]
    negative_indices = np.where(edges_labels == 0)[0]

    # Calculate number of negative edges to retain
    num_neg_to_retain = int((1 - drop_fraction) * len(negative_indices))

    # Randomly select a subset of negative edges to retain
    retained_negative_indices = np.random.choice(negative_indices, size=num_neg_to_retain, replace=False)

    # Combine the positive edges with the retained negative edges
    retained_indices = np.concatenate((positive_indices, retained_negative_indices))

    # Shuffle the retained indices to mix positive and negative edges
    np.random.shuffle(retained_indices)

    # Select the retained edges, labels, and features
    edges = edges[retained_indices]
    edges_labels = edges_labels[retained_indices]
    edge_features = edge_features[retained_indices]

    return edges, edges_labels, edge_features

