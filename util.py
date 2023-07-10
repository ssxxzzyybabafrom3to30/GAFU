import os
import os.path as osp
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import argparse
import shutil
import tqdm
import random
from torch import Tensor, transpose, matmul, sigmoid, diag
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

def load_file_comparison(filename, observation_ids, observation_sign, mac_all, file_type, prune):
    max_id = 0
    if observation_ids:
        max_id = max(max_id, int(observation_ids[-1][0][1:]))

    with open(filename, "r") as f_in:
        while True:
            line = f_in.readline().rstrip(" \n")
            if not line:
                break
            if file_type == "path":
                if line.startswith("Start:"):
                    device_id = os.path.basename(filename)[:2]
                    breakpoints = [[float(coor) for coor in item.split(
                        ",")] for item in f_in.readline().rstrip(" \n").split(" ")]
                    timestamps = [
                        int(ts) for ts in f_in.readline().rstrip(" \n").split(" ")]
                    continue
                timestamp, rssi_pairs = line.split(" ", 1)
                ground_truth = interpolate_point(
                    int(timestamp), timestamps, breakpoints)
            elif file_type == "db":
                device_id = os.path.basename(filename)
                coor, rssi_pairs = line.split(" ", 1)
                ground_truth = [float(item) for item in coor.split(",")]
            elif file_type == "new":
                wifi_json = json.loads(line)
                timestamp = wifi_json['sysTimeMs']
                rssi_pairs = ''
                for item in wifi_json['data']:
                    rssi_pairs += str(item['bssid'].replace(':','')) + ',' + str(item['rssi']) + ' '
                rssi_pairs = rssi_pairs.strip(' ')
                ground_truth = [None,None]
                device_id = None
            rssi_dict = {}
            for rssi_pair in rssi_pairs.split(" "):
                mac = rssi_pair.split(",")[0]
                rssi = rssi_pair.split(",")[1]
                # remove virtual mac
                if prune and is_virtual_mac(mac):
                    continue
                rssi_dict[mac] = float(rssi)
                if mac not in mac_all:
                    mac_all.append(mac)
            if rssi_dict:
                observation_ids.append(
                    ["{}{}".format(observation_sign, max_id+1), device_id, ground_truth, rssi_dict])
                max_id += 1

    print("{} loaded".format(filename))
    return observation_ids, mac_all

def find_indices(cur_list, item):
    indices = []
    for idx, value in enumerate(cur_list):
        if value == item:
            indices.append(idx)
    return indices

def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))


def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series


def rssi2weight(offset, rssi):
    return offset + rssi


def is_virtual_mac(mac_addr):
    mac_addr = mac_addr.replace(":", "").upper()
    first_hex = int(mac_addr[0:2], 16)
    return first_hex & 0x02 != 0

def k_virtual(Z, K=6):
    id_dict = {}
    for i1 in range(Z.shape[0]):
        dist_list = []
        sorted_list = []
        for i2 in range(Z.shape[0]):
            dist_list.append(np.dot(Z[i1],Z[i2])/(np.linalg.norm(Z[i1])*np.linalg.norm(Z[i2])))
        sorted_list = dist_list.copy()
        sorted_list.sort(reverse=True)
        for k in range(K+1):
            cur_id = dist_list.index(sorted_list[k])
            if k!=cur_id:
                if i1 not in id_dict.keys():
                    id_dict[i1] = [cur_id]
                else:
                    id_dict[i1].append(cur_id)
    return id_dict

def get_data(edgelist_file, Z=np.zeros((100, 100)), emb_dim=8):
    x_set = set()
    edge_index_array = [[],[]]
    edge_weight_array = []

    f_in = open(edgelist_file, "r")
    lines = f_in.readlines()

    # x_set to list
    for line in lines:
        x_set.add(line.split()[0])
        x_set.add(line.split()[1])
    x_list = list(x_set)

    # x
    x_array = []
    for cur_x in x_list:
        new_x = -1 if cur_x.startswith("i") else 1
        x_array.append([random.uniform(0, 1)*new_x for _ in range(emb_dim)])

    # edge_index and edge_weight
    edge_index_array = [[],[]]
    edge_weight_array = []
    for line in lines:
        edge_index_array[0].extend([x_list.index(line.split()[0]), x_list.index(line.split()[1])])
        edge_index_array[1].extend([x_list.index(line.split()[1]), x_list.index(line.split()[0])])
        edge_weight_array.extend([float(line.split()[2]), float(line.split()[2])])

    # virtual edge, Z from AE
    k_virtual_dict = k_virtual(Z)
    for cur_id in k_virtual_dict.keys():
        for id in k_virtual_dict[cur_id]:
            edge_index_array[0].extend([cur_id, id])
            edge_index_array[1].extend([id, cur_id])
            edge_weight_array.extend([0.5, 0.5])

    f_in.close()

    x = torch.tensor([x_array, x_array], dtype=torch.float)
    edge_index = torch.tensor(edge_index_array, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_array, dtype=torch.float)

    return x_list, Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

def interpolate_point(timestamp, timestamps, breakpoints):
    if timestamp <= timestamps[0]:
        print("timestamp too small: {} <= {}".format(timestamp, timestamps[0]))
        return breakpoints[0]
    if timestamp >= timestamps[-1]:
        print("timestamp too large: {} >= {}".format(
            timestamp, timestamps[-1]))
        return breakpoints[-1]

    for idx in range(len(timestamps)-1):
        if timestamps[idx] <= timestamp <= timestamps[idx+1]:
            return [breakpoints[idx][coor_id] + (timestamp - timestamps[idx]) /
                    (timestamps[idx+1] - timestamps[idx]) *
                    (breakpoints[idx+1][coor_id] - breakpoints[idx][coor_id])
                    for coor_id in [0, 1]]    
