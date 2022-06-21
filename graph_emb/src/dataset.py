import pandas as pd
import networkx as nx
from tqdm import tqdm 
import torch
import random
from torch_geometric.data import Data
import os

class DataProcess():
    def __init__(self):
        super(DataProcess, self).__init__()

    def load_data_from_path(edge_path, node_path):
        edge_df = pd.read_csv(edge_path,usecols=['source', 'target'])
        node_df = pd.read_csv(node_path)
        return edge_df, node_df
    
    def load_hetero_data_from_path(edge_path, node_path):
        edge_df = pd.read_csv(edge_path)
        node_df = pd.read_csv(node_path)
        return edge_df, node_df
    
    def load_nodes_from_path(node_path):
        node_df = pd.read_csv(node_path)
        return node_df

    def load_data(edge_path, node_path):
        orig_path = os.getcwd()
        ls = orig_path.split('/')
        index = ls.index('domain-associations')+1 
        root_path = '/'.join(ls[:index])
        return DataProcess.load_data_from_path(f'{root_path}{edge_path}', f'{root_path}{node_path}')

    def graph_gen(edge_df):
        graph = nx.from_pandas_edgelist(edge_df, 'source', 'target')
        return graph


    def feat_gen(node_df, feats): 
        node_feat = pd.DataFrame({'node_id': node_df['node_id'], 'node' :node_df['node']})
        for i in feats: 
            node_feat[i] = node_df[i]
        node_feat['label'] = node_df['label']
        return node_feat


    def tensor_gen(args, edge_df, node_df): 
        dict_n = {}
        
        #reset indexing for training  
        for i,node in enumerate(tqdm(node_df['node_id'])): 
            dict_n[i] = node
        edg = pd.DataFrame({'source': edge_df['source_idx'], \
            'target': edge_df['target_idx']})
        for i,node in enumerate(tqdm(edg['source'])):
            if node and edg['target'][i] in set(dict_n.values()):
                edg['source'][i] = list(dict_n.keys())\
                [list(dict_n.values()).index(node)]
                edg['target'][i] = list(dict_n.keys())\
                [list(dict_n.values()).index(edg['target'][i])]

        features = torch.tensor(node_df[args.features].values.tolist(), dtype=torch.float)

        edges = torch.tensor(edg.values.tolist(), dtype=torch.long)
        y = torch.tensor(node_df['label'], dtype=torch.long)

        data = Data(x=features, edge_index=edges.t().contiguous(), y=y)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        for i in range (data.num_nodes):
            x = random.uniform(0,10)
            if x<=8:
                data.train_mask[i]=1
            else:
                data.test_mask[i]=1
        return data
        
    def load_data_generic(file_name, *coloumns):
        file_df = pd.read_csv(file_name)
        return file_df[list(coloumns)]