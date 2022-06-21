import sys
import torch
import pandas as pd 
import numpy
from datetime import date

from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from src.utils_fake import score
from gnn.gcn_sage import Gcn
from src.fakenews import LabelFeature_Experiment


class SAGE_Experiment(LabelFeature_Experiment): 
    def __init__(self, nodes_file, edges_file, feature_names, **kwargs):
        super(SAGE_Experiment, self).__init__(nodes_file, edges_file, feature_names, **kwargs)
    
    def define_batch(self, initial_nodes):
        return NeighborLoader(self.data,
            num_neighbors=[-1,20,20,20],
            batch_size=self.batch_size,
            input_nodes=initial_nodes, # It should be training and testing nodes
            directed=False)
    
    def __prepare_model(self,data) :
        return Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes= 3, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)

    def train(self, model_file, feature_names):
        outer_batches = self.define_batch(self.val_index_tensor)
        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(self.epoch):
            model.train()
            optimizer.zero_grad()
            
            for batch in outer_batches:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                loss = F.nll_loss(out[batch.validation_mask], batch.y[batch.validation_mask])

                loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                val_acc, test_acc = self.__test(model, data, data.validation_mask), self.__test(model, data, data.test_mask)
                print(f'Epoch: {epoch:03d}, Loss {loss:.4f}, Val: {val_acc[0]:.4f}, Test: {test_acc[0]:.4f}')
                self.save_model(model, model_file)
                accs = self.test(model_file, feature_names, raw_directory='../results/raw_results/')
                print('Test Accuracy: {:.4f}'.format(accs[0]), '\tF1: {:.4f}'.format(accs[1]), '\tAUC: {:.4f}'.format(accs[2]))
        return model
    
    @torch.no_grad()
    def __test(self, model, data_test, mask):
        model.eval()
        with torch.no_grad():
            pred = model(data_test.x, data_test.edge_index).argmax(dim=1)
        
        return score(pred[mask], data_test.y[mask])