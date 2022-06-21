import sys
import torch
import pandas as pd 
from sage import SAGE, NeighborSampler
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F

#from src.labelfeature_experiment import LabelFeature_Experiment
from src.fakenews import LabelFeature_Experiment

class SAGE_Unsup(LabelFeature_Experiment):
    
    def __init__(self, nodes_file, edges_file, feature_names, **kwargs):
        super(SAGE_Unsup, self).__init__(nodes_file, edges_file, feature_names, **kwargs)
        data = self.data
        self.train_loader = NeighborSampler(data.edge_index, sizes=[-1,25,10], batch_size=5000,
                        shuffle=True, num_nodes=data.num_nodes)
        model = SAGE(in_channels=data.num_features, hidden_channels=self.dim, 
                num_layers=self.num_layers).to(self.device)
                
        self.data = data.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(self):
        self.model.train()
        
        data = self.data
        x, edge_index = data.x, data.edge_index

        total_loss = 0
        for batch_size, n_id, adjs in self.train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
    #         print(batch_size, n_id, adjs)
            adjs = [adj.to(self.device) for adj in adjs]
            self.optimizer.zero_grad()

            out = self.model(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)
            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss) * out.size(0)

        return total_loss / data.num_nodes


    def test(self):
        self.model.eval()
        
        with torch.no_grad():
            out = self.model.full_forward(self.data.x, self.data.edge_index).cpu()

        #print(out[self.data.validation_mask], self.data.y.cpu()[self.data.validation_mask])
            
        clf = LogisticRegression(max_iter=1000000)
        clf.fit(out[self.data.validation_mask], self.data.y.cpu()[self.data.validation_mask])
        # clf.fit(out[self.data.train_mask], self.data.y[self.data.train_mask])

        val_acc = clf.score(out[self.data.validation_mask], self.data.y.cpu()[self.data.validation_mask])
        test_acc = clf.score(out[self.data.test_mask], self.data.y.cpu()[self.data.test_mask])

        return val_acc, test_acc