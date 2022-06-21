from os import device_encoding
import sys
import torch
import pandas as pd 
import numpy
from datetime import date

from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import ToUndirected


sys.path.append('../')
from gnn.gcn_sage import Gcn
from gnn.gat2 import GAT

from src.dataset import DataProcess
from src.learning_curve import LearningCurve
from src.utils_fake import score

class LabelFeature_Experiment(): 
    def __init__(self, nodes_file, edges_file, feature_names, **kwargs):
        model_type = kwargs['model_type']
        labelfeature_names=kwargs['labelfeature_names']
        epoch=kwargs['epoch']
        num_layers=kwargs['num_layers'] 
        dim=kwargs['dim']
        batch_size=kwargs['outer_batch_size'] 
        train_percentage=kwargs['train_percentage']
        seed=kwargs['seed']
        experiment_id=kwargs['experiment_id']
        gpu_id=kwargs['gpu_id'] 
        inner_batch_size=kwargs['inner_batch_size']
        extra=kwargs['extra']
        
        self.onehot_labelfeature = (len(labelfeature_names) > 2)
        self.kwargs = kwargs
        
        assert model_type in ['gcn', 'sage', 'gat']

        self.edges_ind, self.nodes_df = DataProcess.load_data_from_path(edges_file, nodes_file)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu_id != -1 else 'cpu')
        if self.device != torch.device('cpu') and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8, device=torch._C._cuda_getDevice())
        
        self.__set_masks(self.nodes_df.shape[0], labelfeature_names, train_percentage, seed, experiment_id)
#         self.__set_masks_belief(self.nodes_df.shape[0], labelfeature_names, extra)

        self.data = self.__prepare_data(feature_names)
             
        self.model_type = model_type
        
        self.num_layers = num_layers
        self.dim = dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.inner_batch_size = inner_batch_size
        self.experiment_id = experiment_id
        self.extra = extra

    
    def __prepare_data(self,feature_names):
        features = torch.tensor(self.nodes_df[feature_names].values.tolist(), dtype=torch.float)
        self.edge_index = torch.tensor(self.edges_ind.values.tolist(), dtype=torch.long).t().contiguous()
        self.y = torch.tensor(list(self.nodes_df['label']), dtype = torch.long)
        data = Data(x=features, edge_index=self.edge_index, y=self.y)
        data.train_mask = self.domain_mask
        data.test_mask = self.test_mask
        data.validation_mask = self.train_mask
        
        if self.kwargs.get('use_syn', False):
            syn_file = self.kwargs['syn_file']
            syn_labels = self.kwargs['syn_labels']
            
            data.x = torch.cat((data.x, torch.tensor(DataProcess.load_nodes_from_path(syn_file)[syn_labels].values.tolist(), dtype=torch.float)), 1)

        if data.is_directed():
            data = ToUndirected()(data)
            print("Data converted to undirected:", not data.is_directed())
        
        print(data)
        return data

    def __prepare_model(self,data) :
        if(self.model_type != 'gat'):
            model = Gcn(num_features=data.num_features, dim=self.dim, 
                        num_classes= 3, num_layers=self.num_layers, 
                        model_type=self.model_type).to(self.device)
        else:
            model = GAT(data.num_features, 3).to(self.device) 
        return model
    
    def __set_masks_belief(self, num_nodes, labelfeature_names, belief_id) :
        self.domain_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.domain_mask[self.nodes_df[self.nodes_df['feat_domain'] == 1].index.values] = True
        self.train_mask = torch.Tensor(self.nodes_df[f'train_mask_{belief_id}']).bool()
        self.test_mask = torch.Tensor(self.nodes_df[f'test_mask_{belief_id}']).bool()

        test_indices = self.test_mask.nonzero().t().contiguous().tolist()[0]
        for col in labelfeature_names:
            self.nodes_df.loc[test_indices,col]=0.5
        
        self.domain_mask[test_indices] = False
        self.val_index_tensor = self.train_mask.nonzero().t().contiguous()[0]
        self.test_index_tensor = self.test_mask.nonzero().t().contiguous()[0]

    def __set_masks(self, num_nodes, labelfeature_names, train_percentage, seed, experiment_id) :
        self.domain_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.domain_mask[self.nodes_df[self.nodes_df['feat_domain'] == 1].index.values] = True
        
        # print(max(numpy.unique(self.nodes_df['label'].values.tolist())))
        max_label = max(numpy.unique(self.nodes_df['label'].values.tolist()))
        max_label = 3 # this is because we only care about  labels 0 and 1, we should remove this line normally
        for i in range(max_label):
            mal_index=self.nodes_df[self.nodes_df['label'] == i].index.values 

            numpy.random.seed(seed)
            numpy.random.shuffle(mal_index) 

            train_mal_count = int(len(mal_index) * train_percentage)

            # Compute train/test mask according to experiment_id
            test_mal_count = len(mal_index) - train_mal_count
            train_start_index = (experiment_id *  test_mal_count) % len(mal_index)
            if train_start_index + train_mal_count <= len(mal_index):
                self.train_mask[mal_index[train_start_index:train_start_index+train_mal_count]] = True
            else:
                self.train_mask[mal_index[train_start_index:]] = True
                self.train_mask[mal_index[:(train_start_index+train_mal_count)%len(mal_index)]] = True

            test_mal_indices = numpy.setdiff1d(mal_index, self.train_mask.nonzero().t().contiguous().tolist()[0])
            self.test_mask[test_mal_indices] = True

        test_indices = self.test_mask.nonzero().t().contiguous().tolist()[0]
        
        if not self.onehot_labelfeature:
            for col in labelfeature_names:
                self.nodes_df.loc[test_indices,col]=0.5
        else:
            for i in range(len(labelfeature_names)-1):
                self.nodes_df.loc[test_indices,labelfeature_names[i]]=0
            self.nodes_df.loc[test_indices,labelfeature_names[len(labelfeature_names)-1]]=1

        # val_index=nodes_df[nodes_df['train_mask'] == 1].index.values
        self.domain_mask[test_indices] = False
        self.val_index_tensor = self.train_mask.nonzero().t().contiguous()[0]
        self.test_index_tensor = self.test_mask.nonzero().t().contiguous()[0]
        
        print('Train', sum(self.train_mask), 'Test', sum(self.test_mask))
        

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def define_batch(self, initial_nodes):
        each_layer_sampling_size = -1
        return NeighborLoader(self.data,
            num_neighbors=[each_layer_sampling_size]*self.num_layers,
            batch_size=self.batch_size,
            input_nodes=initial_nodes, # It should be training and testing nodes
            directed=False)

    def train_batches(self):

        outer_batches = self.define_batch(self.val_index_tensor)
        model = self.__prepare_model(self.data)
        
        data = self.data.to(self.device)
        data.validation_mask[self.val_index_tensor] = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        for epoch in range(self.epoch):
            
            model.train()
            optimizer.zero_grad()
            total_loss,total_correct,total_test = 0,0,0
            
            for outer_batch in outer_batches:
                # print(outer_batch)

                val_index_batch = outer_batch.train_mask.nonzero().t().contiguous().tolist()[0][:outer_batch.batch_size]
                numpy.random.shuffle(val_index_batch) 
                
                batch_features = outer_batch.x
                
                prev_batch = None
                for index_batch in self.batch(val_index_batch, self.inner_batch_size):    

                    if prev_batch is not None:
                        ####### TO ADD LABEL FEATURE TO A NODE  
                        for prev in prev_batch:
                            for i in range(4):
                                batch_features[prev][i] = 0

                            batch_features[prev][outer_batch.y[prev]] = 1
                            
                            outer_batch.validation_mask[prev] = False

                    ###### TO REMOVE LABEL FEATURE FROM A NODE
                    for index in index_batch:
                        
                        batch_features[index][0] = 0 if self.onehot_labelfeature else 0.5
                        batch_features[index][1] = 0 if self.onehot_labelfeature else 0.5
                        batch_features[index][2] = 0 if self.onehot_labelfeature else 0.5
                        
                        if self.onehot_labelfeature:
                            batch_features[index][3] = 1
                        # print(batch_features[index][0], batch_features[index][1])
                        outer_batch.validation_mask[index] = True
                    
                    prev_batch = index_batch
                    outer_batch.x = batch_features

                    outer_batch = outer_batch.to(self.device)
                    out = model(outer_batch.x, outer_batch.edge_index)
#                     out = model(outer_batch.x, outer_batch.edge_index, outer_batch)
                    loss = F.nll_loss(out[outer_batch.validation_mask],outer_batch.y[outer_batch.validation_mask])
                    loss.backward()

                    total_loss += loss
                    total_test += len(index_batch)
                    total_correct += (out[outer_batch.validation_mask].argmax(dim=1) == outer_batch.y[outer_batch.validation_mask]).sum()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print(f"Epoch: {epoch:03d} Loss {total_loss} \t Total Correct {total_correct}/{total_test} \t Val {(total_correct/total_test)}")

#             train_acc, val_acc, test_acc = self.__test(model, data)
#             print(f'Epoch: {epoch:03d}, Loss {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
#                             f'Test: {test_acc:.4f}')
        return model

    def train(self, model_file, feature_names):

        model = self.__prepare_model(self.data)
        data = self.data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_f1 = 0
        best_model = None
        for epoch in range(self.epoch):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.validation_mask], data.y[data.validation_mask])

            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                val_acc, test_acc = self.__test(model, data, data.validation_mask), self.__test(model, data, data.test_mask)
                accs = self.test(model_file, feature_names, raw_directory='../results/raw_results/')
                print("F1: ", accs[1])
                #if accs[1] > best_f1:
                best_f1 = accs[1]
                best_model = model
                self.save_model(model, model_file)
                print(f'Epoch: {epoch:03d}, Loss {loss:.4f}, Val: {val_acc[0]:.4f}, Test: {test_acc[0]:.4f}')
                print('Test Accuracy: {:.4f}'.format(accs[0]), '\tF1: {:.4f}'.format(accs[1]), '\tAUC: {:.4f}'.format(accs[2]))
                
                    
        return best_model
    
    @torch.no_grad()
    def get_model(self, model_file, test_feature_labels):

        data_test = self.__prepare_data(test_feature_labels).to(self.device)
        data_test.test_mask = self.test_mask

        modelLoaded = self.__prepare_model(data_test)
        modelLoaded.load_state_dict(torch.load(model_file))
        modelLoaded.eval()
        
        return modelLoaded, data_test
    
    def save_model(self, model, model_file):
        torch.save(model.state_dict(), model_file)
    
    @torch.no_grad()
    def test(self, model_file, test_feature_labels, raw_directory='./data/raw_results/', save_embedding=False):

        modelLoaded, data_test = self.get_model(model_file, test_feature_labels)
        test_indices = self.test_mask.nonzero().t().contiguous().tolist()[0]

        with torch.no_grad():
            LearningCurve.save_prob_result(f'{self.experiment_id}_{self.extra}', 
                                           torch.exp(modelLoaded(data_test.x, data_test.edge_index)), 
                                           data_test.y, test_indices, 
                                           self.nodes_df['node'].tolist(), 
                                           raw_directory)

        return self.__test(modelLoaded, data_test, data_test.test_mask)
#         return self.__test2(modelLoaded, data_test)

    @torch.no_grad()
    def test3(self, model_file, test_feature_labels, raw_directory='./data/raw_results/', save_embedding=False):

        modelLoaded, data_test = self.get_model(model_file, test_feature_labels)
        test_indices = self.test_mask.nonzero().t().contiguous().tolist()[0]

        with torch.no_grad():
            LearningCurve.save_prob_result(f'{self.experiment_id}_{self.extra}', 
                                           torch.exp(modelLoaded(data_test.x, data_test.edge_index)), 
                                           data_test.y, test_indices, 
                                           self.nodes_df['node'].tolist(), 
                                           raw_directory)

        return self.__test3(modelLoaded, data_test, data_test.test_mask)
    

    @torch.no_grad()
    def __test(self, model, data_test, mask):
        model.eval()
        with torch.no_grad():
            pred = model(data_test.x, data_test.edge_index).argmax(dim=1)
        
        return score(pred[mask], data_test.y[mask])
    
    #to get the embedding
    @torch.no_grad()
    def __test3(self, model, data_test, mask):
        model.eval()
        with torch.no_grad():
            pred = model.forward(data_test.x, data_test.edge_index, None, True)
        
        return pred
    
    @torch.no_grad()
    def __test2(self, model, data_test):
        """
        This test function is for label feature approach, currently not being used
        It can be used if testing nodes have their true labels in their label features
        """
        model.eval()

        total_correct,total_test = 0,0
        outer_batches = self.define_batch(self.test_index_tensor)
           
        for outer_batch in outer_batches:
            print(outer_batch)
            test_index_batch = outer_batch.test_mask.nonzero().t().contiguous().tolist()[0]
            outer_batch.test_mask[test_index_batch] = False 
            test_index_batch = test_index_batch[:outer_batch.batch_size]
            
            features = outer_batch.x
            prev = None  

            for t_index in test_index_batch:    
#                 print(t_index)
                if prev is not None:
                    features[prev][outer_batch.y[prev]] = 0.5
                    features[prev][1-outer_batch.y[prev]] = 0.5
                    outer_batch.test_mask[prev] = False

            #         print(f'Prev node {prev}' )

                prev = t_index

                ###### TO REMOVE LABEL FEATURE FROM A NODE
                features[t_index][0] = 0.5
                features[t_index][1] = 0.5
                outer_batch.test_mask[t_index] = True

                if outer_batch.test_mask.sum()>1:
                    print('Active test size', )
                ##### TEST BATCH
            #     print(f'Number of active masks {numpy.unique(data.test_mask, return_counts = True)}');
            #     print(f'Current node {t_index}')
                #####
                outer_batch.x= features
                outer_batch = outer_batch.to(self.device)
                
                with torch.no_grad():
                    pred = model(outer_batch.x, outer_batch.edge_index, outer_batch).argmax(dim=1)

                correct = (pred[outer_batch.test_mask] == outer_batch.y[outer_batch.test_mask]).sum()
                total_correct += correct
                total_test += 1
        return [0, 0,  float(total_correct/total_test)]