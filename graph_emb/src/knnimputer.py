from platform import node
import pandas as pd
import random
import sys

class KNNImputer():

    def __init__(self, nodes_df, edges_df, feature_labels, k=5, filter_feature_name='feat_domain'):
        self.nodes_df = nodes_df
        self.feature_labels = feature_labels
        self.k = k
        self.nanset = dict()
        for feature in feature_labels:
            self.nanset[feature] = set(self.nodes_df[self.nodes_df[feature].isna()]['node_id'].values.tolist())
        
        self.domain_dict = self.compute_domaindict(edges_df)


    def compute_domaindict(self, edges_df):

        domain_dict = {}
        for index,edge in edges_df.iterrows():
            domain_dict.setdefault(edge['target'], set()).add(edge['source'])
            domain_dict.setdefault(edge['source'], set()).add(edge['target'])

        return domain_dict

    def impute_data(self):
        for feature_name in self.nanset:
            print(feature_name)
            for node_id in self.nanset[feature_name]:
                domain_list = self.domain_dict[node_id]
                
                domain_list = list(set(domain_list) - self.nanset[feature_name]) # remove dublicates and non value nodes
                if len(domain_list) > 0:
                    # print('Found for', node_id)
                    domain_list = random.choices(domain_list, k=self.k)
                    value =  self.nodes_df[self.nodes_df['node_id'].isin(domain_list)][feature_name].mean()
                    self.nodes_df.loc[self.nodes_df['node_id'] == node_id, feature_name] = value
    
