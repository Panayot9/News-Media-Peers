import os
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from stellargraph import StellarGraph

_PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ALEXA_DATA_PATH = os.path.abspath(os.path.join(_PROJECT_PATH, 'alexa_data'))


def load_json(path):
    with open(path) as f:
        data = json.load(f)

    return data


def load_level_data(data_path=None, level=0):
    with open(data_path, 'r') as f:
        data = json.load(f)

    output = {record['sites']: record for record in data if record['levels'] <= level}
    print((f"Loaded {len(output)} nodes with records level <= {level} and child size:"
           f"{sum([len(record['overlap_sites']) for record in output.values()])}"))

    return output


def create_nodes(lvl_data, edge_type=None):
    nodes = []
    for k in list(lvl_data.keys()):
        if not lvl_data[k]['overlap_sites']:
            el = (k, k, edge_type) if edge_type else (k, k)
            nodes.append(el)
        else:
            for urls in lvl_data[k]['overlap_sites']:
                el = (k, urls['url'], edge_type) if edge_type else (k, urls['url'])
                nodes.append(el)
    return nodes


def create_weighted_nodes(lvl_data):
    nodes = []
    for k in list(lvl_data.keys()):
        if not lvl_data[k]['overlap_sites']:
            nodes.append((k, k, 0.5))
        else:
            for urls in lvl_data[k]['overlap_sites']:
                nodes.append((k, urls['url'], urls.get('overlap_score', 1)))

    return nodes


def create_graph(lvl_data, root):
    edges = []
    for k in lvl_data[root].keys():
        edges.append((root, k))
        for overlap_site in lvl_data[root][k]['score']:
            edges.append((k, overlap_site['url']))

    return edges


def draw_graph(edges=None, graph=None):
    plt.figure(num=None, figsize=(30, 28), dpi=50)

    if graph:
        nx.draw_networkx(graph.to_networkx())
    else:
        nx.draw_networkx(StellarGraph(edges=edges).to_networkx())


def get_referral_sites_edges(data):
    nodes = []

    for base_url, referral_sites in data.items():
        if not referral_sites:
            nodes.append((base_url, base_url))
        else:
            for referral_site, _ in referral_sites:
                if referral_site != base_url:
                    nodes.append((base_url, referral_site))

    print('Node length:', len(nodes))
    print('Distinct node length:', len(set(nodes)))

    return set(nodes)


def combined_nodes_referral_sites_audience_overlap(data_year='2020', level=1, add_edge_type=False):
    if data_year == '2018':
        referral_sites_files = [
            'modified_corpus_2018_referral_sites.json',
            'modified_corpus_2018_referral_sites_level_1.json',
            'modified_corpus_2018_referral_sites_level_2.json',
            'modified_corpus_2018_referral_sites_level_3.json'
        ]

        audience_overlap_scrapping_file = 'corpus_2018_audience_overlap_sites_scrapping_result.json'
    elif data_year == '2020':
        referral_sites_files = [
            'corpus_2020_referral_sites.json',
            'corpus_2020_referral_sites_level_1.json',
            'corpus_2020_referral_sites_level_2.json',
            'corpus_2020_referral_sites_level_3.json',
        ]

        audience_overlap_scrapping_file = 'corpus_2020_audience_overlap_sites_scrapping_result.json'
    else:
        raise ValueError('Incorrect argument "data_year" should be "2018" or "2020"!')

    referral_sites = {}

    for f in referral_sites_files[:level + 1]:
        loaded_data = load_json(os.path.join(_ALEXA_DATA_PATH, f))
        print(f'For file "{f}" -> load {len(loaded_data)} records')
        referral_sites.update(loaded_data)

    referral_sites_NODES = []

    for base_url, referral_sites in referral_sites.items():
        if not referral_sites:
            el = (base_url, base_url, 'referral_site_to') if add_edge_type else (base_url, base_url)
            referral_sites_NODES.append(el)

        for referral_site, _ in referral_sites:
            if referral_site != base_url:
                el = (base_url, referral_site, 'referral_site_to') if add_edge_type else (base_url, referral_site)
                referral_sites_NODES.append(el)

    audience_overlap_sites = load_level_data(os.path.join(_ALEXA_DATA_PATH, audience_overlap_scrapping_file), level=level)

    if add_edge_type:
        audience_overlap_sites_NODES = create_nodes(audience_overlap_sites, edge_type='similar_by_audience_overlap_to')
    else:
        audience_overlap_sites_NODES = create_nodes(audience_overlap_sites)

    print('referral_sites node size:', len(referral_sites_NODES),
          'audience_overlap node size:', len(audience_overlap_sites_NODES))

    return audience_overlap_sites_NODES + referral_sites_NODES
