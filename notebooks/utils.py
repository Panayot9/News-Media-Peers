import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from stellargraph import StellarGraph

np.random.seed(16)

_PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ALEXA_DATA_PATH = os.path.abspath(os.path.join(_PROJECT_PATH, 'alexa_data'))
_MODEL_STORAGE = os.path.join(_PROJECT_PATH, 'models')
_FEATURES_DIR = _PROJECT_PATH + '/data/{corpus_dir}/features'
_CORPUS_PATH = _PROJECT_PATH + 'data/{corpus_dir}/corpus.tsv'

def load_json(path):
    with open(path) as f:
        data = json.load(f)

    return data


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_level_data(data_path=None, level=0):
    with open(data_path, 'r') as f:
        data = json.load(f)

    output = {record['sites']: record for record in data if record['levels'] <= level}
    print((f"Loaded {len(output)} nodes with records level <= {level} and child size:"
           f"{sum([len(record['overlap_sites']) for record in output.values()])}"))

    return output


def load_corpus(data_year):
    corpus_dir = 'emnlp2018' if data_year == '2018' else 'acl2020'
    corpus_path = _CORPUS_PATH.format(corpus_dir)
    print(f'Loading corpus for {data_year} with path {corpus_path}/')

    with open(corpus_path, 'r') as f:
        corpus = json.load(f)

    return corpus

def load_node2vec_model(model_name):
    return Word2Vec.load(os.path.join(_MODEL_STORAGE, model_name))


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


class ModelWrapper:
    def __init__(self, name, embeddings_wv):
        self.name = name
        self.wv = embeddings_wv

    def __str__(self):
        return self.name


def export_node2vec_as_feature(model_name, data_year='2020'):
    model = load_node2vec_model(model_name)

    if data_year == '2020':
        url_mapping = None
        corpus = load_corpus(data_year)
    elif data_year == '2018':
        url_mapping = {
            "conservativeoutfitters.com": "conservativeoutfitters.com-blogs-news",
            "who.int": "who.int-en",
            "themaven.net": "themaven.net-beingliberal",
            "al-monitor.com": "al-monitor.com-pulse-home.html",
            "pri.org": "pri.org-programs-globalpost",
            "mlive.com": "mlive.com-grand-rapids-#-0",
            "pacificresearch.org": "pacificresearch.org-home",
            "telesurtv.net": "telesurtv.net-english",
            "elpais.com": "elpais.com-elpais-inenglish.html",
            "inquisitr.com": "inquisitr.com-news",
            "cato.org": "cato.org-regulation",
            "jpost.com": "jpost.com-Jerusalem-Report",
            "newcenturytimes.com": "newcenturytimes.com",
            "oregonlive.com": "oregonlive.com-#-0",
            "rfa.org": "rfa.org-english",
            "people.com": "people.com-politics",
            "russia-insider.com": "russia-insider.com-en",
            "nola.com": "nola.com-#-0",
            "host.madison.com": "host.madison.com-wsj",
            "conservapedia.com": "conservapedia.com-Main_Page",
            "futureinamerica.com": "futureinamerica.com-news",
            "indymedia.org": "indymedia.org-or-index.shtml",
            "newyorker.com": "newyorker.com-humor-borowitz-report",
            "rt.com": "rt.com-news",
            "westernjournalism.com": "westernjournalism.com-thepoint",
            "scripps.ucsd.edu": "scripps.ucsd.edu-news",
            "citizensunited.org": "citizensunited.org-index.aspx",
            "gallup.com": "gallup.com-home.aspx",
            "news.harvard.edu": "news.harvard.edu-gazette",
            "spin.com": "spin.com-death-and-taxes",
            "itv.com": "itv.com-news",
            "theguardian.com": "theguardian.com-observer",
            "concernedwomen.org": "concernedwomen.org-blog",
            "npr.org": "npr.org-sections-news",
            "yahoo.com": "yahoo.com-news-?ref=gs",
            "zcomm.org": "zcomm.org-zmag",
            "therealnews.com": "therealnews.com-t2"
        }
        corpus = load_corpus(data_year)
    else:
        raise ValueError(f'Invalid data_year parameter {data_year}')

    feature = {}
    for record in corpus:
        site = record['source_url_normalized']
        model_mapping = url_mapping[site] if url_mapping and site in url_mapping else site

        if data_year == '2018' and site in ['newyorker.com', 'westernjournalism.com', 'pri.org', 'mlive.com']:
            feature[site] = model[site].tolist()
        feature[model_mapping] = model[site].tolist()

    corpus_dir = 'emnlp2018' if data_year == '2018' else 'acl2020'
    feature_path = os.path.join(_FEATURES_DIR.format(corpus_dir), model_name.replace('.model', '.json'))
    dump_json(feature_path, feature)
    print(f'Susseccully save feature to {feature_path}')
