import csv
import json
import logging
import os
import zipfile
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
from gensim.models import Word2Vec
from stellargraph import StellarGraph
import stellargraph
from stellargraph.data import BiasedRandomWalk

np.random.seed(16)

logger = logging.getLogger(__name__)
_PROJECT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ALEXA_DATA_PATH = os.path.abspath(os.path.join(_PROJECT_PATH, 'alexa_data'))
_MODEL_STORAGE = os.path.join(_PROJECT_PATH, 'models')
_FEATURES_DIR = os.path.join(_PROJECT_PATH, 'data', '{corpus_dir}', 'features')
_CORPUS_PATH = os.path.join(_PROJECT_PATH, 'data', '{corpus_dir}', 'corpus.tsv')
_SPLITS_PATH = os.path.join(_PROJECT_PATH, 'data', '{corpus_dir}', 'splits.json')
_PROCESSED_ALEXA_RESPONSES_URL = 'https://drive.google.com/file/d/1die27CFyizjz1-kQ3ZfTl-ZjL74QCsPr/view?usp=sharing'
_PROCESSED_ALEXA_RESPONSES_FILE = os.path.join(_ALEXA_DATA_PATH, 'processed_alexa_responses.zip')
_NODE_FEATURES_FILE = os.path.join(_ALEXA_DATA_PATH, 'node_features.csv')


def download_processed_alexa_responses_archive():
    """Download processed alexa responses archive.
    In the archive you'll find all 104'905 processed files and saved in JSON format.
    Each file contains a JSON object with the following keys:
    [
        'site',
        'comparison_metrics',
        'similar_sites_by_audience_overlap',
        'top_industry_topics_by_social_engagement',
        'top_keywords_by_traffic',
        'alexa_rank_90_days_trends',
        'keyword_gaps',
        'easy_to_rank_keywords',
        'buyer_keywords',
        'optimization_opportunities',
        'top_social_topics',
        'social_engagement',
        'popular_articles',
        'traffic_sources',
        'referral_sites',
        'top_keywords',
        'audience_overlap',
        'alexa_rank',
        'audience_geography_in_past_30_days',
        'site_metrics'
    ]
    """
    logger.info(f'Downloading "processed_alexa_responses.zip" to {_ALEXA_DATA_PATH}')

    with requests.get(_PROCESSED_ALEXA_RESPONSES_URL, stream=True) as r:
        r.raise_for_status()
        with open(_PROCESSED_ALEXA_RESPONSES_FILE, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return _PROCESSED_ALEXA_RESPONSES_FILE


def generate_note_features(target_sites, target_features):
    if not os.path.exists(_PROCESSED_ALEXA_RESPONSES_FILE):
        raise Exception(f'Please download archive file using {_PROCESSED_ALEXA_RESPONSES_URL} to directory {_ALEXA_DATA_PATH}')

    node_feature = defaultdict(dict)
    with zipfile.ZipFile(_PROCESSED_ALEXA_RESPONSES_FILE, 'r') as zip_fd:
        for site in target_sites:
            with zip_fd.open(f'processed_alexa_responses/{site}.json') as f:
                data = json.load(f)
                for section, features in target_features.items():
                    if data.get(section, {}):
                        node_feature[site][section] = {feature: data[section].get(feature) for feature in features}
                    else:
                        node_feature[site][section] = None

    return node_feature


def generate_node_features_file():
    """Target features:
        alexa_rank -> site_rank
        site_metrics -> daily_pageviews_per_visitors
        site_metrics -> daily_time_on_sites
        site_metrics -> total_sites_linking_ins
        site_metrics -> bounce_rate

        Raw features:
        >>> node_features['cnn.com']
        {
            'alexa_rank': {
                'site_rank': '# 79'
            },
            'site_metrics': {
                'daily_pageviews_per_visitor': '2.26',
                'daily_time_on_site': '4:08',
                'total_sites_linking_in': '153,450',
                'bounce_rate': '52.9%'
            }
        }

        Processed features:
        {
            'alexa_rank': {'site_rank': 79},
            'site_metrics': {
                'daily_pageviews_per_visitor': 2.26,
                'daily_time_on_site': 248,  # daily_time_on_site in seconds
                'total_sites_linking_in': 153450,
                'bounce_rate': 0.529        # bounce_rate in decimal
            }
        }
    """
    target_features = {
        'alexa_rank': ['site_rank',],
        'site_metrics': ['daily_pageviews_per_visitor', 'daily_time_on_site', 'total_sites_linking_in', 'bounce_rate']
    }

    if not os.path.exists(_PROCESSED_ALEXA_RESPONSES_FILE):
        raise Exception(f'Please download archive file using {_PROCESSED_ALEXA_RESPONSES_URL} to directory {_ALEXA_DATA_PATH}')

    with zipfile.ZipFile(_PROCESSED_ALEXA_RESPONSES_FILE, 'r') as zip_fd:
        all_sites = [os.path.basename(site).replace('.json', '')
                     for site in zip_fd.namelist()
                     if site != 'processed_alexa_responses/' and site.endswith('.json') and not site.startswith('__MACOSX/')]

    site_features = generate_note_features(all_sites, target_features)

    for features in site_features.values():
        if features.get('alexa_rank') and features['alexa_rank'].get('site_rank'):
            features['alexa_rank']['site_rank'] = int(features['alexa_rank']['site_rank'].strip('# ').replace(',', ''))

        if features.get('site_metrics'):
            if features['site_metrics'].get('daily_pageviews_per_visitor'):
                features['site_metrics']['daily_pageviews_per_visitor'] = float(features['site_metrics']['daily_pageviews_per_visitor'])

            if features['site_metrics'].get('daily_time_on_site'):
                minutes, seconds = features['site_metrics']['daily_time_on_site'].split(':')
                features['site_metrics']['daily_time_on_site'] = int(minutes) * 60 + int(seconds)

            if features['site_metrics'].get('total_sites_linking_in'):
                features['site_metrics']['total_sites_linking_in'] = int(features['site_metrics']['total_sites_linking_in'].replace(',', ''))

            if features['site_metrics'].get('bounce_rate'):
                features['site_metrics']['bounce_rate'] = float(features['site_metrics']['bounce_rate'].replace('%', '')) / 100

    processed_site_feaures = []
    for site, features in site_features.items():
        alexa_rank = None if features.get('alexa_rank') is None else features['alexa_rank'].get('site_rank')

        if features.get('site_metrics') is None:
            daily_pageviews_per_visitor, daily_time_on_site, total_sites_linking_in, bounce_rate = None, None, None, None
        else:
            daily_pageviews_per_visitor = features['site_metrics'].get('daily_pageviews_per_visitor')
            daily_time_on_site = features['site_metrics'].get('daily_time_on_site')
            total_sites_linking_in = features['site_metrics'].get('total_sites_linking_in')
            bounce_rate = features['site_metrics'].get('bounce_rate')

        processed_site_feaures.append({
            'site': site,
            'alexa_rank': alexa_rank,
            'daily_pageviews_per_visitor': daily_pageviews_per_visitor,
            'daily_time_on_site': daily_time_on_site,
            'total_sites_linking_in': total_sites_linking_in,
            'bounce_rate': bounce_rate,
        })

    with open(_NODE_FEATURES_FILE, 'w') as f:
        writer = csv.DictWriter(f, processed_site_feaures[0].keys())
        writer.writeheader()
        writer.writerows(processed_site_feaures)

    logger.info('Successfully generated node features file:', _NODE_FEATURES_FILE)


def load_node_features():
    if not os.path.exists(_NODE_FEATURES_FILE):
        generate_node_features_file()

    return pd.read_csv(_NODE_FEATURES_FILE)


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    return data


def dump_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_level_data(data_path=None, level=0):
    data = load_json(data_path)

    output = {record['sites']: record for record in data if record['levels'] <= level}
    logger.info((f"Loaded {len(output)} nodes with records level <= {level} and child size:"
                f"{sum([len(record['overlap_sites']) for record in output.values()])}"))

    return output


def load_corpus(data_year='2020'):
    corpus_folder = 'emnlp2018' if data_year == '2018' else 'acl2020'
    corpus_path = _CORPUS_PATH.format(corpus_dir=corpus_folder)
    logger.info(f'Loading corpus for {data_year} with path {corpus_path}')

    with open(corpus_path, 'r') as f:
        corpus = [row for row in csv.DictReader(f, delimiter='\t')]

    return corpus


def load_splits(data_year='2020'):
    splits_dir = 'emnlp2018' if data_year == '2018' else 'acl2020'
    splits_path = _SPLITS_PATH.format(corpus_dir=splits_dir)
    logger.info(f'Loading splits for {data_year} with path {splits_path}')

    return load_json(splits_path)


def load_node2vec_model(model_name):
    return Word2Vec.load(os.path.join(_MODEL_STORAGE, model_name))


def save_node2vec_model(model, model_name):
    os.makedirs(_MODEL_STORAGE, exist_ok=True)

    if model_name in os.listdir(_MODEL_STORAGE):
        raise ValueError(f'Model {model_name} already exists in {_MODEL_STORAGE}!')

    model.save(os.path.join(_MODEL_STORAGE, model_name))

    print(f"Successful save of model: {model_name}!")


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


def create_node2vec_model(G, is_weighted, file_name=None, prefix=None, dimensions=[]):
    """Creates a node2vec model and saves it.

    Args:
        G: Stellar graph instance.
        dimensions: List of integer value that tells in which dimension should the embeddings be
        is_weighted: Boolean value that indicates whenever the graph is weighted or not
        file_name: Name of the file where the model will be saved. Please use the file extention '.model'

    Returns:
        A dictionary of the form {'model_name_1': model_1, 'model_name_2': model_2, ...}
    """
    # TODO Add more checks for the input parameters
    if not file_name:
        weight = 'unweighted' if not is_weighted else 'weighted'
        file_names = [f"{prefix}_{weight}_{dimension}D.model" for dimension in dimensions]
    else:
        file_names = [file_name]

    rw = BiasedRandomWalk(G)

    print("Start creating random walks")
    walks = rw.run(
        nodes=list(G.nodes()),  # root nodes
        length=100,             # maximum length of a random walk
        n=10,                   # number of random walks per root node
        p=0.5,                  # Defines (unnormalized) probability, 1/p, of returning to source node
        q=2.0,                  # Defines (unnormalized) probability, 1/q, for moving away from source node
        weighted=is_weighted,   # for weighted random walks
        seed=42,                # random seed fixed for reproducibility
    )
    print("Number of random walks: {}".format(len(walks)))

    str_walks = [[str(n) for n in walk] for walk in walks]

    models = {}
    for d, model_name in zip(dimensions, file_names):
        print(d, model_name)
        model = Word2Vec(str_walks, vector_size=d, window=5, min_count=0, sg=1, workers=2, seed=42)
        save_node2vec_model(model, model_name)
        models[model_name] = model

    return models


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

    logger.info('Node length:', len(nodes))
    logger.info('Distinct node length:', len(set(nodes)))

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
        logger.info(f'For file "{f}" -> load {len(loaded_data)} records')
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

    logger.info('referral_sites node size:', len(referral_sites_NODES),
                'audience_overlap node size:', len(audience_overlap_sites_NODES))

    return audience_overlap_sites_NODES + referral_sites_NODES


def export_model_as_feature(embedding, name, data_year='2020'):
    corpus_dir = 'acl2020' if data_year == '2020' else 'emnlp2018'
    feature_path = os.path.join(_FEATURES_DIR.format(corpus_dir=corpus_dir), f'{name}.json')
    dump_json(feature_path, embedding)

    return feature_path


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
    logger.info(f'Susseccully save feature to {feature_path}')
