# -*- coding: utf-8 -*-
import argparse
import csv
import itertools
import json
import logging
import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd
import mlflow
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from datetime import datetime

np.random.seed(16)

import warnings

warnings.filterwarnings("ignore")

# setup the logging environment
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

params_svm = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]

label2int = {
    "fact": {
        "low": 0,
        "mixed": 1,
        "high": 2
    },
    "bias": {
        "left": 0, 'extreme-left': 0,
        "center": 1, 'right-center': 1, 'left-center': 1,
        "right": 2, 'extreme-right': 2
    },
}

int2label = {
    "fact": {0: "low", 1: "mixed", 2: "high"},
    "bias": {0: "left", 1: "center", 2: "right"},
}

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file://{PROJECT_DIR}/mlruns")


def log_params(kwargs):
    """Log the parameters of the experiment to mlflow."""
    params = {
        'dataset': kwargs['dataset'],
        'task': kwargs['task'],
        'type_training': kwargs['type_training'],
        'normalize_features': kwargs['normalize_features'],
        'features': ", ".join(kwargs['features'])
    }
    mlflow.log_params(params)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def load_prediction_file(file_path, kwargs):
    with open(file_path, 'r') as f:
        csv_reader = csv.DictReader(f)
        if kwargs['task'] == 'fact':
            data = {row['source_url']: [float(row['low']), float(row['mixed']), float(row['high'])]
                    for row in csv_reader}
        else:
            data = {row['source_url']: [float(row['left']), float(row['center']), float(row['right'])]
                    for row in csv_reader}
    return data


def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics given the actual and predicted labels.
    Returns the macro-F1 score, the accuracy, the flip error rate and the
    mean absolute error (MAE).
    The flip error rate is the percentage where an instance was predicted
    as the opposite label (i.e., left-vs-right or high-vs-low).
    """
    # calculate macro-f1
    f1 = f1_score(actual, predicted, average='macro') * 100

    # calculate accuracy
    accuracy = accuracy_score(actual, predicted) * 100

    # calculate the flip error rate
    flip_err = sum([1 for i in range(len(actual)) if abs(actual[i] - predicted[i]) > 1]) / len(actual) * 100

    # calculate mean absolute error (mae)
    mae = sum([abs(actual[i] - predicted[i]) for i in range(len(actual))]) / len(actual)
    mae = mae[0] if not isinstance(mae, float) else mae

    mlflow.log_metric('f1', f1)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('flip_error', flip_err)
    mlflow.log_metric('mae', mae)
    return f1, accuracy, flip_err, mae


def train_model(splits: Dict[str, Dict[str, List[str]]],
                features: Dict[str, Dict[str, List[float]]],
                labels: Dict[str, str], **kwargs):
    # create placeholders where predictions will be cumulated over the different folds
    all_urls = []
    actual = np.zeros(len(labels), dtype=np.int)
    predicted = np.zeros(len(labels), dtype=np.int)
    probs = np.zeros((len(labels), kwargs['num_labels']), dtype=np.float)

    i = 0
    num_folds = len(splits)

    logger.info(f"Start training... {kwargs.get('skip_sites', [])}")

    for f in range(num_folds):
        logger.info(f"Fold: {f}")

        # get the training and testing media for the current fold
        urls = {
            "train": [url for url in splits[str(f)]["train"] if url not in kwargs.get('skip_sites', [])],
            "test": [url for url in splits[str(f)]["test"] if url not in kwargs.get('skip_sites', [])],
        }

        all_urls.extend([site for site in splits[str(f)]["test"] if site not in kwargs.get('skip_sites', [])])

        # initialize the features and labels matrices
        X, y = {}, {}

        # concatenate the different features/labels for the training sources
        X["train"] = np.asarray([list(itertools.chain(*[v[url] for _, v in features.items()]))
                                 for url in urls["train"]]).astype("float")
        y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

        # concatenate the different features/labels for the testing sources
        X["test"] = np.asarray([list(itertools.chain(*[v[url] for _, v in features.items()]))
                                for url in urls["test"]]).astype("float")
        y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

        if kwargs['normalize_features']:
            # normalize the features values
            scaler = MinMaxScaler()
            scaler.fit(X["train"])
            X["train"] = scaler.transform(X["train"])
            X["test"] = scaler.transform(X["test"])

        # fine-tune the model
        clf_cv = GridSearchCV(SVC(), scoring="f1_macro", cv=num_folds, n_jobs=4, param_grid=params_svm)
        clf_cv.fit(X["train"], y["train"])

        # train the final classifier using the best parameters during crossvalidation
        clf = SVC(
            kernel=clf_cv.best_estimator_.kernel,
            gamma=clf_cv.best_estimator_.gamma,
            C=clf_cv.best_estimator_.C,
            probability=True
        )
        clf.fit(X["train"], y["train"])

        # generate predictions
        pred = clf.predict(X["test"])

        # generate probabilites
        prob = clf.predict_proba(X["test"])

        # cumulate the actual and predicted labels, and the probabilities over the different folds.  then, move the index
        actual[i: i + y["test"].shape[0]] = y["test"]
        predicted[i: i + y["test"].shape[0]] = pred
        probs[i: i + y["test"].shape[0], :] = prob
        i += y["test"].shape[0]

    # calculate the performance metrics on the whole set of predictions (5 folds all together)
    f1, accuracy, flip_err, mae = calculate_metrics(actual, predicted)

    # map the actual and predicted labels to their categorical format
    predicted = np.array([int2label[kwargs['task']][int(l)] for l in predicted])
    actual = np.array([int2label[kwargs['task']][int(l)] for l in actual])

    # create a dictionary: the keys are the media, and the values are their actual and predicted labels
    # predictions = {all_urls[i]: (actual[i], predicted[i]) for i in range(len(all_urls))}

    # create a dataframe that contains the list of m actual labels, the predictions with probabilities.  then store it in the output directory
    df_out = pd.DataFrame({
        "source_url": all_urls,
        "actual": actual,
        "predicted": predicted,
        int2label[kwargs['task']][0]: probs[:, 0],
        int2label[kwargs['task']][1]: probs[:, 1],
        int2label[kwargs['task']][2]: probs[:, 2],
    })
    columns = ["source_url", "actual", "predicted"] + [int2label[kwargs['task']][i] for i in range(kwargs['num_labels'])]
    df_out.to_csv(os.path.join(kwargs['out_dir'], "predictions.tsv"), index=False, columns=columns)

    return f1, accuracy, flip_err, mae


def train_combined_model(corpus_path: str, splits_file: str, feature_files: Dict[str, str], **kwargs):
    # read the dataset
    df = pd.read_csv(corpus_path, sep="\t")

    if kwargs.get('skip_sites'):
        df = df[~df["source_url_normalized"].isin(kwargs['skip_sites'])]

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    df['labels'] = df[kwargs['task']].apply(lambda x: label2int[kwargs['task']][x])
    labels = dict(df[['source_url_normalized', 'labels']].values.tolist())

    # load the evaluation splits
    splits = load_json(splits_file)

    # create the features dictionary: each key corresponds to a feature type, and its value is the pre-computed features dictionary
    loaded_features = {}
    for feature, feature_file in feature_files.items():
        loaded_features[feature] = load_json(feature_file)

    with mlflow.start_run():
        log_params(kwargs)
        f1, accuracy, flip_err, mae = train_model(splits, loaded_features, labels, num_labels=kwargs['num_labels'],
                                                  task=kwargs['task'], out_dir=kwargs['out_dir'],
                                                  normalize_features=kwargs['normalize_features'],
                                                  skip_sites=kwargs['skip_sites'])

    # write the experiment results in a tabular format
    res = PrettyTable()
    res.add_column("Macro-F1", [f1])
    res.add_column("Accuracy", [accuracy])
    res.add_column("Flip error-rate", [flip_err])
    res.add_column("MAE", [mae])

    # write the experiment summary and outcome into a text file and save it to the output directory
    with open(os.path.join(kwargs['out_dir'], "results.txt"), "w") as f:
        f.write(kwargs['summary'].get_string(title="Experiment Summary") + "\n")
        f.write(res.get_string(title="Results"))

    return f1, accuracy, flip_err, mae


def train_ensemble_model(corpus_path: str, splits_file: str, feature_files: Dict[str, str], **kwargs):
    """Uses the results from previously trained SVM classifier's probabilities for the three classes
    with different features.

    Example:
    Let say you have two features - A, B. To use them in the ensemble training first you need to
    train SVM classifier with each of the features (separately). The model will save for each of
    the records three probabilities for each of the classes:

        |source_url      | actual | predicted | low | mixed |high |
        | -------------- | ------ | --------- | --- | ----- |---- |
        |allthatsfab.com | mixed  | high      | 0.07| 0.22  | 0.69|

    Args:
        corpus_path (str): [description]
        splits_file (str): [description]
        feature_files (Dict[str, str]): [description]
    """
    # read the dataset
    df = pd.read_csv(corpus_path, sep="\t")

    if kwargs.get('skip_sites'):
        df = df[~df["source_url_normalized"].isin(kwargs['skip_sites'])]

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    df['labels'] = df[kwargs['task']].apply(lambda x: label2int[kwargs['task']][x])
    labels = dict(df[['source_url_normalized', 'labels']].values.tolist())

    # load the evaluation splits
    splits = load_json(splits_file)

    loaded_features = {}
    for feature, feature_file in feature_files.items():
        loaded_features[feature] = load_prediction_file(feature_file, kwargs)

    with mlflow.start_run():
        log_params(kwargs)
        f1, accuracy, flip_err, mae = train_model(splits, loaded_features, labels, num_labels=kwargs['num_labels'],
                                                  task=kwargs['task'], out_dir=kwargs['out_dir'],
                                                  normalize_features=kwargs['normalize_features'],
                                                  skip_sites=kwargs['skip_sites'])

    # write the experiment results in a tabular format
    res = PrettyTable()
    res.add_column("Macro-F1", [f1])
    res.add_column("Accuracy", [accuracy])
    res.add_column("Flip error-rate", [flip_err])
    res.add_column("MAE", [mae])

    # write the experiment summary and outcome into a text file and save it to the output directory
    with open(os.path.join(kwargs['out_dir'], "results.txt"), "w") as f:
        f.write(kwargs['summary'].get_string(title="Experiment Summary") + "\n")
        f.write(res.get_string(title="Results"))

    return f1, accuracy, flip_err, mae


def parse_arguments():
    """Parse commadline arguments."""
    parser = argparse.ArgumentParser()

    # Required command-line arguments
    parser.add_argument(
        "-f",
        "--features",
        type=str,
        default="",
        required=True,
        help="the features that will be used in the current experiment (comma-separated)",
    )
    parser.add_argument(
        "-tk",
        "--task",
        type=str,
        default="fact",
        required=True,
        help="the task for which the model is trained: (fact or bias)",
    )

    # Boolean command-line arguments
    parser.add_argument(
        "-cc",
        "--clear_cache",
        action="store_true",
        help="flag to whether the corresponding features file need to be deleted before re-computing",
    )
    parser.add_argument(
        '-nf',
        '--normalize_features',
        action='store_true',
        help='flag whether to normalize input features. In the case of ensemble it\'s better to be false'
    )

    # Other command-line arguments
    parser.add_argument(
        "-ds",
        "--dataset",
        type=str,
        default="acl2020",
        help="the name of the dataset for which we are building the media objects",
    )
    parser.add_argument(
        "-nl",
        "--num_labels",
        type=int,
        default=3,
        help="the number of classes of the given task",
    )

    parser.add_argument(
        "-tt",
        "--type_training",
        type=str,
        default="combine",
        help="Indicates what type of model training to do. Possible values are: 'combine', 'ensemble'",
    )

    parser.add_argument(
        "-s",
        "--skip_sites",
        type=str,
        default="",
        help="the sites to skip when training - isolated sites",
    )
    return vars(parser.parse_args())  # cast to dict


def run_experiment(features: str = "", task: str = "fact", dataset: str = "acl2020", num_labels: int = 3,
                   type_training: str = "combine", clear_cache: bool = True, normalize_features: bool = False,
                   skip_sites = None):
    """Run classifier traning usign input arguments.

    Args:
        features (str): the features that will be used in the current experiment (comma-separated)
        task (str): the task for which the model is trained: (fact or bias)
        dataset (str): the name of the dataset for which we are building the media objects
        num_labels (int): the number of classes of the given task
        type_training (str): Indicates what type of model training to do. Possible values are: 'combine', 'ensemble'
        clear_cache (bool): flag to whether the corresponding features file need to be deleted before re-computing
        normalize_features (bool): flag whether to normalize input features. In the case of ensemble it's better to be false

    Execption:
        ValueError: if the "type_training" is not one of the possible values.
    """
    if not features:
        raise ValueError("No Features are specified")

    if skip_sites:
        skip_sites = sorted([skip_site.strip() for skip_site in skip_sites.split(",")])

    # create the list of features sorted alphabetically
    features = sorted([feature.strip() for feature in features.split(",")])

    # display the experiment summary in a tabular format
    summary = PrettyTable()
    summary.add_column("task", [task])
    summary.add_column("dataset", [dataset])
    summary.add_column("classification_mode", ["single classifier"])
    summary.add_column("type_training", [type_training])
    summary.add_column("normalize_features", [normalize_features])
    summary.add_column("features", [", ".join(features)])
    print(summary)

    corpus_path = os.path.join(PROJECT_DIR, "data", dataset, "corpus.tsv")
    splits_file = os.path.join(PROJECT_DIR, "data", dataset, "splits.json")

    if type_training == "combine":
        # specify the output directory where the results will be stored
        out_dir = os.path.join(PROJECT_DIR, "data", dataset, "results", f"{task}_{','.join(features)}")

        # remove the output directory (if it already exists and args.clear_cache was set to TRUE)
        shutil.rmtree(out_dir) if clear_cache and os.path.exists(out_dir) else None

        # create the output directory
        os.makedirs(out_dir, exist_ok=True)

        feature_files = {feature: os.path.join(PROJECT_DIR, "data", dataset, "features", f"{feature}.json")
                         for feature in features}

        f1, accuracy, flip_err, mae = train_combined_model(corpus_path, splits_file, feature_files,
                                                           task=task, dataset=dataset, type_training=type_training,
                                                           normalize_features=normalize_features, features=features,
                                                           num_labels=num_labels, out_dir=out_dir, summary=summary,
                                                           skip_sites=skip_sites)

    elif type_training == "ensemble":
        now = datetime.now()

        # specify the output directory where the results will be stored
        out_dir = os.path.join(PROJECT_DIR, "data", dataset, 'results',
                               f"ensemble_{task}_{','.join(features)}_{now.strftime('%Y%m%d')}")

        # remove the output directory (if it already exists and args.clear_cache was set to TRUE)
        # shutil.rmtree(out_dir) if args.clear_cache and os.path.exists(out_dir) else None

        # create the output directory
        os.makedirs(out_dir, exist_ok=True)

        result_dir = os.path.join(PROJECT_DIR, "data", dataset, "results")

        features_files = {}
        for feature in features:
            feature_folder = f'{task}_{feature}'
            if feature_folder not in os.listdir(result_dir):
                raise ValueError(F"Feature '{feature_folder}' was not generated.Please run the code in 'combine' type_training for generating it.")

            if 'predictions.tsv' not in os.listdir(os.path.join(result_dir, feature_folder)):
                raise ValueError(f"Missing 'predictions.tsv' file for '{feature_folder}' in {result_dir}")

            features_files[feature] = os.path.join(result_dir, feature_folder, 'predictions.tsv')

        f1, accuracy, flip_err, mae = train_ensemble_model(corpus_path, splits_file, features_files,
                                                           task=task, dataset=dataset, type_training=type_training,
                                                           normalize_features=normalize_features, features=features,
                                                           num_labels=num_labels, out_dir=out_dir, summary=summary,
                                                           skip_sites=skip_sites)
    else:
        raise ValueError(f"Unsupported type_training ('{type_training}'). Supported ones are 'combine' or 'ensemble'!")

    summary.add_column("Macro-F1", [f1])
    summary.add_column("Accuracy", [accuracy])
    summary.add_column("Flip error-rate", [flip_err])
    summary.add_column("MAE", [mae])

    print(summary)


def run_experiments(experiments):
    """Runs the experiments specified in the experiments list."""
    for experiment in experiments:
        run_experiment(**experiment)


if __name__ == "__main__":
    # parse the command-line arguments
    args = parse_arguments()

    run_experiment(**args)
