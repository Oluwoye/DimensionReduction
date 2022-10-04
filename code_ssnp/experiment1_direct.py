#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
import warnings
from itertools import product

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from code_ssnp.common import compute_all_metrics

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
time_stamp = int(time.time())

from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFont
from skimage import io
from sklearn.manifold import TSNE
from sklearn.cluster import (AgglomerativeClustering, KMeans)
from sklearn.model_selection import train_test_split
from umap import UMAP
from tqdm import tqdm
from argparse import ArgumentParser

import ae
import metrics
import ssnp
import nnproj


def plot(X, y, figname=None):
    if len(np.unique(y)) <= 10:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = plt.get_cmap('tab20')

    fig, ax = plt.subplots(figsize=(20, 20))

    for cl in np.unique(y):
        ax.scatter(X[y == cl, 0], X[y == cl, 1], c=[cmap(cl)], label=cl, s=20)
        ax.axis('off')

    if figname is not None:
        fig.savefig(figname)

    plt.close('all')
    del fig
    del ax


def cantor_pairing(y1, y2):
    y_res = np.array(
        [int(0.5 * (y1[i] + y2[i]) * (y1[i] + y2[i] + 1) + y2[i]) for i in range(0, len(y2))])
    return y_res


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", dest="mode", default="basic", help="Specifies the mode which directories should"
                                                                           "be used. The options are: basic, tfidf,"
                                                                           " full (default:basic)."
                                                                           " Basic will only run the datasets"
                                                                           "of the original paper by Espadoto, tfidf"
                                                                           " will only add the tfidf datasets and bow"
                                                                           " will add bow and tfidf datasets")
    parser.add_argument("-nj", "--n_jobs", dest="n_jobs", default=-1, help="Specifies how many cores are available."
                                                                           " If a value less than 1 or 1 is given"
                                                                           " (default: -1) no parallelization"
                                                                           " will be used.")
    parser.add_argument("-opt", "--optimization", dest="optimization", default="random", help="Specifies which kind"
                                                                                              "of optimization/"
                                                                                              "exploration will"
                                                                                              " be used."
                                                                                              " Only available values"
                                                                                              " are GRID and RANDOM")
    parser.add_argument("-rp", "--random_permutations", dest="random_permutations", default=100, help="If -opt RANDOM,"
                                                                                                      " than this value"
                                                                                                      " specifies how"
                                                                                                      " many"
                                                                                                      " permutations"
                                                                                                      " will be"
                                                                                                      " investigated")
    args = parser.parse_args()
    return args


def perform_non_parametric_drs(X_train, y_train, D_high, dataset_name, results, n_jobs=4, output_dir=os.getcwd()):
    tsne = TSNE(n_jobs=n_jobs, random_state=420)
    X_tsne = tsne.fit_transform(X_train)
    ump = UMAP(random_state=420)
    X_umap = ump.fit_transform(X_train)
    nnp = nnproj.NNProj(init=TSNE(n_jobs=n_jobs, random_state=420))
    nnp.fit(X_train)
    X_nnp = nnp.transform(X_train)
    D_tsne = metrics.compute_distance_list(X_tsne)
    D_umap = metrics.compute_distance_list(X_umap)
    D_nnp = metrics.compute_distance_list(X_nnp)
    results.append((dataset_name, 'TSNE',) + compute_all_metrics(X_train, X_tsne, D_high, D_tsne, y_train)
                   + (-1, -1, -1, -1))
    results.append((dataset_name, 'UMAP',) + compute_all_metrics(X_train, X_umap, D_high, D_umap, y_train)
                   + (-1, -1, -1, -1))
    results.append((dataset_name, 'NNP',) + compute_all_metrics(X_train, X_nnp, D_high, D_nnp, y_train)
                   + (-1, -1, -1, -1))
    for X_, label in zip([X_umap, X_tsne, X_nnp],
                         ['UMAP', 'TSNE', 'NNP']):
        fname = os.path.join(output_dir, '{0}_{1}.png'.
                             format(dataset_name, label))
        print(fname)
        plot(X_, y_train, fname)

    return results


def main():
    # There are two seeds to account for. The global seed and the operational seed.
    # See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/random_seed.py
    random.seed(420)
    args = get_args()
    mode = str(args.mode).lower()
    n_jobs = int(args.n_jobs) - 1  # -1 for accounting for the main thread
    opt = str(args.optimization).lower()
    rp = int(args.random_permutations)
    verbose = False
    results = []
    output_dir = 'results_direct'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = '../data'
    if mode == "full":
        data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters', '20_newsgroups_bow', '20_newsgroups_tfidf',
                     'ag_news_bow',
                     'ag_news_tfidf', 'hatespeech_bow', 'hatespeech_tfidf', 'imdb_bow', 'imdb_tfidf', 'sms_spam_bow',
                     'sms_spam_tfidf']
    elif mode == "tfidf":
        data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters', '20_newsgroups_tfidf',
                     'ag_news_tfidf', 'hatespeech_tfidf', 'imdb_tfidf', 'sms_spam_tfidf']
    elif mode == "basic":
        data_dirs = ['mnist', 'fashionmnist', 'har', 'reuters']
    else:
        raise ValueError("Unrecognized mode " + mode + " only available options are: basic, tfidf and full"
                                                       " (default: basic).")
    classes_mult_set = [1, 2, 3, 4, 5]
    epochs_set = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    patience_set = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_delta_set = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

    parameter_grid = list(product(epochs_set, classes_mult_set, patience_set, min_delta_set))
    if opt == "grid":
        parameter_set = parameter_grid
    elif opt == "random":
        parameter_set = random.sample(parameter_grid, rp)
    else:
        raise ValueError("Only GRID and RANDOM are permitted values for opt (default RANDOM)")

    for d in data_dirs:
        dataset_name = d

        print("Starting non-parametric-analysis")
        print('------------------------------------------------------')
        print('Dataset: {0}'.format(dataset_name))

        X = np.load(os.path.join(data_dir, d, 'X.npy'))
        y = np.load(os.path.join(data_dir, d, 'y.npy'))

        n_samples = X.shape[0]
        train_size = min(int(n_samples * 0.9), 5000)

        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
        D_high = metrics.compute_distance_list(X_train)
        results = perform_non_parametric_drs(X_train, y_train, D_high, dataset_name, results, n_jobs=n_jobs,
                                             output_dir=output_dir)

    for num_epoch, num_classes_mult, patience, min_delta in tqdm(parameter_set):
        parameter_results = compute_parametrized_layouts(num_classes_mult, data_dir, data_dirs, min_delta, num_epoch,
                                                         output_dir, patience, verbose)
        results.extend(parameter_results)

        df = pd.DataFrame(results, columns=['dataset_name',
                                            'test_name',
                                            'T_train',
                                            'C_train',
                                            'R_train',
                                            'S_train',
                                            'N_train',
                                            'MSE_train',
                                            'ha_train',
                                            'sh_train',
                                            'd_train',
                                            'sdbw_train',
                                            'num_epoch',
                                            'n_cluster',
                                            'patience',
                                            'min_delta'])

        df.to_csv(os.path.join(output_dir, "metrics_" + str(time_stamp) + ".csv"), header=True, index=False)
    # don't plot NNP
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = ['SSNP-KMeans', 'SSNP-AG', 'AE', 'TSNE', 'UMAP', 'SSNP-GT']
    images = glob(output_dir + '/*.png')
    base = 2000
    for d in data_dirs:
        dataset_name = d
        to_paste = []

        for i, label in enumerate(pri_images):
            to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

        img = np.zeros((base, base * 6, 3)).astype('uint8')

        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:, i * base:(i + 1) * base, :] = tmp[:, :, :3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + '/composite_full_{0}.png'.format(dataset_name))

        for i, label in enumerate(pri_images):
            print('/composite_full_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 50)
    pri_images = ['SSNP-KMeans', 'SSNP-AG', 'AE']
    images = glob(output_dir + '/*.png')
    base = 2000
    for d in data_dirs:
        dataset_name = d
        to_paste = []

        for i, label in enumerate(pri_images):
            to_paste += [f for f in images if os.path.basename(f) == '{0}_{1}.png'.format(dataset_name, label)]

        img = np.zeros((base, base * 3, 3)).astype('uint8')

        for i, im in enumerate(to_paste):
            tmp = io.imread(im)
            img[:, i * base:(i + 1) * base, :] = tmp[:, :, :3]

        pimg = Image.fromarray(img)
        pimg.save(output_dir + '/composite_{0}.png'.format(dataset_name))

        for i, label in enumerate(pri_images):
            print('/composite_{0}.png'.format(dataset_name), "{0} {1}".format(dataset_name, label))


def compute_parametrized_layouts(classes_mult, data_dir, data_dirs, min_delta, num_epoch, output_dir,
                                 patience, verbose):
    results = []

    for d in data_dirs:
        dataset_name = d

        print('------------------------------------------------------')
        print('Dataset: {0}'.format(dataset_name))

        X = np.load(os.path.join(data_dir, d, 'X.npy'))
        y = np.load(os.path.join(data_dir, d, 'y.npy'))

        print("X shape: " + str(X.shape))
        print("y shape: " + str(y.shape))
        print(np.unique(y))

        n_classes = len(np.unique(y)) * classes_mult
        print("Using num_clusters: " + str(n_classes))
        n_samples = X.shape[0]

        train_size = min(int(n_samples * 0.9), 5000)

        X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, random_state=420, stratify=y)
        D_high = metrics.compute_distance_list(X_train)

        epochs = num_epoch

        ssnpgt = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        ssnpgt.fit(X_train, y_train)
        X_ssnpgt = ssnpgt.transform(X_train)

        ssnpkm = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        C = KMeans(n_clusters=n_classes)
        y_km = C.fit_predict(X_train)
        ssnpkm.fit(X_train, y_km)
        X_ssnpkm = ssnpkm.transform(X_train)

        ssnpif = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        outlier_model = IsolationForest()
        y_if = outlier_model.fit_predict(X_train)
        y_if = np.array([0 if el == -1 else el for el in y_if])
        ssnpif.fit(X_train, y_if)
        X_ssnpif = ssnpif.transform(X_train)

        ssnpkmif = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        y_res_kmif = cantor_pairing(y_km, y_if)
        ssnpkmif.fit(X_train, y_res_kmif)
        X_ssnpkmif = ssnpkmif.transform(X_train)

        ssnpag = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        C = AgglomerativeClustering(n_clusters=n_classes)
        y_ag = C.fit_predict(X_train)
        ssnpag.fit(X_train, y_ag)
        X_ssnpag = ssnpag.transform(X_train)

        ssnpagif = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        y_res_agif = cantor_pairing(y_ag, y_if)
        ssnpagif.fit(X_train, y_res_agif)
        X_ssnpagif = ssnpagif.transform(X_train)

        ssnpkmagif = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam',
                               bottleneck_activation='linear')
        y_res_kmagif = cantor_pairing(y_res_kmif, y_ag)
        ssnpkmagif.fit(X_train, y_res_kmagif)
        X_ssnpkmagif = ssnpkmagif.transform(X_train)

        ssnplof = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam', bottleneck_activation='linear')
        outlier_model = LocalOutlierFactor()
        y_lof = outlier_model.fit_predict(X_train)
        y_lof = np.array([0 if el == -1 else el for el in y_lof])
        ssnplof.fit(X_train, y_lof)
        X_ssnplof = ssnplof.transform(X_train)

        ssnpkmlof = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam',
                              bottleneck_activation='linear')
        y_res_kmlof = cantor_pairing(y_km, y_lof)
        ssnpkmlof.fit(X_train, y_res_kmlof)
        X_ssnpkmlof = ssnpkmlof.transform(X_train)

        ssnpaglof = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam',
                              bottleneck_activation='linear')
        y_res_aglof = cantor_pairing(y_ag, y_lof)
        ssnpaglof.fit(X_train, y_res_aglof)
        X_ssnpaglof = ssnpaglof.transform(X_train)

        ssnpkmaglof = ssnp.SSNP(epochs=epochs, verbose=verbose, patience=0, opt='adam',
                                bottleneck_activation='linear')
        y_res_kmaglof = cantor_pairing(y_res_kmlof, y_ag)
        ssnpkmaglof.fit(X_train, y_res_kmaglof)
        X_ssnpkmaglof = ssnpkmaglof.transform(X_train)

        aep = ae.AutoencoderProjection(epochs=epochs, verbose=0)
        aep.fit(X_train)
        X_aep = aep.transform(X_train)

        D_ssnpgt = metrics.compute_distance_list(X_ssnpgt)
        D_ssnpkm = metrics.compute_distance_list(X_ssnpkm)
        D_ssnpif = metrics.compute_distance_list(X_ssnpif)
        D_ssnpkmif = metrics.compute_distance_list(X_ssnpkmif)
        D_ssnpag = metrics.compute_distance_list(X_ssnpag)
        D_ssnpagif = metrics.compute_distance_list(X_ssnpagif)
        D_ssnpkmagif = metrics.compute_distance_list(X_ssnpkmagif)
        D_ssnplof = metrics.compute_distance_list(X_ssnplof)
        D_ssnpkmlof = metrics.compute_distance_list(X_ssnpkmlof)
        D_ssnpaglof = metrics.compute_distance_list(X_ssnpaglof)
        D_ssnpkmaglof = metrics.compute_distance_list(X_ssnpkmaglof)
        D_aep = metrics.compute_distance_list(X_aep)

        results.append(
            (dataset_name, 'SSNP-GT',) + compute_all_metrics(X_train, X_ssnpgt, D_high, D_ssnpgt, y_train)
            + (num_epoch, -1, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-KMeans',) + compute_all_metrics(X_train, X_ssnpkm, D_high, D_ssnpkm, y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-IF',) + compute_all_metrics(X_train, X_ssnpif, D_high, D_ssnpif, y_train)
            + (num_epoch, -1, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-KMeans-IF',) + compute_all_metrics(X_train, X_ssnpkmif, D_high, D_ssnpkmif,
                                                                    y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-AG',) + compute_all_metrics(X_train, X_ssnpag, D_high, D_ssnpag, y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-AG-IF',) + compute_all_metrics(X_train, X_ssnpagif, D_high, D_ssnpagif, y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-KM-AG-IF',) + compute_all_metrics(X_train, X_ssnpkmagif, D_high, D_ssnpkmagif,
                                                                   y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-LOF',) + compute_all_metrics(X_train, X_ssnplof, D_high, D_ssnplof, y_train)
            + (num_epoch, -1, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-KMeans-LOF',) + compute_all_metrics(X_train, X_ssnpkmlof, D_high, D_ssnpkmlof,
                                                                     y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-AG-LOF',) + compute_all_metrics(X_train, X_ssnpaglof, D_high, D_ssnpaglof, y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append(
            (dataset_name, 'SSNP-KM-AG-LOF',) + compute_all_metrics(X_train, X_ssnpkmaglof, D_high, D_ssnpkmaglof,
                                                                    y_train)
            + (num_epoch, n_classes, patience, min_delta))
        results.append((dataset_name, 'AE',) + compute_all_metrics(X_train, X_aep, D_high, D_aep, y_train)
                       + (num_epoch, -1, patience, min_delta))

        for X_, label in zip([X_ssnpgt, X_ssnpkm, X_ssnpif, X_ssnpkmif, X_ssnpag, X_ssnpagif, X_ssnpkmagif,
                              X_ssnplof, X_ssnpkmlof, X_ssnpaglof, X_ssnpkmaglof, X_aep],
                             ['SSNP-GT', 'SSNP-KMeans', 'SSNP-IF', 'SSNP-KMeans-IF', 'SSNP-AG', 'SSNP-AG-IF',
                              'SSNP-KM-AG-IF',
                              'SSNP-LOF', 'SSNP-KMeans-LOF', 'SSNP-AG-LOF', 'SSNP-KM-AG-LOF', 'AE']):
            if "KM" in label or "AG" in label:
                fname = os.path.join(output_dir, '{0}_{1}_epochs_{2}_n_cluster_{3}_patience_{4}_min_delta_{5}.png'.
                                     format(dataset_name, label, num_epoch, n_classes, patience, min_delta))
            else:
                fname = os.path.join(output_dir, '{0}_{1}_epochs_{2}_patience_{3}_min_delta_{4}.png'.
                                     format(dataset_name, label, num_epoch, patience, min_delta))
            print(fname)
            plot(X_, y_train, fname)

    return results


if __name__ == '__main__':
    main()
