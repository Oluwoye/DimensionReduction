#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
from itertools import product

import numpy as np
import pandas as pd
from glob import glob


LEARNING_RATES = [0.01, 0.05, 0.1, 0.2]
ITERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
PROJECTIONS = ['LMDS', 'TSNE', 'PCA']
PARAM_GRID = list(product(PROJECTIONS, ITERS, LEARNING_RATES))


def convert_npy_to_sdr_format():
    base_dir = sys.argv[1]
    data_file = 'data_.txt'
    header_file = 'data__header.txt'
    label_file = 'data__labels.txt'
    datasets = glob(base_dir + '/*')
    for d in datasets:
        if '_sdr' in d:
            continue

        print(d)

        np_file = os.path.join(d, 'X.npy')
        np_label_file = os.path.join(d, 'y.npy')

        if os.path.exists(np_file):
            X = np.load(np_file)
        else:
            src_file = glob(os.path.join(d, '*-src.csv'))[0]
            df = pd.read_csv(src_file, sep=';')
            X = df.to_numpy()

        y = None

        if os.path.exists(np_label_file):
            y = np.load(np_label_file)

        src_files = glob(os.path.join(d, '*-labels.csv'))

        if len(src_files) > 0:
            src_file = src_files[0]
            df = pd.read_csv(src_file, sep=';')
            y = df.to_numpy()

        data_path = os.path.join(d, data_file)
        header_path = os.path.join(d, header_file)
        label_path = os.path.join(d, label_file)

        np.savetxt(data_path, X, delimiter='\t')
        f = open(header_path, 'w')
        f.write('{0} {1}'.format(X.shape[0], X.shape[1]))
        f.close()

        if y is not None:
            np.savetxt(label_path, y, delimiter='\t')

        for pg in PARAM_GRID:
            proj, n_iters, lr = pg
            path = '{0}/_sdr/{1}_{2}_{3}_{4}/data'.format(base_dir, os.path.basename(d), proj, n_iters, lr)

            print(path)

            if not os.path.exists(path):
                os.makedirs(path)

            data_link_path = os.path.join(path, data_file)
            header_link_path = os.path.join(path, header_file)
            label_link_path = os.path.join(path, label_file)

            if os.path.islink(data_link_path):
                os.remove(data_link_path)

            if os.path.islink(header_link_path):
                os.remove(header_link_path)

            os.symlink(data_path, data_link_path)
            os.symlink(header_path, header_link_path)

            if y is not None:
                if os.path.islink(label_link_path):
                    os.remove(label_link_path)

                os.symlink(label_path, label_link_path)


if __name__ == "__main__":
    convert_npy_to_sdr_format()
