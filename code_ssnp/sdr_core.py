#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from distutils.spawn import find_executable


class SDR:
    def __init__(self, mode, projection, num_iter, learning_rate, path, command, verbose):
        self.known_projections = {
            'LMDS': 11,
            'MDS': 10,
            'PCA': 14,
            'RandomProjection': 15,
            'TSNE': 17
        }

        self.mode = mode
        self.projection = self.known_projections.get(projection, None)
        self.num_iter = num_iter
        self.learning_rate = learning_rate
        self.path = path
        self.command = command
        self.verbose = verbose

        if self.projection is None:
            raise ValueError('Invalid projection value: %s. Valid values are %s'
                             % (projection, ','.join(self.known_projections.keys())))

    def run(self):
        if not find_executable(self.command):
            raise ValueError('Command %s not found' % self.command)

        cmdline = [self.command, str(self.mode), str(self.projection), str(self.num_iter), str(self.learning_rate),
                   self.path]

        if self.verbose:
            print('#################################################')
            print(' '.join(cmdline))

        rc = subprocess.run(cmdline, universal_newlines=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=86400, check=True)

        if self.verbose:
            print('return code: ', rc.returncode)
            print('stdout:')
            print('_________________________________________________')
            print(rc.stdout)
            print('_________________________________________________')
            print('stderr:')
            print('_________________________________________________')
            print(rc.stderr)
            print('#################################################')

        if rc.returncode != 0:
            raise RuntimeError('Error running SDR')
