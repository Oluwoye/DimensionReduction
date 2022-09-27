#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# COMMAND USED: python3 02_run_sdr.py datasets 1 /home/yk/Documents/sdr-nnp-linux/Code/LGCDR_v1/build/sdr
import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from multiprocessing import Process, Queue

from code_ssnp.sdr_core import SDR


def producer(q, input_dir, num_consumers):
    dirs = sorted(glob(input_dir + '/*'))
    total = len(dirs)

    for c, d in enumerate(dirs):
        dname = os.path.basename(d)
        dataset, proj, iters, lr = dname.split('_')

        q.put({'path': os.path.join(d, "data"), 'dataset': dataset, 'proj': proj, 'iters': int(iters), 'lr': float(lr),
               'current': c + 1,
               'total': total, 'stop': False})

    for c in range(num_consumers):
        q.put({'path': None, 'dataset': None, 'proj': None, 'iters': 0, 'lr': 0.0, 'current': 0, 'total': 0,
               'stop': True})


def consumer(i, q, sdr_command):
    print(i, 'Starting worker')

    while True:
        data = q.get()
        path = data['path']
        dataset = data['dataset']
        proj = data['proj']
        iters = data['iters']
        lr = data['lr']
        current = data['current']
        total = data['total']
        stop = data['stop']

        if stop:
            break

        print(i, 'Start {0} - {1} of {2}'.format(path, current, total))
        p = SDR(2, proj, iters, lr, path, sdr_command, True)
        p.run()
        print(i, 'Finish {0} - {1} of {2}'.format(path, current, total))

    print(i, 'Finishing worker')


if __name__ == "__main__":
    q = Queue()
    # datasets 1 /home/yk/Documents/sdr-nnp-linux/Code/LGCDR_v1/build/sdr
    plist = []
    input_dir = sys.argv[1]  # 'datasets'
    num_consumers = int(sys.argv[2])
    sdr_command = sys.argv[3]  # '/home/yk/Documents/sdr-nnp-linux/Code/LGCDR_v1/build/sdr'
    # producer(q, input_dir, 1)
    # consumer(0, q, sdr_command)

    for i in range(num_consumers):
        plist.append(Process(target=consumer, args=(i, q, sdr_command)))

    prod = Process(target=producer, args=(q, input_dir, num_consumers))
    prod.start()

    for i in range(num_consumers):
        plist[i].start()

    for i in range(num_consumers):
        plist[i].join()