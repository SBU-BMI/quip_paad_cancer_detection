import os
import numpy as np
import math
import scipy.stats as ss

def low_res(fpath):
    x = np.zeros((10000000,), np.uint32);
    y = np.zeros((10000000,), np.uint32);
    p = np.zeros((10000000,), np.float32);
    n = np.zeros((10000000,), np.float32);

    nline = 0;
    with open(fpath) as f:
        for line in f:
            fields = line.split();
            if (len(fields) == 4):
                x[nline] = int(fields[0]);
                y[nline] = int(fields[1]);
                p[nline] = float(fields[2]);
                n[nline] = float(fields[3]);
                nline += 1;
    x = x[0:nline];
    y = y[0:nline];
    p = p[0:nline];
    n = n[0:nline];

    max_x = np.max(x) + np.min(x);
    x = (ss.rankdata(x, method='dense') - 1).astype(np.uint32);
    y = (ss.rankdata(y, method='dense') - 1).astype(np.uint32);
    step = max_x // (np.max(x) + 1);

    imp = np.zeros((np.max(x)+1, np.max(y)+1), np.float32);
    for it in range(len(x)):
        imp[x[it], y[it]] = p[it];

    imn = np.zeros((np.max(x)+1, np.max(y)+1), np.float32);
    for it in range(len(x)):
        imn[x[it], y[it]] = n[it];

    s = 4   # step from high res to low res
    f = open(fpath + '.low_res', 'w');
    for i in range(imp.shape[0]//s):
        for j in range(imp.shape[1]//s):
            p_val = np.max(imp[i*s:i*s+s, j*s:j*s+s]);
            n_val = np.min(imn[i*s:i*s+s, j*s:j*s+s]);
            f.write('{} {} {} {}\n'.format( \
                int(round((i+0.5)*step*s)), int(round((j+0.5)*step*s)), round(p_val, 6), round(n_val, 6)));
    f.close();


directory = './patch-level-merged/';
for fn in os.listdir(directory):
    fpath = directory + fn;
    if not os.path.isfile(fpath):
        continue;
    low_res(fpath);

