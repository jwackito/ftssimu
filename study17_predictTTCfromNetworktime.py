import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV,Ridge, LinearRegression
import pickle

def re(y, yhat):
    return abs(y - yhat)/y

def fogp(y, yhat, thr):
    e = re(y, yhat)

    return sum((e < thr).astype(int))/len(y)

def objective(vars, x, data):
    rate = vars[0]
    overhead = vars[1]
    #if overhead > 3:
    #    overhead = 3
    diskrw_limit = vars[2]
    model = x/((x/rate)+overhead)
    model[model>float(diskrw_limit)] = diskrw_limit
    return data - model

def calculate_polys(xfers):
    polys = []
    np.random.seed(43)
    for i in range(300):
        sample = xfers.sample(500)
        vars = [xfers.RATE.mean(), 1., 100.0]
        rate = 1
        overhead = 1
        diskrw = 1
        try:
            out = least_squares(objective, vars, args=(sample.SIZE, sample.RATE),bounds=(0,np.inf))
            rate = out.x[0]
            overhead = out.x[1]
            diskrw = out.x[2]
        except ValueError:
            print('Warning, returning shit cause least squares doesnt converge')
            rate = xfers.RATE.mean()
            overhead = 1.
            diskrw = 100.
        polys.append([rate, overhead, diskrw])
    return polys

def make_prediction(size, poly):
    rate = poly[0]
    overhead = poly[1]
    diskrw = poly[2]
    thr = size/((size/rate)+overhead)
    try:
        if thr > diskrw:
            thr = diskrw
    except ValueError:
        thr[thr > diskrw] = diskrw
    return thr

def pritiprintpoly(poly):
    print('rate: %0.2fMiB/s\noverhead: %0.2fs\ndisk rw: %0.2fMiB/s'%(poly[0]/2**20, poly[1], poly[2]/2**20))

def generate_log2ticks(lo, hi, step=1, sufix='B/s'):
    tics = [2**x for x in range(lo, hi, step)]
    log2tics = [x for x in range(lo, hi, step)]
    labels = []
    for x in tics:
        if x < 2**10:
            labels.append(str(x) + sufix)
        elif x >= 2**10 and x < 2**20:
            labels.append(str(x/2**10) + 'Ki' + sufix)
        elif x >= 2**20 and x < 2**30:
            labels.append(str(x/2**20) + 'Mi' + sufix)
        elif x >= 2**30 and x < 2**40:
            labels.append(str(x/2**30) + 'Gi' + sufix)
        elif x >= 2**40:
            labels.append(str(x/2**40) + 'Ti' + sufix)
    assert(len(tics) == len(labels)), str(tics) + '\n' + str(labels)
    return log2tics, labels

links = ['CERN-PROD__TRIUMF-LCG2', 'IFIC-LCG2__SMU_HPC', 'BNL-ATLAS__TRIUMF-LCG2', 'CERN-PROD__IN2P3-CC','CERN-PROD__INFN-T1', 'CERN-PROD__NDGF-T1', 'BNL-ATLAS__CERN-PROD', 'CERN-PROD__CERN-PROD', 'UNI-BONN__wuppertalprod', 'CERN-PROD__BNL-ATLAS']
#links = ['CERN-PROD__CERN-PROD']
for link in links:
    print('######## %s #########'%link)
    xfers = read_xfers('data/transfers-20190606-20190731-%s.csv'%link)
    # link & RTIME + QTIME + NTIME & RTIME + NTIME & QTIME + NTIME \\ \cline{1-4}
    print("%s & %0.4f & %0.4f & %0.4f"%(sha512(bytes(str(link),'utf-8')).hexdigest()[-15:],
        fogp(xfers.RTIME+xfers.QTIME+xfers.NTIME, xfers.NTIME ,.1),
        fogp(xfers.RTIME+xfers.NTIME, xfers.NTIME ,.1),
        fogp(xfers.QTIME+xfers.NTIME, xfers.NTIME ,.1),
        ))

