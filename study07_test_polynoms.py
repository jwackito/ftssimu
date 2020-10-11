# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers
from scipy.optimize import least_squares
from sklearn.linear_model import RidgeCV,Ridge, LinearRegression
import pickle

#xfers = read_xfers('data/transfers-FTSBNL-20190606-20190731.csv', nrows=10000000)
#
#rules = pd.read_csv('data/xfers_per_rule.csv')
#rules.nxfers = rules.nxfers.astype(int)
#rules.bytes = rules.bytes.astype(float)
#
#xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers

def objective(vars, x, data):
    rate = vars[0]
    overhead = vars[1]
    if overhead > 3:
        overhead = 3
    diskrw_limit = vars[2]
    model = x/((x/rate)+overhead)
    model[model>float(diskrw_limit)] = diskrw_limit
    return data - model

def calculate_polys(xfers):
    polys = []
    np.random.seed(43)
    for i in range(100):
        sample = xfers.sample(500)
        vars = [xfers.RATE.mean(), 1., 100.0]
        rate = 1
        overhead = 1
        diskrw = 1
        out = least_squares(objective, vars, args=(sample.SIZE, sample.RATE),bounds=(0,np.inf))
        rate = out.x[0]
        overhead = out.x[1]
        diskrw = out.x[2]
        polys.append([rate, overhead, diskrw])
    return polys

def make_prediction(size, poly):
    rate = poly[0]
    overhead = poly[1]
    diskrw = poly[2]
    thr = size/((size/rate)+overhead)
    if thr > diskrw:
        thr = diskrw
    return size/thr

def  get_xfers_for_rule(r):         
    return df[df.rule_id == r]

#results = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_xfers_for_rule)(r) for r in list(set(rules[rules.nxfers == 20].ruleid)))

def get_prediction_for_rule(r):
    r = get_xfers_for_rule(r)
    rid = r.rule_id.values[0]
    if (r.created.min() == r.created.max()):
        created_at_once = 1
    else:
        created_at_once = 0
    link_multiplicity = len(set(r.link)) 
    ttc = (r.ended.max() - r.created.min()).total_seconds()
    rtime = (r.submitted.max() - r.created.min()).total_seconds()
    qtime = (r.started.max() - r.submitted.min()).total_seconds()
    ntime = (r.ended.max() - r.started.min()).total_seconds()
    submitted_std = ((r.submitted - r.submitted.min()).astype(int)/10**9).std()
    started_std = ((r.started - r.started.min()).astype(int)/10**9).std()
    dirty = 0
    pred_per_link = []
    for l in list(set(r.link)):
        if str(l) is not 'nan':
            #link_rates = np.nan_to_num(xfers[xfers.link == l].RATE, max(1,xfers[xfers.link == l].RATE.mean()))
            pass
        else:
            dirty = 1
            continue
        poly = polys[np.random.choice(len(polys))]
        rate = poly[0]
        overhead = poly[1]
        diskrw = poly[2]
        X = r[r.link == l]
        X['pred_ttc'] = X.SIZE/((X.SIZE/rate)+overhead)
        X[X.pred_ttc > diskrw] = diskrw
        X['pred_ttc'] = X.SIZE/X['pred_ttc']
        pred_per_link.append(X.pred_ttc.sum())
    pred_mean = np.mean(pred_per_link)
    pred_median = np.median(pred_per_link)
    pred_95p = np.percentile(pred_per_link,95)
    pred_max = np.max(pred_per_link)
    return [rid, created_at_once, dirty, link_multiplicity, ttc, rtime, qtime, ntime, submitted_std, started_std, pred_mean, pred_median, pred_95p, pred_max] 

print('Calculating polynoms...')
polys = calculate_polys(xfers)
print('Done...')

dname = 'data/testpolydata_nxfers20_2019.csv'
df = pd.read_csv(dname)
df['stimes'] = pd.to_datetime(df.submitted)
df.created = pd.to_datetime(df.created)
df.submitted = pd.to_datetime(df.submitted)
df.started = pd.to_datetime(df.started)
df.ended = pd.to_datetime(df.ended)
yte = df.RTIME + df.QTIME + df.NTIME

pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in list(set(df.rule_id)))
#pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in ['a7540cfc06bc450d8e66ce0e9fb382cc'])
df2 = pd.DataFrame(np.array(pred_rules), columns=['rid', 'created_at_once', 'dirty', 'link_multiplicity', 'ttc', 'rtime', 'qtime', 'ntime', 'submitted_std', 'started_std', 'pred_mean', 'pred_median', 'pred_95p', 'pred_max'])
for cname in df2.columns[1:]:
    try:
        df2[cname] = df2[cname].astype(float)
    except ValueError:
        continue
df2.to_csv('data/rule_ttc_predction_study07.csv', index=False)
