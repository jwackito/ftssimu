from functions import read_xfers
import multiprocessing
from joblib import Parallel, delayed

def get_xfers_per_rid(rid):
    print(rid)
    return argwhere(xfersruleid == rid)

def get_rules_per_rid(rid):
    print(rid)
    return argwhere(ruleruleid == rid)

rids = pd.read_csv('data/rules-CERN-PROD__BNL-ATLAS.csv')

print('reading rules')
rules = pd.read_csv('data/xfers_per_rule_2.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
for d in ['min_created', 'min_submitted',
       'min_started', 'min_ended', 'max_created', 'max_submitted',
       'max_started', 'max_ended']:
    rules[d] = pd.to_datetime(rules[d])
rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9
rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)
ruleruleid = rules.ruleid.values

print('readin xfers')
xfers = read_xfers('data/xfers_462f4fd2e55e4c9cbcaa710371765563')
xfersruleid = xfers.rule_id.values
print('done')

#idxs = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_xfers_per_rid)(r) for r in rids.ruleid.values)

def plot_rule(r): 
    rtime, qtime, ntime = get_times(r) 
    plt.subplots() 
    plotxfers(xfers) 
    plt.plot(rtime, [31]*len(rtime), 'r', label='Rule Rucio Queue Time') 
    plt.plot(qtime, [32]*len(qtime), 'g', label='Rule FTS Queue Time') 
    plt.plot(ntime, [33]*len(ntime), 'b', label='Rule Network Time')
    plt.legend()

def plotxfers(xf):
    nxfers = len(xf)
    nx = 0
    for x in xf.itertuples():
        rtime, qtime, ntime = get_timex(x)
        plt.plot(rtime, [nx]*len(rtime), color='#1f77b4')
        plt.plot(qtime, [nx]*len(qtime), color='#ff7f0e')
        plt.plot(ntime, [nx]*len(ntime), color='#2ca02c')
        nx += 1
    plt.plot(rtime, [nx]*len(rtime), color='#1f77b4', label='Transfer Rucio Time')
    plt.plot(qtime, [nx]*len(qtime), color='#ff7f0e', label='Transfer FTS Time')
    plt.plot(ntime, [nx]*len(ntime), color='#2ca02c', label='Transfer Network Time')

def get_timex(xfer):
    rtime = pd.date_range(start=xfer.created, end=xfer.submitted,freq='s').tolist()
    qtime = pd.date_range(start=xfer.submitted, end=xfer.started,freq='s').tolist()
    ntime = pd.date_range(start=xfer.started, end=xfer.ended,freq='s').tolist()
    return rtime, qtime, ntime

def get_times(r):
    rtime = pd.date_range(start=r.min_created.values[0], end=r.max_submitted.values[0],freq='s').tolist()
    qtime = pd.date_range(start=r.min_submitted.values[0], end=r.max_started.values[0],freq='s').tolist()
    ntime = pd.date_range(start=r.min_started.values[0], end=r.max_ended.values[0],freq='s').tolist()
    return rtime, qtime, ntime
