# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers


xfers = read_xfers('data/transfers-FTSBNL-20190606-20190731.csv', nrows=10000000)

rules = pd.read_csv('data/xfers_per_rule.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)

xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers

def  get_xfers_for_rule(r):         
    return xfers[xfers.rule_id == r]

#results = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_xfers_for_rule)(r) for r in list(set(rules[rules.nxfers == 20].ruleid)))

def get_prediction_for_rule(r):
    r = get_xfers_for_rule(r)
    rid = r.rule_id.values[0]
    nxfers = len(r)
    if (r.created.min() == r.created.max()):
        created_at_once = 1
    else:
        created_at_once = 0
    link_multiplicity = len(set(r.link))
    try:
        destination = r.link.values[0].split('__')[1]
    except AttributeError:
        destination = 'nan'

    ttc = (r.ended.max() - r.created.min()).total_seconds()
    rtime = (r.submitted.max() - r.created.min()).total_seconds()
    qtime = (r.started.max() - r.submitted.min()).total_seconds()
    ntime = (r.ended.max() - r.started.min()).total_seconds()
    submitted_avg = ((r.submitted - r.submitted.min()).astype(int)/10**9).mean()
    submitted_std = ((r.submitted - r.submitted.min()).astype(int)/10**9).std()
    started_avg = ((r.started - r.started.min()).astype(int)/10**9).mean()
    started_std = ((r.started - r.started.min()).astype(int)/10**9).std()
    rtimeimg = np.zeros(int((r.ended.max() - r.created.min()).total_seconds()))
    qtimeimg = np.zeros(int((r.ended.max() - r.created.min()).total_seconds()))
    ntimeimg = np.zeros(int((r.ended.max() - r.created.min()).total_seconds()))
    tdelta = r.created.min()
    #for x in r.itertuples():
    #    rtimeimg[int((x.created - tdelta).total_seconds()) : int((x.created - tdelta).total_seconds() + x.RTIME)] += 1
    #    qtimeimg[int((x.submitted - tdelta).total_seconds()) : int((x.submitted - tdelta).total_seconds() + x.QTIME)] += 1
    #    ntimeimg[int((x.started - tdelta).total_seconds()) : int((x.started - tdelta).total_seconds() + x.NTIME)] += 1
    return [rid, destination, nxfers, created_at_once, link_multiplicity, ttc, rtime, qtime, ntime, submitted_avg, submitted_std, started_avg, started_std]#, rtimeimg, qtimeimg, ntimeimg] 
np.random.seed(42)
pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in list(set(rules.sample(3000).ruleid)))
df = pd.DataFrame(np.array(pred_rules), columns=['rid', 'destination', 'nxfers', 'created_at_once', 'link_multiplicity', 'ttc', 'rtime', 'qtime', 'ntime', 'submitted_avg', 'submitted_std', 'started_avg', 'started_std'])
for cname in df.columns[2:]:
    df[cname] = df[cname].astype(float)
df.to_csv('data/rule_ttc_predction_study09_randomsample3000.csv', index=False)
