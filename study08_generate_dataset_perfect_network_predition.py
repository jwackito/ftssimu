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
    pred_using_rtime = r.RTIME.sum()
    pred_using_qtime = r.QTIME.sum()
    pred_using_ntime = r.NTIME.sum()
    dirty = 0
    for l in list(set(r.link)):
        if l is not np.nan:
            link_rates = np.nan_to_num(xfers[xfers.link == l].RATE, max(1,xfers[xfers.link == l].RATE.mean()))
        else:
            dirty = 1
            continue
    return [rid, created_at_once, dirty, link_multiplicity, ttc, rtime, qtime, ntime, submitted_std, started_std, pred_using_rtime , pred_using_qtime, pred_using_ntime] 

pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in list(set(rules[rules.nxfers == 20].ruleid)))
df = pd.DataFrame(np.array(pred_rules), columns=['rid', 'created_at_once', 'dirty', 'link_multiplicity', 'ttc', 'rtime', 'qtime', 'ntime', 'submitted_std', 'started_std', 'pred_rtime', 'pred_qtime', 'pred_ntime'])
for cname in df.columns[1:]:
    df[cname] = df[cname].astype(float)
df.to_csv('data/rule_ttc_predction_study08.csv', index=False)
