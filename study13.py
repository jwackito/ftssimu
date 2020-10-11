# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers
import numpy as np
from sklearn.linear_model import LinearRegression

xfers = read_xfers('data/transfers-FTSBNL-DONE-20190606-20190731.csv')
rules = pd.read_csv('data/xfers_per_rule_2.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
for d in ['min_created', 'min_submitted',
       'min_started', 'min_ended', 'max_created', 'max_submitted',
       'max_started', 'max_ended']:
    rules[d] = pd.to_datetime(rules[d])
rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)
rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9
idleok_criteria = 0.05
rules['idle_ok'] = ((rules.ridle + rules.qidle + rules.nidle) < (rules.ttc * 0.05)).astype(int)
xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers

def  get_xfers_for_rule(r):
    return xfers[xfers.rule_id == r]

def get_prediction_for_rule(r, run):
    rid = r
    r = get_xfers_for_rule(r)
    if (r.created.min() == r.created.max()):
        created_at_once = 1
    else:
        created_at_once = 0
    link_multiplicity = len(set(r.link))
    nxfers = len(r)
    ttc = (r.ended.max() - r.created.min()).total_seconds()
    rtime = (r.submitted.max() - r.created.min()).total_seconds()
    qtime = (r.started.max() - r.submitted.min()).total_seconds()
    ntime = (r.ended.max() - r.started.min()).total_seconds()
    cut = r.sort_values(by='ended')
    predictors = []
    ncompleted = 0
    for x in cut.itertuples():
        ncompleted += 1
        predictors.append([(x.ended - cut.created.min()).total_seconds(), ncompleted*100/nxfers])
    xy = np.array(predictors)
    y = xy[:,0]
    x = xy[:,1]
    polys = []
    for pointnumbers in range(2,11):
        model = LinearRegression(fit_intercept=True)
        model.fit(x[:pointnumbers].reshape(-1,1), y[:pointnumbers])
        polys.append(model.predict([[100]]))
    return [rid, nxfers, created_at_once, link_multiplicity, ttc, rtime, qtime, ntime, polys[0], polys[1], polys[2], polys[3], polys[4], polys[5], polys[6], polys[7], polys[8], run]


columns = ['rid', 'nxfers', 'created_at_once', 'link_multiplicity', 'ttc', 'rtime', 'qtime', 'ntime', 'pred_2', 'pred_3', 'pred_4', 'pred_5', 'pred_6', 'pred_7', 'pred_8', 'pred_9', 'pred_10', 'run']
df = pd.DataFrame([], columns=columns)
for run_number in range(1000):
    print(run_number)
    pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r, run_number) for r in list(set(rules[(rules.nxfers >= 15)&(rules.createdatonce == 1)].sample(500).ruleid)))
    pred_rules = np.array(pred_rules)
    df = df.append(pd.DataFrame(np.array(pred_rules), columns=columns))
for cname in df.columns[1:]:
    df[cname] = df[cname].astype(float)
df.to_csv('data/rule_ttc_prediction_study13.csv', index=False)
