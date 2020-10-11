# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers


#xfersnumbertoanalyze = 100
xfers = read_xfers('data/transfers-FTSBNL-DONE-20190606-20190731.csv', nrows=10000000)
rules = pd.read_csv('data/xfers_per_rule_2.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
for d in ['min_created', 'min_submitted',
       'min_started', 'min_ended', 'max_created', 'max_submitted',
       'max_started', 'max_ended']:
    rules[d] = pd.to_datetime(rules[d])
rules = rules[rules.nxfers ==  xfersnumbertoanalyze]
xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers
xfers = xfers[xfers.nxfers == xfersnumbertoanalyze]
percentile = 5
def  get_xfers_for_rule(r):
    return xfers[xfers.rule_id == r]

def get_prediction_for_rule(r):
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
    predictors = [[0.,0.]]
    ncompleted = 0
    for x in cut.itertuples():
        ncompleted += 1
        predictors.append([(x.ended - cut.created.min()).total_seconds(), ncompleted*100/nxfers])
    xy = np.array(predictors)
    y = xy[:,0]
    x = xy[:,1]
    p = np.polyfit(x[:2], y[:2], 1)
    p = np.poly1d(p)
    p1 = p
    pred_5 = p(100)
    p = np.polyfit(x[:int(nxfers * .5)], y[:int(nxfers * .5)], 1)
    p = np.poly1d(p)
    p2 = p
    pred_50 = p(100)
    p = np.polyfit(x[:int(nxfers * .95)], y[:int(nxfers * .95)], 1)
    p = np.poly1d(p)
    pred_95 = p(100)
    p3 = p
    return [rid, nxfers, created_at_once, link_multiplicity, ttc, rtime, qtime, ntime, pred_5, pred_50, pred_95]

pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in list(set(rules[rules.nxfers > 10].sample(100).ruleid)))
df = pd.DataFrame(np.array(pred_rules), columns=['rid', 'nxfers', 'created_at_once', 'link_multiplicity', 'ttc', 'rtime', 'qtime', 'ntime', 'pred_5', 'pred_50', 'pred_95'])
for cname in df.columns[1:]:
    df[cname] = df[cname].astype(float)
df.to_csv('data/rule_ttc_prediction_study11.csv', index=False)i


def get_prediction_for_rule(r):
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
    predictors = [[0.,0.]]
    ncompleted = 0
    for x in cut.itertuples():
        ncompleted += 1
        predictors.append([(x.ended - cut.created.min()).total_seconds(), ncompleted*100/nxfers])
    xy = np.array(predictors)
    y = xy[:,0]
    x = xy[:,1]
    p = np.polyfit(x[:5], y[:5], 1)
    p = np.poly1d(p)
    p1 = p
    pred_5 = p(100)
    p = np.polyfit(x[:int(nxfers * .5)], y[:int(nxfers * .5)], 1)
    p = np.poly1d(p)
    p2 = p
    pred_50 = p(100)
    p = np.polyfit(x[:int(nxfers * .95)], y[:int(nxfers * .95)], 1)
    p = np.poly1d(p)
    pred_95 = p(100)
    p3 = p
    plt.plot(np.linspace(1,100,nxfers), (cut.ended - cut.created.min()).values.astype(int)/10**9)
    plt.plot(range(1, 101), p1(range(1,101)))
    plt.plot(range(1, 101), p2(range(1,101)))
    plt.plot(range(1, 101), p3(range(1,101)))
    return [rid, nxfers, created_at_once, link_multiplicity, ttc, rtime, qtime, ntime, pred_5, pred_50, pred_95, p1, p2, p3]
