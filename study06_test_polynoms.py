# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers
from sklearn.linear_model import RidgeCV,Ridge, LinearRegression
import pickle

#xfers = read_xfers('data/transfers-FTSBNL-20190606-20190731.csv', nrows=10000000)
#
#rules = pd.read_csv('data/xfers_per_rule.csv')
#rules.nxfers = rules.nxfers.astype(int)
#rules.bytes = rules.bytes.astype(float)
#
#xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers

# data/polymodels3/model_log1p_polydeg1_lss_trainedwith_link-agnostic_89832__linkqueued_max-linksactive_max-linkmeanthr-retry_count-SIZE-ftsqueued_mean-ftsqueued_median-ftsbqueued_median-ftsqueued_95p__.model 
observables = {}
observables[0] = 'linkqueued_mean'          # mean queued at link
observables[1] = 'linkbqueued_mean'         # mean queued bytes at link 
observables[2] = 'linkqueued_median'        # median queued at link 
observables[3] = 'linkbqueued_median'       # median queued bytes at link
observables[4] = 'linkqueued_95p'           # 95 percentile queued at link
observables[5] = 'linkbqueued_95p'          # 95 percentile queued bytes at link
observables[6] = 'linkqueued_max'           # max queued at link
observables[7] = 'linkbqueued_max'          # max queued bytes at link
observables[8] = 'linksactive_max'          # number of active links at submission
observables[9] = 'linkmeanthr'              # mean thr of the link
observables[10] = 'linkmeanqtime'           # average queue time of the link
observables[11] = 'retry_count'           # average queue time of the link
observables[12] = 'SIZE'           # average queue time of the link
observables[13] = 'ftsqueued_mean'          # mean queued at link
observables[14] = 'ftsbqueued_mean'         # mean queued bytes at link 
observables[14] = 'ftsqueued_median'        # median queued at link 
observables[15] = 'ftsbqueued_median'       # median queued bytes at link
observables[16] = 'ftsqueued_95p'           # 95 percentile queued at link

dname = 'data/testpolydata_nxfers20_2019.csv'
model_name = 'data/polymodels4/model_QTIME_polydeg1_lss_trainedwith_link-agnostic_14294__linkbqueued_mean-linkqueued_median-linkbqueued_95p-linkqueued_max-linkmeanthr-SIZE__.model'
#model_name = 'data/polymodels3/model_polydeg1_lss_trainedwith_link-agnostic_89211__linkqueued_95p-linkbqueued_95p-linkqueued_max-linkbqueued_max-linkmeanthr-linkmeanqtime-SIZE-ftsqueued_mean-ftsbqueued_median__.model'
#model_name = 'data/polymodels3/model_polydeg2_lss_trainedwith_link-agnostic_71731__linkqueued_mean-linkbqueued_mean-linkqueued_max-linkbqueued_max-linkmeanthr-retry_count-SIZE-ftsqueued_mean-ftsbqueued_median__.model'
model = pickle.load(open(model_name,'rb'))
model_columns = model_name.split('__')[1].split('-')

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
        X = pd.DataFrame(r[r.link == l], columns=model_columns).values
        prediction = model.predict(X)
        pred_per_link.append(prediction.sum())
    pred_mean = np.mean(pred_per_link)
    pred_median = np.median(pred_per_link)
    pred_95p = np.percentile(pred_per_link,95)
    pred_max = np.max(pred_per_link)
    return [rid, created_at_once, dirty, link_multiplicity, ttc, rtime, qtime, ntime, submitted_std, started_std, pred_mean, pred_median, pred_95p, pred_max] 

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
df2.to_csv('data/rule_ttc_predction_study06.csv', index=False)
