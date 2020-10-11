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

results = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_xfers_for_rule)(r) for r in list(set(rules[rules.nxfers == 20].ruleid)))

def get_prediction_for_rule(r):
    rid = r.rule_id.values[0]
    if (r.created.min() == r.created.max()):
        created_at_once = True
    else:
        created_at_once = False
    real_ttc = (r.ended.max() - r.created.min()).total_seconds()
    real_rtime = (r.submitted.max() - r.created.min()).total_seconds()
    real_qtime = (r.started.max() - r.submitted.min()).total_seconds()
    real_ntime = (r.ended.max() - r.started.min()).total_seconds()
    real_max_rate = r.RATE.max()
    link_max_thr = xfers[xfers.link == r.link.values[0]].RATE.max()
    link_mean_thr = xfers[xfers.link == r.link.values[0]].RATE.mean()
    pred_using_max_thr = r.SIZE.sum()/link_max_thr
    pred_using_mean_thr = r.SIZE.sum()/link_mean_thr
    error_ttc_using_max = real_ttc - pred_using_max_thr
    error_ttc_using_mean = real_ttc - pred_using_mean_thr
    error_ntime_using_max = real_ntime - pred_using_max_thr
    error_ntime_using_mean = real_ntime - pred_using_mean_thr
    return [rid, created_at_once, real_ttc, real_rtime, real_qtime, real_ntime, real_max_rate, link_max_thr, link_mean_thr,
            pred_using_max_thr, pred_using_mean_thr, error_ttc_using_max, error_ttc_using_mean, error_ntime_using_max, error_ntime_using_mean] 

pred_rules = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in results[:100])
df = pd.DataFrame(np.array(pred_rules), columns=['rid', 'created_at_once', 'real_ttc', 'real_rtime', 'real_qtime', 'real_ntime', 'real_max_rate', 'link_max_thr', 'link_mean_thr', 'pred_using_max_thr', 'pred_using_mean_thr', 'error_ttc_using_max', 'error_ttc_using_mean', 'error_ntime_using_max', 'error_ntime_using_mean'])
for cname in df.columns()[2:]:
    df[cname] = df[cname].astype(float)

plt.subplots()
plt.plot(df.real_ttc, df.error_ttc_using_max,'.', label='Error predicting with max thr')
plt.plot(df.real_ttc, df.error_ttc_using_mean,'.', label='Error predicting with mean thr')
plt.plot(int(range(df.real_ttc.max())), 'r', ms=1, label='y = x')
plt.title('Error in TTC prediction')
plt.ylabel('real TTC - predition')
plt.xlabel('real TTC in seconds')

df.to_csv('')

plt.subplots()
plt.hist(xfers[xfers.nxfers != 20].SIZE/2**20, bins=10000, label='nxfers != 20')
plt.hist(xfers[xfers.nxfers == 20].SIZE/2**20, bins=10000, label='nxfers == 20')
plt.yscale('log')
plt.xlabel('Size of the file transferred in MiB')
plt.ylabel('freq')
plt.legend()
plt.title('Distribution of the size of the transfers, for rules with nxfers each')

#plt.subplots()
#plt.hist(xfers[xfers.nxfers != 20].QTIME, bins=10000, label='nxfers != 20')
#plt.hist(xfers[xfers.nxfers == 20].QTIME, bins=10000, label='nxfers == 20')
#plt.yscale('log')
#plt.xlabel('Queue time in seconds')
#plt.ylabel('freq')
#plt.legend()
#plt.title('Distribution of the queue time of the transfers, for rules with nxfers each')
#
#plt.subplots()
#plt.hist(xfers[xfers.nxfers != 20].QTIME+xfers[xfers.nxfers != 20].NTIME, bins=10000, label='nxfers != 20')
#plt.hist(xfers[xfers.nxfers == 20].QTIME+xfers[xfers.nxfers == 20].NTIME, bins=10000, label='nxfers == 20')
#plt.yscale('log')
#plt.xlabel('Queue + Ntime time in seconds')
#plt.ylabel('freq')
#plt.legend()
#plt.title('Distribution of the queue+ntime time of the transfers, for rules with nxfers each')
#
#plt.subplots()
#plt.plot(xfers[xfers.nxfers != 20].submitted, xfers[xfers.nxfers != 20].NTIME/xfers[xfers.nxfers != 20].QTIME, '.', label='nxfers != 20')
#plt.plot(xfers[xfers.nxfers == 20].submitted, xfers[xfers.nxfers == 20].NTIME/xfers[xfers.nxfers == 20].QTIME, '.', label='nxfers == 20')
##plt.yscale('log')
#plt.xlabel('ntime/qtime time in seconds')
#plt.ylabel('freq')
#plt.legend()
#plt.title('ntime/qtime time of the transfers, for rules with nxfers each')
#
