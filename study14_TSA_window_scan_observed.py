import multiprocessing
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg


rules = pd.read_csv('data/xfers_per_rule_2.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
for d in ['min_created', 'min_submitted',
       'min_started', 'min_ended', 'max_created', 'max_submitted',
       'max_started', 'max_ended']:
    rules[d] = pd.to_datetime(rules[d])
rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9
rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)

rules = rules[rules.createdatonce == 1]
# Tell pandas to treat the infinite as NaN
pd.set_option('use_inf_as_na', True)
def re(y, yhat):
    return abs(y - yhat)/y

def fogp(y, yhat, thr):
    e = re(y, yhat)
    return sum((e < thr).astype(int))/len(y)

def get_observed_finished_rules_at(ts, window=60):
    cut = rules[(rules.min_created >= (ts - dt.timedelta(seconds=window))) & (rules.max_ended < ts)]
    return cut

def generate_median_mean_pred_for_rule(rid, periods=60):
    r = rules[rules.ruleid == rid].iloc[0]
    # calculate the real mean of the previous window
    cut = get_observed_finished_rules_at(r.min_created, periods)
    return [rid, r.ttc, cut.ttc.min(), cut.ttc.median(), cut.ttc.mean(), cut.ttc.max()]

fogp_wmin = {}
fogp_wmedian = {}
fogp_wmean = {}
fogp_wmax = {}
windowsizes = list(range(1,10))+list(range(10,100,10)) + list(range(100,1000,100)) + list(range(1000, 5000,1000))
for window in windowsizes:
    fogp_wmin[window] = []
    fogp_wmedian[window] = []
    fogp_wmean[window] = []
    fogp_wmax[window] = []
for run in range(100):
    print('############### RUN  %i ################'%run)
    indices = np.random.choice(rules.index, 300)
    #for window in [15,30,45,60,75,90,105,120]:
    for window in windowsizes:
        print('############### WINDOW  %i ################'%window)
        rids = Parallel(n_jobs=12, backend='multiprocessing')(delayed(generate_median_mean_pred_for_rule)(r, window) for r in rules.loc[indices].ruleid.values)
        rids = np.array(rids)
        rids = pd.DataFrame(rids, columns=['ruleid', 'ttc', 'wmin', 'wmedian', 'wmean', 'wmax'])
        rids.ttc = rids.ttc.astype(float)
        rids.wmin = rids.wmin.astype(float)
        rids.wmedian = rids.wmedian.astype(float)
        rids.wmean = rids.wmean.astype(float)
        rids.wmax = rids.wmax.astype(float)
        fogp_wmin[window].append(fogp(rids.ttc, rids.wmin, 0.1))
        fogp_wmedian[window].append(fogp(rids.ttc, rids.wmedian, 0.1))
        fogp_wmean[window].append(fogp(rids.ttc, rids.wmean, 0.1))
        fogp_wmax[window].append(fogp(rids.ttc, rids.wmax, 0.1))
    #for window in [15,30,45,60,75,90,105,120]:
    for window in windowsizes:
        print('WINDOW: %i'%window)
        print('   wmin:    ', fogp_wmin[window][run])
        print('   wmedian: ', fogp_wmedian[window][run])
        print('   wmean:   ', fogp_wmean[window][run])
        print('   wmax:    ', fogp_wmax[window][run])

pd.DataFrame(fogp_wmin).to_csv('data/fogp_rules_stats_for_observed_min.csv',index=False)
pd.DataFrame(fogp_wmedian).to_csv('data/fogp_rules_stats_for_observed_median.csv',index=False)
pd.DataFrame(fogp_wmean).to_csv('data/fogp_rules_stats_for_observed_mean.csv',index=False)
pd.DataFrame(fogp_wmax).to_csv('data/fogp_rules_stats_for_observed_max.csv',index=False)
