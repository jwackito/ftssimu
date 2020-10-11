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
def  get_finished_rules_at(ts, window=1000*60):
    cut = rules[(rules.min_created > (ts - dt.timedelta(seconds=window))) & (rules.max_ended < (ts - dt.timedelta(seconds=8*60)))]
    return cut

def get_timeseries_for_timestamp(ts, freq='60s'):
    cut = get_finished_rules_at(ts)
    try:
        x = pd.date_range(cut.min_created.min(), cut.min_created.max(), freq=freq)
    except ValueError:
        print('Returning shit for rule %s cause of NaT in get_timeseries_for_timestamp'%cut.ruleid)
        return [ts, [], [], []]
    dates = []
    medianttc = []
    meanttc = []
    for s, e in zip(x,x[1:]):
        cut2 = cut[(cut.min_created > s) & (cut.min_created < e)]
        dates.append(s)
        medianttc.append(cut2.ttc.median())
        meanttc.append(cut2.ttc.mean())
    return [ts, dates, medianttc, meanttc]

def generate_bestposible_timeseries_for_rule(rid, freq='60s'):
    r = rules[rules.ruleid == rid].iloc[0]
    print('Calculating rule %s'%rid)
    ts, dates, medianttc, meanttc = get_timeseries_for_timestamp(r.min_created,freq=freq)
    df = pd.DataFrame(np.array([dates, medianttc, meanttc]).T, columns=['ts', 'medianttc', 'meanttc']).fillna(method='ffill')
    df.to_csv('data/timeseries/%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
    return [rid, r.ttc]

def traintest_autoreg(train, rid):
    ok = 1
    if len(train) < 60:
        train = np.concatenate([np.zeros(60),train])
        ok = 0
    try:
        model = AutoReg(train, lags=60)
        model = model.fit()
    except ValueError:
        pred = [train[-1]]
        print('Returning shit for rule %s cause of train len'%rid)
        ok = 0
    except MissingDataError:
        pred = [train[-1]]
        print('Returning shit for rule %s cause of NaN'%rid)
        ok = 0
    else:
        pred = model.predict(start=len(train), end=len(train)+8, dynamic=False)
    return [pred, ok]

def get_prediction_for_rule(rid):
    #print('Making TSA for rule %s'%rid)
    ts = pd.read_csv('data/timeseries/%s.csv.bz2'%rid)
    pred_median, ok = traintest_autoreg(ts.medianttc.values, rid)
    pred_mean, ok = traintest_autoreg(ts.meanttc.values, rid)
    return [rid, pred_median[-1], pred_mean[-1], ok]

rids = Parallel(n_jobs=12, backend='multiprocessing')(delayed(generate_bestposible_timeseries_for_rule)(r) for r in rules.ruleid.values)
rids = np.array(rids)
rids = pd.DataFrame(rids, columns=['ruleid', 'ttc'])
rids.ttc = rids.ttc.astype(float)
fogp_median = []
fogp_mean = []
for i in range(200):
    preds = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r) for r in rids.ruleid.sample(1000))
    preds = np.array(preds)
    preds = pd.DataFrame(preds, columns=['ruleid', 'pmedian', 'pmean', 'ok'])
    preds.pmedian = preds.pmedian.astype(float)
    preds.pmean = preds.pmean.astype(float)
    preds.ok = preds.ok.astype(int)
    preds = preds[preds.ok == 1]
    preds = preds.merge(rids, on='ruleid')
    fogp_median.append(fogp(preds.ttc, preds.pmedian, 0.1))
    fogp_mean.append(fogp(preds.ttc, preds.pmean, 0.1))
