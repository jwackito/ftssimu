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

def get_observed_finished_rules_at(t0, rho, lags, lookback):
    cut = rules[(rules.min_created >= (t0 - dt.timedelta(seconds=rho*lookback+rho*lags))) & (rules.max_ended < t0)]
    return cut

def get_real_finished_rules_at(t0, rho, lags, lookback):
    cut = rules[(rules.min_created >= (t0 - dt.timedelta(seconds=rho*lookback+rho*lags))) & (rules.min_created < t0)]
    return cut

def get_timeseries_for_rules(r, cut, rho, lags, lookback):
    try:
        x = pd.date_range(r.min_created - dt.timedelta(seconds=rho*lookback+rho*lags), r.min_created, freq=str(rho)+'s')
    except ValueError as e:
        #print('Returning shit for rule %s cause of NaT in get_timeseries_for_timestamp'%cut.ruleid)
        return [ts, [], [], [], [], []]
    print(r.min_created,end='\r')
    dates = []
    minttc = []
    medianttc = []
    meanttc = []
    maxttc = []
    for s, e in zip(x,x[1:]):
        cut2 = cut[(cut.min_created > s) & (cut.min_created < e)]
        dates.append(s)
        minttc.append(cut2.ttc.min())
        medianttc.append(cut2.ttc.median())
        meanttc.append(cut2.ttc.mean())
        maxttc.append(cut2.ttc.max())
    df = pd.DataFrame(np.array([dates, minttc, medianttc, meanttc, maxttc]).T, columns=['ts', 'minttc', 'medianttc', 'meanttc', 'maxttc']).fillna(method='ffill')
    return df

def predict_beta_hat_mu(r,df, lags, lookahead):
    rid = r.ruleid
    df = df.fillna(method='bfill')
    for mu in ['minttc', 'medianttc', 'meanttc', 'maxttc']:
        train = df[mu].values[:-(lookahead+1)]
        pred = df[mu].values[-(lookahead+1)] 
        preds = [pred]*(lookahead+1)
        df['p'+mu] = np.concatenate((train, preds))
    return df

def real_beta_mu(r, df, lags, lookahead):
    rid = r.ruleid
    df = df.fillna(method='bfill')
    for mu in ['minttc', 'medianttc', 'meanttc', 'maxttc']:
        df['p'+mu] = [df[mu].values.mean()]*len(df) 
    return df



def get_prediction_for_rule(rid, rho, lags, lookback):
    lookahead = 8 * 60 // rho
    r = rules[rules.ruleid == rid].iloc[0]
    cutobs = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
    #cutreal = get_real_finished_rules_at(r.min_created, rho, lags, lookback)
    tsobs = get_timeseries_for_rules(r, cutobs, rho, lags, lookback)
    #tsreal = get_timeseries_for_rules(r, cutreal, rho, lags, lookback)
    tsobs = predict_beta_hat_mu(r, tsobs, lags, lookahead)
    #tsreal = real_beta_mu(r, tsreal, lags, lookahead)
#    tsobs.to_csv('data/timeseries/obs_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
#    tsreal.to_csv('data/timeseries/real_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
    #obspmin = tsobs.pminttc.values[-1]
    obspmedian = tsobs.pmedianttc.values[-1]
    #obspmean = tsobs.pmeanttc.values[-1]
    #obspmax = tsobs.pmaxttc.values[-1]
    #realpmin = tsreal.pminttc.values[-1]
    #realpmedian = tsreal.pmedianttc.values[-1]
    #realpmean = tsreal.pmeanttc.values[-1]
    #realpmax = tsreal.pmaxttc.values[-1]
    #realmin = tsreal.minttc.values[-1]
    #realmedian = tsreal.medianttc.values[-1]
    #realmean = tsreal.meanttc.values[-1]
    #realmax = tsreal.maxttc.values[-1]
    return r.ruleid, r.ttc, obspmedian

print('Creating Test dataset')
startt = dt.datetime(2019,7,11)
endt = dt.datetime(2019,7,29)
st = time.time()
indices = rules.sort_values(by='min_created')[(rules.min_created > startt) & (rules.min_created < endt)].index
ruleids = rules.loc[indices].ruleid.values

rhos = [30]
lags = [45]  # lags de 30 segundos
lookback = [240]  # lookback de 6, 12 y 24 horas dividido 30 segundos
for run in range(1):
    print('RUN: %d'%run)
    #indices = np.random.choice(rules[(rules.min_created > dt.datetime(2019,6,9)) & (rules.min_created < dt.datetime(2019,7,30))].index, 200)
    for rho in rhos:
        for lag in lags:
            for lb in lookback:
                print('Calculating predictions for ρ=%d lags=%d lookback=%d'%(rho, lag, lb), end='\r')
                rids = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r, rho, lag, lb) for r in rules.loc[indices].ruleid.values)
                rids = pd.DataFrame(rids, columns=['rid', 'ttc', 'obspmedian'])
                rids.to_csv('data/tsa/gammamedian_TSA_predictions_rho_%d_lags_%d_lookback_%d_run_%d.csv'%(rho, lag, lb, run), index=False)

