import multiprocessing
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg

# do the TSA logaritmic

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

def traintest_autoreg(r, df, lags, lookahead):
    ok = 1
    rid = r.ruleid
    df = df.fillna(method='bfill')
    for mu in ['minttc', 'medianttc', 'meanttc', 'maxttc']:
        train = df[mu].values[:-lookahead]
        if len(train) < lags:
            train = np.concatenate([np.zeros(lags),train])
            ok = 0
        train = np.log10(train)
        model = AutoReg(train, lags=lags)
        model = model.fit()
        pred = model.predict(start=len(train), end=len(train)+lookahead-1, dynamic=False)
        #print(df)
        #print(r.ruleid, len(df), len(train), len(pred), len(train) + len(pred))
        df['p'+mu] = 10**np.concatenate((train, pred))
    return df

def get_prediction_for_rule(rid, rho, lags, lookback):
    lookahead = 10 * 60 // rho
    r = rules[rules.ruleid == rid].iloc[0]
    cutobs = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
    cutreal = get_real_finished_rules_at(r.min_created, rho, lags, lookback)
    tsobs = get_timeseries_for_rules(r, cutobs, rho, lags, lookback)
    tsreal = get_timeseries_for_rules(r, cutreal, rho, lags, lookback)
    tsobs = traintest_autoreg(r, tsobs, lags, lookahead)
    tsreal = traintest_autoreg(r, tsreal, lags, lookahead)
    tsobs.to_csv('data/timeseries/obs_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
    tsreal.to_csv('data/timeseries/real_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
    obspmin = tsobs.pminttc.values[-1]
    obspmedian = tsobs.pmedianttc.values[-1]
    obspmean = tsobs.pmeanttc.values[-1]
    obspmax = tsobs.pmaxttc.values[-1]
    realpmin = tsreal.pminttc.values[-1]
    realpmedian = tsreal.pmedianttc.values[-1]
    realpmean = tsreal.pmeanttc.values[-1]
    realpmax = tsreal.pmaxttc.values[-1]
    return r.ruleid, r.ttc, obspmin, obspmedian, obspmean, obspmax, realpmin, realpmedian, realpmean, realpmax

# ρ variable, lags y lookback fijos.
rhos = list(range(1,10)) + list(range(10,310,10))
for run in range(30):
    print('RUN: %d'%run)
    indices = np.random.choice(rules[(rules.min_created > dt.datetime(2019,6,9)) & (rules.min_created < dt.datetime(2019,7,1))].index, 100) 
    for rho in rhos:
        print('Calculating predictions for ρ = %d'%rho, end='\r')
        lags = 3600//rho
        lookback = 86400//rho
        rids = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r, rho, lags, lookback) for r in rules.loc[indices].ruleid.values)
        rids = pd.DataFrame(rids, columns=['rid', 'ttc', 'obspmin', 'obspmedian', 'obspmean', 'obspmax', 'realpmin', 'realpmedian', 'realpmean', 'realpmax'])
        rids.to_csv('data/tsa/TSA_predictions_log_june_rho_%d_run_%d.csv'%(rho,run), index=False)

# ρ fijo, lags y lookback variables.
rho = 30
lags = [15, 20, 30, 45, 60, 90, 120, 240]  # lags de 30 segundos
lookback = [60*60*6//rho, 60*60*12//rho, 60*60*24//rho]  # lookback de 6, 12 y 24 horas dividido 30 segundos 
for run in range(30):
    print('RUN: %d'%run)
    indices = np.random.choice(rules[(rules.min_created > dt.datetime(2019,6,9)) & (rules.min_created < dt.datetime(2019,7,30))].index, 100)
    for lag in lags:
        for lb in lookback:
            print('Calculating predictions for ρ=%d lags=%d lookback=%d'%(rho, lag, lb), end='\r')
            rids = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_prediction_for_rule)(r, rho, lag, lb) for r in rules.loc[indices].ruleid.values)
            rids = pd.DataFrame(rids, columns=['rid', 'ttc', 'obspmin', 'obspmedian', 'obspmean', 'obspmax', 'realpmin', 'realpmedian', 'realpmean', 'realpmax'])
            rids.to_csv('data/tsa/TSA_predictions_log_rho_%d_lags_%d_lookback_%d_run_%d.csv'%(rho, lag, lb, run), index=False)

# PLOTING

names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
fogpstats = np.zeros((len(names), 31, 30))
for name in names:
    for rho in rhos:
        for run in range(30): 
            df = pd.read_csv('data/tsa/TSA_predictions_june_rho_%d_run_%d.csv'%(rho, run))
            fogpstats[names.index(name)][rhos.index(rho)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('obspmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June min')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June median')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June mean')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June max')

names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
fogpstats = np.zeros((len(names), 31, 30))
for name in names:
    for rho in rhos:
        for run in range(30): 
            df = pd.read_csv('data/tsa/TSA_predictions_july_rho_%d_run_%d.csv'%(rho, run))
            fogpstats[names.index(name)][rhos.index(rho)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('obspmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July min')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July median')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July mean')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July max')

names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
fogpstats = np.zeros((len(names), 31, 30))
for name in names:
    for rho in rhos:
        for run in range(30): 
            df = pd.read_csv('data/tsa/TSA_predictions_june_rho_%d_run_%d.csv'%(rho, run))
            fogpstats[names.index(name)][rhos.index(rho)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('realpmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June min')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June median')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June mean')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('June max')

names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
fogpstats = np.zeros((len(names), 31, 30))
for name in names:
    for rho in rhos:
        for run in range(30): 
            df = pd.read_csv('data/tsa/TSA_predictions_july_rho_%d_run_%d.csv'%(rho, run))
            fogpstats[names.index(name)][rhos.index(rho)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('realpmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July min')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July median')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July mean')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos[:-1]]); plt.yticks([0,.1,.2,.3]); plt.title('July max')


names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
fogpstats = np.zeros((len(names), 31, 30))
for name in names:
    for rho in rhos:
        for run in range(30): 
            df = pd.read_csv('data/tsa/TSA_predictions_july_rho_%d_run_%d.csv'%(rho, run))
            fogpstats[names.index(name)][rhos.index(rho)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('obspmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('obs min')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('obs median')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('obs mean')
plt.subplots();plt.boxplot(fogpstats[names.index('obspmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('obs max')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmin'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('real min')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmedian'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('real median')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmean'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('real mean')
plt.subplots();plt.boxplot(fogpstats[names.index('realpmax'),:,:].T, showmeans=True);plt.xticks(range(1,31), [str(r) for r in rhos]); plt.yticks([0,.1,.2,.3]); plt.title('real max')


# plot ρ fijo
names = ['obspmin','obspmedian','obspmean','obspmax','realpmin','realpmedian','realpmean','realpmax']
lags = [15, 20, 30, 45, 60, 90, 120, 240]  # lags de 30 segundos
lookback = [60*60*6//rho, 60*60*12//rho, 60*60*24//rho]  # lookback de 6, 12 y 24 horas dividido 30 segundos
fogpstats = np.zeros((len(names), 1, len(lags), len(lookback), 84))
rhos = [30]
for rho in rhos:
    for lag in lags:
        for lb in lookback:
            for run in range(84):
                print(name, 'data/tsa/TSA_predictions_rho_%d_lags_%d_lookback_%d_run_%d.csv'%(rho, lag, lb, run))
                df = pd.read_csv('data/tsa/TSA_predictions_rho_%d_lags_%d_lookback_%d_run_%d.csv'%(rho, lag, lb, run))
                for name in names:
                    fogpstats[names.index(name)][rhos.index(rho)][lags.index(lag)][lookback.index(lb)][run] = fogp(df.ttc, df[name], 0.1)

plt.subplots();plt.boxplot(fogpstats[names.index('obspmedian'),0,:,:,lookback.index(720)].T, showmeans=True);plt.xticks(range(1,9), [str(r) for r in lags]); plt.yticks([0,.1,.2,.3]);
