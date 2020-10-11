import multiprocessing
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg
from functions import fogp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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

def create_dataset(dataset, look_back=1, look_ahead=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        b = dataset[i + look_back:i + look_back + look_ahead, 0]
        if len(b)< look_ahead:
            break
        dataY.append(dataset[i + look_back:i + look_back + look_ahead, 0])
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
    return numpy.array(dataX), numpy.array(dataY)

def traintestLSTM(r, df, lags, lookahead):
    rid = r.ruleid
    df = df.fillna(method='bfill')
    losses = {}
    models = []
    for mu in ['medianttc', 'meanttc']:
        print('Training for ', mu, '...')
        train = df[mu].values[:-lookahead]
        zmin = train.min()
        zmax = train.max()
        train = (train - zmin)/zmax
        model = Sequential()
        for i in range(layers-1):
            model.add(LSTM(neurons, input_shape=(1, lags), return_sequences=True))
        model.add(LSTM(neurons, input_shape=(1, lags)))
        model.add(Dense(lookahead))
        model.compile(loss='mean_squared_error', optimizer='adam')
        trainx, trainy = create_dataset(train.reshape(-1,1),lags,lookahead)
        trainx = numpy.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
        loss = model.fit(trainx, trainy, epochs=epochs, batch_size=1)
        models.append(model)
        losses[mu] = loss
        x = train[-lags:]
        x = x.reshape(1,1,lags)
        preds = model.predict(x)
        preds = (preds[0] * zmax) + zmin
        train = (train * zmax) + zmin
        df['p'+mu] = np.concatenate((train, preds))
    return df, losses, models

def predictWithLSTM(r, df, lags, lookahead, models):
    rid = r.ruleid
    df = df.fillna(method='bfill')
    for mu, mindex in zip(['medianttc', 'meanttc'], [0,1]):
        print('Training for ', mu, '...')
        train = df[mu].values[:-lookahead]
        zmin = train.min()
        zmax = train.max()
        train = (train - zmin)/zmax
        trainx, trainy = create_dataset(train.reshape(-1,1),lags,lookahead)
        trainx = numpy.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
        x = train[-lags:]
        x = x.reshape(1,1,lags)
        preds = models[mindex].predict(x)
        preds = (preds[0] * zmax) + zmin
        train = (train * zmax) + zmin
        df['p'+mu] = np.concatenate((train, preds))
    return df



def get_prediction_for_rule_first(rid, rho, lags, lookback):
    lookahead = 8 * 60 // rho
    r = rules[rules.ruleid == rid].iloc[0]
    cutobs = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
    cutreal = get_real_finished_rules_at(r.min_created, rho, lags, lookback)
    tsobs = get_timeseries_for_rules(r, cutobs, rho, lags, lookback)
    tsreal = get_timeseries_for_rules(r, cutreal, rho, lags, lookback)
    tsobs, lossobs, models = traintestLSTM(r, tsobs, lags, lookahead)
#    tsreal, lossreal = traintestLSTM(r, tsreal, lags, lookahead)
#    tsobs.to_csv('data/timeseries/obs_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
#    tsreal.to_csv('data/timeseries/real_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
#    obspmin = tsobs.pminttc.values[-1]
    obspmedian = tsobs.pmedianttc.values[-1]
    obspmean = tsobs.pmeanttc.values[-1]
#    obspmax = tsobs.pmaxttc.values[-1]
#    realpmin = tsreal.pminttc.values[-1]
#    realpmedian = tsreal.pmedianttc.values[-1]
#    realpmean = tsreal.pmeanttc.values[-1]
#    realpmax = tsreal.pmaxttc.values[-1]
    return r.ruleid, r.ttc, obspmedian, obspmean, tsobs, tsreal, lossobs, models

def get_prediction_for_rule(rid, rho, lags, lookback, models):
    lookahead = 8 * 60 // rho
    r = rules[rules.ruleid == rid].iloc[0]
    cutobs = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
    cutreal = get_real_finished_rules_at(r.min_created, rho, lags, lookback)
    tsobs = get_timeseries_for_rules(r, cutobs, rho, lags, lookback)
    tsreal = get_timeseries_for_rules(r, cutreal, rho, lags, lookback)
    tsobs = predictWithLSTM(r, tsobs, lags, lookahead, models)
#    tsreal, lossreal = traintestLSTM(r, tsreal, lags, lookahead)
#    tsobs.to_csv('data/timeseries/obs_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
#    tsreal.to_csv('data/timeseries/real_log%s.csv.bz2'%r.ruleid, index=False, compression='bz2')
#    obspmin = tsobs.pminttc.values[-1]
    obspmedian = tsobs.pmedianttc.values[-1]
    obspmean = tsobs.pmeanttc.values[-1]
#    obspmax = tsobs.pmaxttc.values[-1]
#    realpmin = tsreal.pminttc.values[-1]
#    realpmedian = tsreal.pmedianttc.values[-1]
#    realpmean = tsreal.pmeanttc.values[-1]
#    realpmax = tsreal.pmaxttc.values[-1]
    return r.ruleid, r.ttc, obspmedian, obspmean, tsobs, tsreal


rho = 30
lags = 10
lookback = 480
lookahead = 16
layers = 1
neurons = 128
epochs = 50

rid = 'c25d74b6c99d4d2fa24aac43311e2482'
ruleid, ttc, obspmedian, obspmean, tsobs, tsreal, lossobs, models = get_prediction_for_rule_first(rid, rho, lags, lookback)

plt.subplots()
plt.plot(tsreal.ts,tsreal.medianttc,'-*', label='$\\beta_\mu$')
plt.plot(tsobs.ts,tsobs.medianttc,'-^', label='$\\beta_{\hat{\mu}}$')
plt.plot(tsobs.ts.values[:-(8*60//rho)+1],tsobs.pmedianttc.values[:-(8*60//rho)+1], label='$LSTM(\\rho, \\lambda, \\omega)$ fit data')
plt.plot(tsobs.ts.values[-(8*60//rho):],tsobs.pmedianttc.values[-(8*60//rho):], label='$LSTM(\\rho,  \\lambda, \\omega)$ prediction')
plt.plot(rules[rules.ruleid == ruleid].min_created.values[0],ttc,'*', label='Target Rule TTC')
plt.plot(rules[rules.ruleid == ruleid].min_created.values[0],tsobs.pmedianttc.values[-1],'k*', label='$\hat{\mu}$')
plt.legend()
plt.title('Rule TTC prediction based on $\gamma_{median}(\\rho=%d, \\lambda=%d, \psi=%d,\omega=%d)$ for one rule'%(rho, lags, lookback, lookahead))

rid = '733da9b9d3a24f92abda0ea992ecb472'
ruleid, ttc, obspmedian, obspmean, tsobs, tsreal = get_prediction_for_rule(rid, rho, lags, lookback, models)
plt.subplots()
plt.plot(tsreal.ts,tsreal.medianttc,'-*', label='$\\beta_\mu$')
plt.plot(tsobs.ts,tsobs.medianttc,'-^', label='$\\beta_{\hat{\mu}}$')
plt.plot(tsobs.ts.values[:-(8*60//rho)+1],tsobs.pmedianttc.values[:-(8*60//rho)+1], label='$LSTM(\\rho, \\lambda, \\omega)$ fit data')
plt.plot(tsobs.ts.values[-(8*60//rho):],tsobs.pmedianttc.values[-(8*60//rho):], label='$LSTM(\\rho,  \\lambda, \\omega)$ prediction')
plt.plot(rules[rules.ruleid == ruleid].min_created.values[0],ttc,'*', label='Target Rule TTC')
plt.plot(rules[rules.ruleid == ruleid].min_created.values[0],tsobs.pmedianttc.values[-1],'k*', label='$\hat{\mu}$')
plt.legend()
plt.title('Rule TTC prediction based on $\gamma_{median}(\\rho=%d, \\lambda=%d, \psi=%d,\omega=%d)$ for one rule'%(rho, lags, lookback, lookahead))
