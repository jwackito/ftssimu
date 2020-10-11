import multiprocessing
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg
from functions import fogp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, MaxPooling1D, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# do the TSA logaritmic

#rules = pd.read_csv('data/xfers_per_rule_2.csv')
#rules.nxfers = rules.nxfers.astype(int)
#rules.bytes = rules.bytes.astype(float)
#for d in ['min_created', 'min_submitted',
#       'min_started', 'min_ended', 'max_created', 'max_submitted',
#       'max_started', 'max_ended']:
#    rules[d] = pd.to_datetime(rules[d])
#rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9
#rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)
#
#rules = rules[rules.createdatonce == 1]
## Tell pandas to treat the infinite as NaN
#pd.set_option('use_inf_as_na', True)

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
        print('Returning shit for rule %s cause of NaT in get_timeseries_for_timestamp'%cut.ruleid)
        return [ts, [], [], [], [], []]
    dates = []
    sumbytes = []
    sumxfers = []
    minttc = []
    medianttc = []
    meanttc = []
    maxttc = []
    for s, e in zip(x,x[1:]):
        cut2 = cut[(cut.min_created > s) & (cut.min_created < e)]
        dates.append(s)
        sumbytes.append(cut2.bytes.sum())
        sumxfers.append(cut2.nxfers.sum())
        minttc.append(cut2.ttc.min())
        medianttc.append(cut2.ttc.median())
        meanttc.append(cut2.ttc.mean())
        maxttc.append(cut2.ttc.max())
    df = pd.DataFrame(np.array([dates, sumbytes, sumxfers, minttc, medianttc, meanttc, maxttc]).T, columns=['ts', 'bytes', 'nxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc']).fillna(method='ffill').fillna(method='bfill')
    return df

def create_dataset(rids):
    tss = {}
    tss['ts'] = []
    tss['bytes'] = []
    tss['nxfers'] = []
    tss['minttc'] = []
    tss['medianttc'] = []
    tss['meanttc'] = []
    tss['maxttc'] = []
    tss['target'] = []
    i = len(rids)
    for rid in rids:
        r = rules[rules.ruleid == rid].iloc[0]
        cut = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
        ts = get_timeseries_for_rules(r, cut, rho, lags, lookback)
        #print(i, 'len(ts): ',len(ts), end='\r')
        for metric in ['ts', 'bytes', 'nxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc']:
            tss[metric].append(ts[metric].values[:-lookahead])
        tss['target'].append(r.ttc)
        i -= 1

    tss['bytes']       = np.array(tss['bytes']) 
    tss['nxfers']       = np.array(tss['nxfers']) 
    tss['minttc']       = np.array(tss['minttc']) 
    tss['medianttc']    = np.array(tss['medianttc'])
    tss['meanttc']      = np.array(tss['meanttc'])
    tss['maxttc']       = np.array(tss['maxttc'])
    tss['target']       = np.array(tss['target'])
    return tss

def traintestLSTM(ruleids):
    losses = {}
    models = {}
    for mu in ['medianttc', 'meanttc']:
        print('Training for ', mu, '...')
        trainx = np.array([(tss[mu] - tssmu[mu])/tssstd[mu], (tss['bytes'] - tssmu['bytes'])/tssstd['bytes'], (tss['nxfers'] - tssmu['nxfers'])/tssstd['nxfers']])
        #zmin = train.min()
        #zmax = train.max()
        #train = (train - zmin)/zmax
        trainy = (tss['target'] - tssmu['target'])/tssstd['target']
        print(trainx.shape)
        trainx = numpy.reshape(trainx, (trainx.shape[1], trainx.shape[2], 3))
        model = Sequential()
        # from Chollet
        model.add(Conv1D(neurons, 9, input_shape=(None, trainx.shape[-1]),padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(3))
        model.add(Conv1D(neurons//2, 7,padding='same'))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(3))
        model.add(Conv1D(neurons//4, 5,padding='same'))
        model.add(LeakyReLU())
        model.add(LSTM(neurons, dropout=.1, recurrent_dropout=.50))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        print('TRAINING ######################')
        loss = model.fit(trainx, trainy, epochs=epochs, batch_size=10, shuffle=True)
        models[mu] = model
        losses[mu] = loss
    return tss, losses, models

def predictWithLSTM(rid):
    r = rules[rules.ruleid == rid].iloc[0]
    tss = create_dataset([rid])
    tss['targetts'] = [r.min_created]
    for mu in ['medianttc', 'meanttc']:
        testx = (tss[mu] - tssmu[mu])/tssstd[mu]
        testx = np.array([(tss[mu] - tssmu[mu])/tssstd[mu], (tss['bytes'] - tssmu['bytes'])/tssstd['bytes'], (tss['nxfers'] - tssmu['nxfers'])/tssstd['nxfers']])
        testy = (tss['target'] - tssmu['target'])/tssstd['target']
        testx = numpy.reshape(testx, (testx.shape[1], testx.shape[2],3))
        testx = testx.reshape(1,lags-lookahead,3)
        preds = models[mu].predict(testx)
        tss['p'+mu] = (preds*tssstd['target'])+tssmu['target']
    return tss

def predict_and_plot(rid):
    tss = predictWithLSTM(rid)
    plt.subplots()
    plt.plot(tss['ts'][0], tss['medianttc'][0], label='$\\sigma_{\\hat{\\mu}}$')
    plt.plot(tss['ts'][0], tss['bytes'][0], label='sum bytes')
    plt.plot(tss['ts'][0], tss['nxfers'][0], label='sum nxfers')
    plt.plot(tss['targetts'][0], tss['target'][0], '*', label='target')
    plt.plot(tss['targetts'][0], tss['pmedianttc'][0],'*', label='$\\delta_{median(\\beta_{\\hat{\\mu}})}$')
    plt.plot(tss['targetts'][0], tss['pmeanttc'][0],'*', label='$\\delta_{mean(\\beta_{\\hat{\\mu}})}$')
    plt.legend()
    return tss

def predict_and_fogp(rids, thr=.1):
    targets = []
    pmedians = []
    pmeans = []
    for rid in rids:
        tss = predictWithLSTM(rid)
        targets.append(tss['target'][0])
        pmedians.append(tss['pmedianttc'][0])
        pmeans.append(tss['pmeanttc'][0])
    targets = np.array(targets)
    pmedians = np.array(pmedians)
    pmeans = np.array(pmeans)
    return targets, pmedians, pmeans, fogp(targets, pmedians.flatten(), thr), fogp(targets, pmeans.flatten(), thr)

rho = 30
lags = 90
lookback = 0
lookahead = 18

startt = dt.datetime(2019,6,30,21)
endt = dt.datetime(2019,7,1,4,0)

indices = rules.sort_values(by='min_created')[(rules.min_created > startt) & (rules.min_created < endt)].index
ruleids = rules.loc[indices].ruleid.values
tss = create_dataset(ruleids)
tssmu = {}
tssstd = {}
tssmu['bytes'] = tss['bytes'].mean()
tssmu['nxfers'] = tss['nxfers'].mean()
tssmu['medianttc'] = tss['medianttc'].mean()
tssmu['meanttc'] = tss['meanttc'].mean()
tssmu['target'] = tss['target'].mean()
tssstd['bytes'] = tss['bytes'].std()
tssstd['nxfers'] = tss['nxfers'].std()
tssstd['medianttc'] = tss['medianttc'].std()
tssstd['meanttc'] = tss['meanttc'].std()
tssstd['target'] = tss['target'].std()

layers = 1
neurons = 32
epochs = 60

df, losses, models = traintestLSTM(ruleids)

rid = '40b3d48e074148c287479e74b2db183c'
tss2 = predict_and_plot(rid)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt) & (rules.min_created < (endt+dt.timedelta(minutes=10)))].index
rids = rules.loc[indices].ruleid.values
targets, pmedians, pmeans, fogpmedian, fogpmean = predict_and_fogp(rids, thr=.1)
print('len: ', len(pmedians), '\nfogpmedian: ', fogpmedian,'\n', 'fogpmean: ', fogpmean)
plt.subplots(); plt.plot(targets), plt.plot(pmedians)
plt.subplots(); plt.plot(targets), plt.plot(pmeans)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt + dt.timedelta(minutes=10)) & (rules.min_created < (endt+dt.timedelta(minutes=20)))].index
rids = rules.loc[indices].ruleid.values
targets, pmedians, pmeans, fogpmedian, fogpmean = predict_and_fogp(rids, thr=.1)
print('len: ', len(pmedians), '\nfogpmedian: ', fogpmedian,'\n', 'fogpmean: ', fogpmean)
plt.subplots(); plt.plot(targets), plt.plot(pmedians)
plt.subplots(); plt.plot(targets), plt.plot(pmeans)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt + dt.timedelta(minutes=60)) & (rules.min_created < (endt+dt.timedelta(minutes=70)))].index
rids = rules.loc[indices].ruleid.values
targets, pmedians, pmeans, fogpmedian, fogpmean = predict_and_fogp(rids, thr=.1)
print('len: ', len(pmedians), '\nfogpmedian: ', fogpmedian,'\n', 'fogpmean: ', fogpmean)
plt.subplots(); plt.plot(targets), plt.plot(pmedians)
plt.subplots(); plt.plot(targets), plt.plot(pmeans)
