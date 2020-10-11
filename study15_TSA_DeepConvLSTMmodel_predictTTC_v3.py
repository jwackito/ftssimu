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
InteractiveSession.close()
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

def get_data_for_rule(rid):
    r = rules[rules.ruleid == rid].iloc[0]
    keys = ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'target']
    tss = {}
    for k in keys:
        tss[k] = []
    cut = get_observed_finished_rules_at(r.min_created, rho, lags, lookback)
    ts = get_timeseries_for_rules(r, cut, rho, lags, lookback) 
    for metric in ['ts', 'minttc', 'medianttc', 'meanttc', 'maxttc']:
        tss[metric].append(ts[metric].values)
    tss['fnxfers'].append(ts['nxfers'].values)
    tss['fbytes'].append(ts['bytes'].values)
    cut = get_real_finished_rules_at(r.min_created, rho, lags, lookback)
    ts = get_timeseries_for_rules(r, cut, rho, lags, lookback)
    for metric in ['bytes', 'nxfers']:
        tss[metric].append(ts[metric].values)
    tss['ubytes'] = list(np.array(tss['bytes']) - np.array(tss['fbytes']))
    tss['unxfers'] = list(np.array(tss['nxfers']) - np.array(tss['fnxfers']))
    tss['target'].append(r.ttc)
    return tss


def create_dataset(rids):
    keys = ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'target']
    tss = {}
    for k in keys:
        tss[k] = []
    i = len(rids)
    ts = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_data_for_rule)(r) for r in rids)
    for tt in ts:
        for metric in keys:
            tss[metric].append(tt[metric][0])
    for k in keys:
        tss[k] = np.array(tss[k])
    return tss

def traintestLSTM(tss, tss_val):
    losses = {}
    models = {}
    # making only one model using all the features
    for mu in ['model']:
        print('Training for everything...')
        trainx = []
        trainx_val = []
        #for k in ['minttc', 'medianttc', 'meanttc', 'maxttc', 'fnxfers', 'unxfers', 'nxfers', 'bytes', 'ubytes', 'fbytes']:
        for k in ['medianttc', 'maxttc', 'unxfers', 'ubytes']:
        #for k in ['medianttc', 'maxttc', 'fnxfers', 'unxfers', 'nxfers']:
            trainx.append((tss[k] - tssmu[k])/tssstd[k])
            trainx_val.append((tss_val[k] - tssmu[k])/tssstd[k])
        trainx = np.array(trainx)
        trainx_val = np.array(trainx_val)
        #zmin = train.min()
        #zmax = train.max()
        #train = (train - zmin)/zmax
        trainy = (tss['target'] - tssmu['target'])/tssstd['target']
        trainy_val = (tss_val['target'] - tssmu['target'])/tssstd['target']
        print(trainx.shape)
        trainx = numpy.reshape(trainx, (trainx.shape[1], trainx.shape[2], channels))
        trainx_val = numpy.reshape(trainx_val, (trainx_val.shape[1], trainx_val.shape[2], channels))
        model = Sequential()
        # from Chollet
        model.add(Conv1D(neurons, 5, input_shape=(None, trainx.shape[-1]),padding='valid'))
        model.add(LeakyReLU())
        model.add(MaxPooling1D(3))
        model.add(Conv1D(neurons, 5,padding='valid'))
        model.add(LeakyReLU())
        model.add(LSTM(neurons, dropout=.1, recurrent_dropout=.0))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='RMSProp')
        print('TRAINING ######################')
        loss = model.fit(trainx, trainy, epochs=epochs, batch_size=10, shuffle=True, validation_data=(trainx_val, trainy_val))
        models[mu] = model
        losses[mu] = loss
    return tss, losses, models

def predictWithLSTM(tss):
    #r = rules[rules.ruleid == rid].iloc[0]
    #tss = create_dataset([rid])
    #tss['targetts'] = [r.min_created]
    for mu in ['model']:
        testx = []
        #for k in ['minttc', 'medianttc', 'meanttc', 'maxttc', 'fnxfers', 'unxfers', 'nxfers', 'bytes', 'ubytes', 'fbytes']:
        for k in ['medianttc', 'maxttc', 'unxfers', 'ubytes']:
            testx.append((tss[k] - tssmu[k])/tssstd[k])
        testx = np.array(testx)
        testy = (tss['target'] - tssmu['target'])/tssstd['target']
        testx = numpy.reshape(testx, (testx.shape[1], testx.shape[2],channels))
        #testx = testx.reshape(testx)
        preds = models[mu].predict(testx)
        tss['p'+mu] = (preds*tssstd['target'])+tssmu['target']
    return tss

def predict_and_plot(rid):
    tss = create_dataset([rid])
    tss = predictWithLSTM(tss)
    plt.subplots()
    plt.plot(tss['ts'][0], tss['minttc'][0], label='$\\sigma_{\\hat{min}}$')
    plt.plot(tss['ts'][0], tss['medianttc'][0], label='$\\sigma_{\\hat{median}}$')
    plt.plot(tss['ts'][0], tss['meanttc'][0], label='$\\sigma_{\\hat{mean}}$')
    plt.plot(tss['ts'][0], tss['maxttc'][0], label='$\\sigma_{\\hat{max}}$')
    plt.plot(tss['ts'][0], tss['ubytes'][0], label='sum bytes')
    plt.plot(tss['ts'][0], tss['unxfers'][0], label='sum unfinished nxfers')
    plt.plot(tss['ts'][0][-1], tss['target'][0], '*', label='target')
    plt.plot(tss['ts'][0][-1], tss['pmodel'][0],'*', label='$\\delta_{(\\beta_{\\hat{\\mu}})}$')
    plt.legend()
    return tss

def predict_and_fogp(tss, thr=.1):
    targets = []
    pmodel = []
    tss = predictWithLSTM(tss)
    targets = np.array(tss['target'])
    pmodel = np.array(tss['pmodel'])
    return targets, pmodel, fogp(targets, pmodel.flatten(), thr)

def savetss(name, tss):
    for k in tss.keys(): 
        pd.DataFrame(tss[k]).to_csv('%s_%s.csv'%(name, k), index=False)

def readtss(name): 
    keys =  ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'target'] 
    tss = {} 
    for k in keys: 
        tss[k] = pd.read_csv('%s_%s.csv'%(name, k)).values 
        if k == 'ts': 
            tss[k] = np.array([pd.to_datetime(d) for d in tss[k]]) 
    tss['target'] = tss['target'].flatten()
    return tss

rho = 30
lags = 240
lookback = 0
lookahead = 18

startt = dt.datetime(2019,7,7)
#startt = dt.datetime(2019,6,30,21,0)
endt = dt.datetime(2019,7,8)

create_train_test = False
suffix = '_7jul-8jul'
if not create_train_test:
    print('Reading datasets')
    tss = readtss('train'+suffix)
    tss_val = readtss('validation'+suffix)
    tss1 = readtss('test1'+suffix)
    tss2 = readtss('test2'+suffix)
    tss3 = readtss('test3'+suffix)
    tss4 = readtss('test4'+suffix)

indices = rules.sort_values(by='min_created')[(rules.min_created > startt) & (rules.min_created < endt)].index
ruleids = rules.loc[indices].ruleid.values
if create_train_test:
    tss = create_dataset(ruleids)
    savetss('train'+suffix, tss)
tssmu = {}
tssstd = {}

for k in ['bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'target']:
    tssmu[k] = tss[k].mean()
    tssstd[k] = tss[k].std()

indices = rules.sort_values(by='min_created')[(rules.min_created > endt) & (rules.min_created < (endt+dt.timedelta(hours=12)))].index
rids = rules.loc[indices].ruleid.values
if create_train_test:
    tss_val = create_dataset(rids)
    savetss('validation'+suffix, tss_val)

# channels  = number of features to look at
channels = 10
layers = 1
neurons = 32
epochs = 120

df, losses, models = traintestLSTM(tss, tss_val)

rid = '40b3d48e074148c287479e74b2db183c'
tss_one = predict_and_plot(rid)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt) & (rules.min_created < (endt+dt.timedelta(minutes=10)))].index
rids = rules.loc[indices].ruleid.values
if create_train_test:
    tss1 = create_dataset(rids)
    savetss('test1'+suffix, tss1)
targets, pmodel, fogpmodel = predict_and_fogp(tss1, thr=.1)
print('len: ', len(pmodel), '\nfogpmodel: ', fogpmodel)
plt.subplots(); plt.plot(targets), plt.plot(pmodel)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt + dt.timedelta(minutes=10)) & (rules.min_created < (endt+dt.timedelta(minutes=20)))].index
rids = rules.loc[indices].ruleid.values
if create_train_test:
    tss2 = create_dataset(rids)
    savetss('test2'+suffix, tss2)
targets, pmodel, fogpmodel = predict_and_fogp(tss2, thr=.1)
print('len: ', len(pmodel), '\nfogpmodel: ', fogpmodel)
plt.subplots(); plt.plot(targets), plt.plot(pmodel)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt + dt.timedelta(minutes=60)) & (rules.min_created < (endt+dt.timedelta(minutes=70)))].index
rids = rules.loc[indices].ruleid.values
if create_train_test:
    tss3 = create_dataset(rids)
    savetss('test3'+suffix, tss3)
targets, pmodel, fogpmodel = predict_and_fogp(tss3, thr=.1)
print('len: ', len(pmodel), '\nfogpmodel: ', fogpmodel)
plt.subplots(); plt.plot(targets), plt.plot(pmodel)

indices = rules.sort_values(by='min_created')[(rules.min_created > dt.datetime(2019,7,8,8)) & (rules.min_created < dt.datetime(2019,7,8,8,20))].index
rids = rules.loc[indices].ruleid.values
if create_train_test:
    tss4 = create_dataset(rids)
    savetss('test4'+suffix, tss4)
targets, pmodel, fogpmodel = predict_and_fogp(tss4, thr=.1)
print('len: ', len(pmodel), '\nfogpmodel: ', fogpmodel)
plt.subplots(); plt.plot(tss4['ts'][:,-1], targets), plt.plot(tss4['ts'][:,-1],pmodel)
