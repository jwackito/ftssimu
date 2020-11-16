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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# do the TSA logaritmic

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

def get_links_for_rule(rid):
    return rulelinkmap[rid]


def get_data_for_rule(rid):
    r = rules[rules.ruleid == rid].iloc[0]
    keys = ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'links', 'target_bytes', 'target_nxfers', 'target']
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
    tss['links'] = get_links_for_rule(rid)
    tss['ubytes'] = list(np.array(tss['bytes']) - np.array(tss['fbytes']))
    tss['unxfers'] = list(np.array(tss['nxfers']) - np.array(tss['fnxfers']))
    tss['target_bytes'].append(r.bytes)
    tss['target_nxfers'].append(r.nxfers)
    tss['target'].append(r.ttc)
    return tss


def create_dataset(rids):
    keys = ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'links', 'target_bytes', 'target_nxfers' ,'target']
    tss = {}
    for k in keys:
        tss[k] = []
    i = len(rids)
    ts = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_data_for_rule)(r) for r in rids)
    for tt in ts:
        for metric in keys:
            if metric == 'links':
                tss[metric].append(tt[metric])
            else:
                tss[metric].append(tt[metric][0])
    for k in keys:
        tss[k] = np.array(tss[k])
    return tss

def build_FunnelNet_v1_model(shp):
    tsinput = keras.Input(shape=(None,shp),name='ts_input')
    x = layers.Conv1D(neurons, 5, padding='valid')(tsinput)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(neurons, 5, padding='valid')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LSTM(neurons, dropout=0.1, recurrent_dropout=.50)(x)
    tsoutput = layers.Dense(1)(x)
    
    pwinput = keras.Input(shape=(2), name='pw_input')
    x = layers.Dense(neurons, activation='relu')(pwinput)
    pwoutput = layers.Dense(1)(x)
    
    embinput = keras.Input(shape=(None,), name='emb_input')
    x = layers.Embedding(10000, neurons)(embinput)
    x = layers.LSTM(neurons)(x)
##   uncomment for embedding-pointwise connected model
    x = layers.Dense(1)(x)
    x = layers.concatenate([pwoutput, x])
    emboutput = layers.Dense(1, name='emb_output')(x)

    x = layers.concatenate([tsoutput, pwoutput, emboutput])
    model_output = layers.Dense(1,name='moutput')(x)

    model = keras.Model(inputs=[tsinput,pwinput, embinput], outputs=[model_output])
    return model

def build_Chollet_Jena_10ch_model(shp):
    tsinput = keras.Input(shape=(None,shp),name='ts_input')
    x = layers.Conv1D(neurons, 5, padding='valid')(tsinput)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling1D(3)(x)
    x = layers.Conv1D(neurons, 5, padding='valid')(x)
    x = layers.LeakyReLU()(x)
    x = layers.LSTM(neurons, dropout=0.1, recurrent_dropout=.50)(x)
    tsoutput = layers.Dense(1)(x)
    model_output = layers.Dense(1,name='moutput')(tsoutput)

    model = keras.Model(inputs=[tsinput], outputs=[model_output])
    return model

def transform_data(tss):
    trainx = []
    trainpwx = []
    keys = ['minttc', 'medianttc', 'meanttc', 'maxttc', 'fnxfers', 'unxfers', 'nxfers', 'bytes', 'ubytes', 'fbytes']
    channels = len(keys)
    for k in keys:
        trainx.append((tss[k] - tssmu[k])/tssstd[k])
    trainx = np.array(trainx)
    trainy = (tss['target'] - tssmu['target'])/tssstd['target']
    trainx = numpy.reshape(trainx, (trainx.shape[1], trainx.shape[2], channels))
    trainpwx = [(tss['target_bytes'] - tssmu['target_bytes'])/tssstd['target_bytes'], (tss['target_nxfers'] - tssmu['target_nxfers'])/tssstd['target_nxfers']]
    trainpwx = np.array(trainpwx).T
    trainembx = [np.array(t.texts_to_sequences(l)).flatten() for l in tss['links']]
    trainembx = keras.preprocessing.sequence.pad_sequences(trainembx, maxlen=50)
    print(trainx.shape, trainpwx.shape, trainembx.shape, trainy.shape)
    return trainx, trainpwx, trainembx, trainy

def trainvalChollet(tss, tss_val):
    # making only one model using all the features
    traintsx, trainpwx, trainembx, trainy = transform_data(tss)
    traintsx_val, trainpwx_val, trainembx_val, trainy_val = transform_data(tss_val)

    model = build_Chollet_Jena_10ch_model(traintsx.shape[-1])
    model.compile(loss='mae', optimizer='RMSProp')

    loss = model.fit({'ts_input':traintsx},
            {'moutput': trainy}, epochs=epochs, batch_size=1024, shuffle=True, 
            validation_data=({'ts_input':traintsx_val}, {'moutput': trainy_val}))
    tss['pchollet'] = model.predict({'ts_input':traintsx})
    return tss, loss, model

def predictWithChollet(tss, model):
    testtsx, testpwx, testembx, testy = transform_data(tss)
    preds = model.predict({'ts_input':testtsx})
    tss['pchollet'] = (preds*tssstd['target'])+tssmu['target']
    return tss

def trainvalFunnel(tss, tss_val):
    # making only one model using all the features
    traintsx, trainpwx, trainembx, trainy = transform_data(tss)
    traintsx_val, trainpwx_val, trainembx_val, trainy_val = transform_data(tss_val)

    model = build_FunnelNet_v1_model(traintsx.shape[-1])
    model.compile(loss='mae', optimizer='RMSProp')

    loss = model.fit({'ts_input':traintsx, 'pw_input':trainpwx, 'emb_input':trainembx},
            {'moutput': trainy}, epochs=epochs, batch_size=1024, shuffle=True, 
            validation_data=({'ts_input':traintsx_val, 'pw_input':trainpwx_val, 'emb_input': trainembx_val}, {'moutput': trainy_val}))
    tss['pfunnel'] = model.predict({'ts_input':traintsx, 'pw_input':trainpwx, 'emb_input': trainembx})
    return tss, loss, model

def predictWithFunnel(tss, model):
    testtsx, testpwx, testembx, testy = transform_data(tss)
    preds = model.predict({'ts_input':testtsx, 'pw_input':testpwx, 'emb_input':testembx})
    tss['pfunnel'] = (preds*tssstd['target'])+tssmu['target']
    return tss

def predict_and_fogp(tss, thr=.1):
    targets = []
    pmodel = []
    tss = predictWithChollet(tss,chollet_model)
    tss = predictWithFunnel(tss, funnel_model)
    targets = np.array(tss['target'])
    pchollet = np.array(tss['pchollet'])
    pfunnel = np.array(tss['pfunnel'])
    return targets, pchollet, pfunnel, fogp(targets, pchollet.flatten(), thr), fogp(targets, pfunnel.flatten(), thr)

def savetss(name, tss):
    for k in tss.keys(): 
        pd.DataFrame(tss[k]).to_csv('%s_%s.csv'%(name, k), index=False)

def readtss(name): 
    keys =  ['ts', 'bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'links', 'target_bytes', 'target_nxfers', 'target'] 
    tss = {} 
    for k in keys: 
        tss[k] = pd.read_csv('%s_%s.csv'%(name, k)).values 
        if k == 'ts': 
            tss[k] = np.array([pd.to_datetime(d) for d in tss[k]])
        if k == 'links':
            linklist = []
            for ls in tss[k]:
                linklist.append(np.array(ls[0].replace('[','').replace(']','').replace("'",'').replace('\n','').split(' ')))
            tss[k] = linklist
    tss['target_bytes'] = tss['target_bytes'].flatten()
    tss['target_nxfers'] = tss['target_nxfers'].flatten()
    tss['target'] = tss['target'].flatten()
    return tss

print('Reading Rules Dataset')
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
print('Reading rule-link map')
rulelinkmap = pickle.load(open('data/rulelinkmap.pickle','rb'))

print('Reading Training datasets')
train = readtss('data/train_ultimate')
print('Reading Validation datasets')
train_val = readtss('data/validation_ultimate')

print('Learning tokens')
t = keras.preprocessing.text.Tokenizer(filters=' ', oov_token='XXXX')
t.fit_on_texts(pd.read_csv('data/rule_links_unique').link.values)

print('Normalizing datasets')
tssmu = {}
tssstd = {}
for k in ['bytes', 'fbytes', 'ubytes', 'nxfers', 'fnxfers', 'unxfers', 'minttc', 'medianttc', 'meanttc', 'maxttc', 'target_bytes', 'target_nxfers', 'target']:
    tssmu[k] = train[k].mean()
    tssstd[k] = train[k].std()

rho = 30
lags = 240
lookback = 0
lookahead = 18

channels = 10
neurons = 32
epochs = 120

train, loss_chollet, chollet_model = trainvalChollet(train, train_val)
train, loss_funnel, funnel_model = trainvalFunnel(train, train_val)

chollet_model.save('CholletModel_120epochs_recurrentdropout.model')
funnel_model.save('FunnelModel_120epochs_recurrentdropout.model')

print('Reading Testing datasets')
test = readtss('data/test_ultimate')

plt.subplots()
plt.plot(loss_chollet.history['loss'], label='train');
plt.plot(loss_chollet.history['val_loss'],label='validation');
plt.title('MAE Loss Train/Validation Chollet Jena Model')
plt.legend()

plt.subplots()
plt.plot(loss_funnel.history['loss'], label='train');
plt.plot(loss_funnel.history['val_loss'],label='validation');
plt.title('MAE Loss Train/Validation FunnelNeti Model')
plt.legend()

targets, pchollet, pfunnel, fogpchollet, fogpfunnel = predict_and_fogp(test, thr=.1)
plt.subplots()
plt.plot(test['ts'][:,-1], targets, label='target: %d samples'%len(targets))
plt.plot(test['ts'][:,-1], pchollet, label='pred Chollet-Jena: %0.4f'%fogpchollet)
plt.plot(test['ts'][:,-1], pfunnel, label='pred FunnelNet: %0.4f'%fogpfunnel)
plt.title('Prediction over Test')
plt.legend()
