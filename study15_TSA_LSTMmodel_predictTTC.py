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
    df = pd.DataFrame(np.array([dates, minttc, medianttc, meanttc, maxttc]).T, columns=['ts', 'minttc', 'medianttc', 'meanttc', 'maxttc']).fillna(method='ffill').fillna(method='bfill')
    return df

def create_dataset(rids):
    tss = {}
    tss['ts'] = []
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
        for metric in ['ts', 'minttc', 'medianttc', 'meanttc', 'maxttc']:
            tss[metric].append(ts[metric].values[:-lookahead])
        tss['target'].append(r.ttc)
        i -= 1
    return tss

def traintestLSTM(ruleids):
    tss = create_dataset(ruleids)
    losses = {}
    models = {}
    for mu in ['medianttc', 'meanttc']:
        print('Training for ', mu, '...')
        trainx = np.array(tss[mu])
        #zmin = train.min()
        #zmax = train.max()
        #train = (train - zmin)/zmax
        trainy = np.array(tss['target'])
        model = Sequential()
        for i in range(layers-1):
            model.add(LSTM(neurons, input_shape=(1, lags-lookahead), return_sequences=True))
        model.add(LSTM(neurons, input_shape=(1, lags-lookahead)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        #trainx, trainy = create_dataset(train.reshape(-1,1),lags,lookahead)
        trainx = numpy.reshape(trainx, (trainx.shape[0], 1, trainx.shape[1]))
        loss = model.fit(trainx, trainy, epochs=epochs, batch_size=1)
        models[mu] = model
        losses[mu] = loss
    return pd.DataFrame(tss), losses, models

def predictWithLSTM(rid):
    r = rules[rules.ruleid == rid].iloc[0]
    tss = create_dataset([rid])
    tss['targetts'] = [r.min_created]
    for mu in ['medianttc', 'meanttc']:
        testx = np.array(tss[mu])
        testy = np.array(tss['target'])
        testx = numpy.reshape(testx, (testx.shape[0], 1, testx.shape[1]))
        testx = testx.reshape(1,1,lags-lookahead)
        preds = models[mu].predict(testx)
        tss['p'+mu] = preds
    return tss

def predict_and_plot(rid):
    tss = predictWithLSTM(rid)
    plt.subplots()
    plt.plot(tss['ts'][0], tss['medianttc'][0], label='$\\sigma_{\\hat{\\mu}}$')
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
lags = 20
lookback = 0
lookahead = 20
layers = 1
neurons = 16
epochs = 50


startt = dt.datetime(2019,7,1)
endt = dt.datetime(2019,7,1,0,30)

indices = rules.sort_values(by='min_created')[(rules.min_created > startt) & (rules.min_created < endt)].index
df, losses, models = traintestLSTM(rules.loc[indices].ruleid.values)

rid = '40b3d48e074148c287479e74b2db183c'
tss = predict_and_plot(rid)

indices = rules.sort_values(by='min_created')[(rules.min_created > endt) & (rules.min_created < (endt+dt.timedelta(minutes=10)))].index
rids = rules.loc[indices].ruleid.values
targets, pmedians, pmeans, fogpmedian, fogpmean = predict_and_fogp(rids, thr=.1)

