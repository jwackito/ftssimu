from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv('data/rules_timeseries.csv')
df.min_created = pd.to_datetime(df.min_created)

x = df.medianttc

def traintest_autoreg(train, test, lags=60):
    model = AutoReg(train, lags=lags)
    model = model.fit()
    pred = model.predict(start=len(train), end=len(train)+len(test)-1)
    return pred

def pltpred(x, start=0, trainlen=100, testlen=20):
    train, test = log10(x.values[start:start+trainlen]), log10(x.values[start+trainlen:start+trainlen+testlen])
    pred = traintest_autoreg(train, test)
    plt.plot(df.min_created[start+trainlen:start+trainlen+testlen], 10**test, label='obs')
    plt.plot(df.min_created[start+trainlen:start+trainlen+testlen], 10**pred, label='pred')
    plt.legend()
    plt.ylabel('Rule TTC')
    plt.xlabel('Rule creation time')

def getrmse(start=0, trainlen=100, testlen=20):
    train, test = log10(x.values[start:start+trainlen]), log10(x.values[start+trainlen:start+trainlen+testlen])
    rmses = []
    for lag in [1,2,10, 20, 30, 50, 60, 80, 90]:
        pred = traintest_autoreg(train, test, lag)
        print('Pred%d RMSE: %0.3f'%(lag, sqrt(mse(test, pred))))
        rmses.append(sqrt(mse(10**test, 10**pred)))
    return rmses

def pltpred_sliding_window(x, start=0, trainlen=500, testlen=20):
    tests = log10(x.values[start+trainlen:start+trainlen+testlen])
    preds = []
    for i in range(testlen):
        train, test = log10(x.values[start+i:start+i+trainlen]), log10(x.values[start+i+trainlen:start+i+trainlen+1])
        pred = traintest_autoreg(train, test)
        preds.append(pred)
    preds = np.array(preds)
    plt.plot(df.min_created[start+trainlen:start+trainlen+testlen], 10**tests, label='obs')
    plt.plot(df.min_created[start+trainlen:start+trainlen+testlen], 10**preds, label='pred')
    plt.legend()
    plt.ylabel('Rule TTC')
    plt.xlabel('Rule creation time')

def pred_sliding_window(df, metric, start=0, trainlen=1000, testlen=0):
    if testlen == 0:
        testlen = len(df) - trainlen
    min_created = df.min_created[start+trainlen:start+trainlen+testlen]
    x = df[metric]
    preds = []
    for i in range(testlen):
        train, test = log10(x.values[start+i:start+i+trainlen]), log10(x.values[start+i+trainlen:start+i+trainlen+1])
        pred = traintest_autoreg(train, test, lags=60)
        preds.append(pred[0])
    preds = 10**np.array(preds)
    return pd.DataFrame(np.array([min_created, preds]).T, columns=['created_min', 'pred'])

def makepredictionforrule_previuos(r, df):
    '''Make a prediction for a rule, based on the min, median, mean or max of the previous 60 seconds'''
    try: 
        return df[(df.min_created < rule.min_created) & (df.min_created >= (rule.min_created - dt.timedelta(seconds=60)))][['minttc', 'medianttc', 'meanttc', 'maxttc']].values[0] 
    except IndexError: 
        print(r.min_created, (r.min_created - dt.timedelta(seconds=60))) 
        return [np.nan, np.nan, np.nan, np.nan]

def makepredictionforrule_future(r, df):
    '''Make a prediction for a rule, based on the min, median, mean or max of the curren 60 seconds'''
    try: 
        return df[(df.min_created < rule.min_created + dt.timedelta(seconds=60)) & (df.min_created >= rule.min_created)][['minttc', 'medianttc', 'meanttc', 'maxttc']].values[0] 
    except IndexError: 
        print(r.min_created, (r.min_created + dt.timedelta(seconds=60))) 
        return [np.nan, np.nan, np.nan, np.nan]


def generate_prediction_for_rules_with_real_timseries_values():
    rules = pd.read_csv('data/xfers_per_rule_2.csv') 
    rules.nxfers = rules.nxfers.astype(int) 
    rules.bytes = rules.bytes.astype(float) 
    for d in ['min_created', 'min_submitted', 
           'min_started', 'min_ended', 'max_created', 'max_submitted', 
           'max_started', 'max_ended']: 
        rules[d] = pd.to_datetime(rules[d]) 
    rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9 
    rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int) 
    df = pd.read_csv('data/rules_timeseries.csv')
    df.min_created = pd.to_datetime(df.min_created)
    pred = []
    i = 0 
    rulelen = len(rules) 
    for rule in rules.itertuples(): 
        pred.append(makepredictionforrule_previous(rule, df)) 
        print('%d/%d'%(i, rulelen),end='\r') 
        i += 1
    pred = np.array(pred)
    rules['pred_tsa_realmin'] = pred[:,0]
    rules['pred_tsa_realmedian'] = pred[:,1]
    rules['pred_tsa_realmean'] = pred[:,2]
    rules['pred_tsa_realmax'] = pred[:,3]
    rules.to_csv('data/xfers_per_rule2_with_TSA_realmetric_previous_predictions.csv', index=False)
    return rules

def generate_prediction_for_rules_with_future_timseries_values():
    rules = pd.read_csv('data/xfers_per_rule_2.csv') 
    rules.nxfers = rules.nxfers.astype(int) 
    rules.bytes = rules.bytes.astype(float) 
    for d in ['min_created', 'min_submitted', 
           'min_started', 'min_ended', 'max_created', 'max_submitted', 
           'max_started', 'max_ended']: 
        rules[d] = pd.to_datetime(rules[d]) 
    rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9 
    rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int) 
    df = pd.read_csv('data/rules_timeseries.csv')
    df.min_created = pd.to_datetime(df.min_created)
    pred = []
    i = 0 
    rulelen = len(rules) 
    for rule in rules.itertuples(): 
        pred.append(makepredictionforrule_future(rule, df)) 
        print('%d/%d'%(i, rulelen),end='\r') 
        i += 1
    pred = np.array(pred)
    rules['pred_tsa_realmin'] = pred[:,0]
    rules['pred_tsa_realmedian'] = pred[:,1]
    rules['pred_tsa_realmean'] = pred[:,2]
    rules['pred_tsa_realmax'] = pred[:,3]
    rules.to_csv('data/xfers_per_rule2_with_TSA_realmetric_future_predictions.csv', index=False)
    return rules
