import pandas as pd

from keras.layers import Embedding, Flatten, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from functions import read_xfers, cut_outliers


def get_context(t, xfers):
    cut = xfers[t.ended > xfers.submitted]
    cut = cut[t.submitted < cut.ended]
    return cut

def convert_to_sequences(xfers, tokenizer):
    l = xfers.to_csv(header=None, index=False).split('\n')
    l1 = [[s.replace(' ','_').replace(':', '_').replace(',', ' '). replace('_', '').replace('-','')] for s in l]
    l2 = [s[0] for s in l1][:-1]
    return tokenizer.texts_to_sequences(l2)

def preprocess(t, context, tokenizer):
    data = []
    # delete spureous columns
    context.pop('started')
    context.pop('ended')
    context.pop('anomalous1')
    context.pop('anomalous2')
    context.pop('NTIME')
    context.pop('QTIME')
    context.pop('RATE')
    # make timestamps relative to t.submitted
    context['dcreated'] = (context.created - t.submitted).values.astype(int)//10**9
    context['dsubmitted'] = (context.submitted - t.submitted).values.astype(int)//10**9
    #context['dstarted'] = (context.started - t.submitted).values.astype(int)//10**9
    #context['dended'] = (context.ended - t.submitted).values.astype(int)//10**9
    for c in context.itertuples():
        l = [a[0] for a in tokenizer.texts_to_sequences([c.account, c.activity, c.src_name, c.dst_name, c.source_rse_id, c.dest_rse_id])]
        l.append(c.dcreated)
        l.append(c.dsubmitted)
        l.append(c.SIZE)
        data.append(l)
    # append target information
    dcreated = int((t.created - t.submitted).total_seconds())
    dsubmitted = int((t.submitted - t.submitted).total_seconds())
    l = [a[0] for a in tokenizer.texts_to_sequences([t.account, t.activity, t.src_name, t.dst_name, t.source_rse_id, t.dest_rse_id])]
    l.append(dcreated)
    l.append(dsubmitted)
    l.append(t.SIZE)
    data.append(l)
    # append target
    data.append([t.anomalous2])
    return data 

def generate_datasample(t, xfers):
    context = get_context(t, xfers)
    context['created'] = (context.created - t.created).astype(int)//10**9
    context['submitted'] = (context.submitted - t.created).astype(int)//10**9
    context['started'] = (context.started - t.created).astype(int)//10**9
    context['ended'] = (context.ended - t.created).astype(int)//10**9
    return context



#xfers = pd.read_csv('data/transfers-2018-04-15.csv', usecols=['account', 'activity', 'src_name', 'dst_name', 'source_rse_id','dest_rse_id', 'created_at', 'started_at', 'submitted_at', 'transferred_at', 'bytes'])
#xfers['activity'] = [s.replace(' ', '_') for s in xfers.activity.values]
#xfers['created'] = pd.to_datetime(xfers.created_at)
#xfers['submitted'] = pd.to_datetime(xfers.submitted_at)
#xfers['started'] = pd.to_datetime(xfers.started_at)
#xfers['ended'] = pd.to_datetime(xfers.transferred_at)
##xfers['isubmitted'] = (xfers.submitted - xfers.submitted.min()).values.astype(int)//10**9
##xfers['istarted'] = (xfers.started - xfers.submitted.min()).values.astype(int)//10**9
##xfers['iended'] = (xfers.ended - xfers.submitted.min()).values.astype(int)//10**9
#xfers['SIZE'] = xfers.bytes
#xfers.pop('created_at')
#xfers.pop('submitted_at')
#xfers.pop('started_at')
#xfers.pop('transferred_at')
#xfers.pop('bytes')
#xfers['NTIME'] = ((xfers.ended - xfers.started).values / 10**9).astype(int)
#xfers['QTIME'] = ((xfers.started - xfers.submitted).values / 10**9).astype(int)
#xfers['RATE'] = xfers.SIZE/xfers.NTIME
#xfers['anomalous1'] = np.logical_and((xfers.NTIME.values > 780), (xfers.QTIME.values > 4600)).astype(int)
#xfers['anomalous2'] = (xfers.RATE/xfers.SIZE < 0.02).astype(int)
#xfers = xfers[xfers.RATE != inf]

src = 'CERN-PROD'
dst = 'BNL-ATLAS'

xfers = pd.read_csv('xfers_richer_queued_per_activity_20180401-20180420_v3.csv')
activities = list(set(xfers.activity))
for act in activities:
    xfers['qshare_'+act] = (xfers['q__'+act]/(xfers.q__total))
data = [xfers.SIZE.values, xfers.RTIME.values, 
    pd.Categorical(xfers.activity,ordered=False).codes,
    pd.Categorical(xfers.src_rse_name,ordered=False).codes,
    pd.Categorical(xfers.dst_rse_name,ordered=False).codes,
    xfers.q__total.values,
    xfers.s__total.values,
]
for act in activities:
    data.append(xfers['qshare_'+act].values)
data.append(xfers.QTIME.values)
data = np.array(data)
share = .7
data_train = data[:,:int(data.shape[1]*share)]
data_test = data[:,int(data.shape[1]*share):]

xtrain = data_train[:-1,:]
ytrain = data_train[-1,:]
xtest = data_test[:-1,:]
ytest = data_test[-1,:]


print('Building the model')
model = Sequential([
    Dense(64, input_shape=(),return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(1, activation='relu'),
])
model.compile(optimizer='adam', loss='mse')



print('Generating train examples')
data = []
for n, a in zip(abnormal_train.itertuples(), normal_train.itertuples()):
    # fit an normal case
    context = get_context(n,xfers)
    data = preprocess(n, context, tokenizer)
    target = data[-1]
    train_on = np.zeros((max_time_steps,9))
    train_on[-len(data[:-1]):] = data[-max_time_steps-1:-1]
    model.fit(train_on.reshape(1,-1, 9), target, epochs=3)
    # fit a abnormal case
    context = get_context(a,xfers)
    data = preprocess(a, context, tokenizer)
    target = data[-1]
    train_on = np.zeros((max_time_steps,9))
    train_on[-len(data[:-1]):] = data[-max_time_steps-1:-1]
    model.fit(train_on.reshape(1,-1, 9), target, epochs=3)

print('Predicting 40 abnormal and 40 normal transfers')
results_abnormal = []
results_normal = []
for n, a in zip(abnormal_test.itertuples(), normal_test.itertuples()):
    # fit an normal case
    context = get_context(n,xfers)
    data = preprocess(n, context, tokenizer)
    target = data[-1]
    train_on = np.zeros((max_time_steps,9))
    train_on[-len(data[:-1]):] = data[-max_time_steps-1:-1]
    pred = model.predict(train_on.reshape(1,-1, 9))
    print(target, pred)
    results_normal.append(pred)
    # fit a abnormal case
    context = get_context(a,xfers)
    data = preprocess(a, context, tokenizer)
    target = data[-1]
    train_on = np.zeros((max_time_steps,9))
    train_on[-len(data[:-1]):] = data[-max_time_steps-1:-1]
    pred = model.predict(train_on.reshape(1,-1, 9))
    print(target, pred)
    results_abnormal.append(pred)
