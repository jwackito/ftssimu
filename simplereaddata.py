import pandas as pd

from keras.layers import Embedding, Flatten, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer

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

def read_xfers(path, src, dst):
    xfers = pd.read_csv(path, usecols=['account', 'activity', 'src_name', 'dst_name', 'source_rse_id','dest_rse_id', 'created_at', 'started_at', 'submitted_at', 'transferred_at', 'bytes'])
    xfers['activity'] = [s.replace(' ', '_') for s in xfers.activity.values]
    xfers['created'] = pd.to_datetime(xfers.created_at)
    xfers['submitted'] = pd.to_datetime(xfers.submitted_at)
    xfers['started'] = pd.to_datetime(xfers.started_at)
    xfers['ended'] = pd.to_datetime(xfers.transferred_at)
    #xfers['isubmitted'] = (xfers.submitted - xfers.submitted.min()).values.astype(int)//10**9
    #xfers['istarted'] = (xfers.started - xfers.submitted.min()).values.astype(int)//10**9
    #xfers['iended'] = (xfers.ended - xfers.submitted.min()).values.astype(int)//10**9
    xfers['SIZE'] = xfers.bytes
    xfers.pop('created_at')
    xfers.pop('submitted_at')
    xfers.pop('started_at')
    xfers.pop('transferred_at')
    xfers.pop('bytes')
    xfers['RTIME'] = ((xfers.submitted - xfers.created).values / 10**9).astype(int)
    xfers['NTIME'] = ((xfers.ended - xfers.started).values / 10**9).astype(int)
    xfers['QTIME'] = ((xfers.started - xfers.submitted).values / 10**9).astype(int)
    xfers['RATE'] = xfers.SIZE/xfers.NTIME
    xfers['anomalous1'] = np.logical_and((xfers.NTIME.values > 780), (xfers.QTIME.values > 4600)).astype(int)
    xfers['anomalous2'] = (xfers.RATE/xfers.SIZE < 0.02).astype(int)
    xfers = xfers[xfers.RATE != inf]
    SD_link = xfers[xfers.src_name == src]
    SD_link = SD_link[SD_link.dst_name == dst]
    return xfers, SD_link

SD_link = read_xfers('data/transfers-2018-06-16.csv', 'MWT2', 'AGLT2')

abnormal = SD_link[SD_link['anomalous2'] == 1]
normal = SD_link[SD_link['anomalous2'] == 0].sample(len(abnormal))

trainindex = int(len(abnormal)*.7)

abnormal_train = abnormal.iloc[:trainindex]
abnormal_test = abnormal.iloc[trainindex:]
normal_train = normal.iloc[:trainindex]
normal_test = normal.iloc[trainindex:]

print('Len:abnormal', len(abnormal))
print('Len:normal', len(normal))
