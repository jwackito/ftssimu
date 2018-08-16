import pandas as pd
import numpy as np
from keras.layers import Embedding, Flatten, LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from functions import read_xfers
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

#src = 'MWT2'
#dst = 'AGLT2'
src = 'BNL-ATLAS'
dst = 'MWT2'
#filepath = 'data/BNL_MWT2_20180313.csv'
#filepath = 'data/BNL_MWT2_20180405.csv'
#filepath = 'data/BNL_MWT2_20180618.csv'
#filepath = 'data/transfers-2018-04-15.csv'
#filepath = 'data/BNL_MWT2_20180619.csv'
#filepath = 'data/BNL_MWT2_20180515.csv'
#filepath = 'data/BNL_MWT2_20180522.csv' # failed due to high R²
#filepath = 'data/BNL_MWT2_20180429.csv'
filepath = 'data/sample.csv'

xfers, link = read_xfers(filepath, src, dst)
print('len(xfers):',len(xfers))
print('len(link):',len(link))

target=[]; m=[]; s=[]; mini=[]; maxi=[]; i=0                                          
for t in xfers.itertuples():
    print(i)
    c = generate_datasample(t, xfers)
    m.append([c.created.mean(), c.submitted.mean(), c.started.mean(), c.ended.mean()])
    s.append([c.created.std(), c.submitted.std(), c.started.std(), c.ended.std()])
    mini.append([c.created.min(), c.submitted.min(), c.started.min(), c.ended.min()])
    maxi.append([c.created.max(), c.submitted.max(), c.started.max(), c.ended.max()])
    target.append([t.id, len(c), t.RATE, t.RTIME, t.NTIME, t.QTIME, t.SIZE, t.created, t.submitted])
    i+=1
    if i > 5000:
        break
target = np.array(target)
m = np.array(m)
s = np.array(s)
mini = np.array(mini)
maxi = np.array(maxi)
df = pd.DataFrame()
df['id'] = target[:,0]
df['ncount'] = target[:,1].astype(int)
df['rate'] = target[:,2]
df['rtime'] = target[:,3].astype(int)
df['ntime'] = target[:,4].astype(int)
df['qtime'] = target[:,5].astype(int)
df['sizeb'] = target[:,6].astype(int)
df['created'] = target[:,7]
df['submitted'] = target[:,8]
df['mean_created'] = m[:,0]
df['mean_submitted'] = m[:,1]
df['mean_started'] = m[:,2]
df['mean_ended'] = m[:,3]
df['std_created'] = s[:,0]
df['std_submitted'] = s[:,1]
df['std_started'] = s[:,2]
df['std_ended'] = s[:,3]
df['min_created'] = mini[:,0]
df['min_submitted'] = mini[:,1]
df['min_started'] = mini[:,2]
df['min_ended'] = mini[:,3]
df['max_created'] = maxi[:,0]
df['max_submitted'] = maxi[:,1]
df['max_started'] = maxi[:,2]
df['max_ended'] = maxi[:,3]

df = df.sort_values('created')
df = df.sort_values('submitted')
df.to_hdf('sample_simple.h5','table')
