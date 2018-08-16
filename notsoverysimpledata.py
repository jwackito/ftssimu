import numpy as np
import pandas as pd
import datetime

from pandas.tools.plotting import scatter_matrix

# queued
def calculate_queued(cut):
    queued = np.array([0]*int((xfers.ended.values.max() - xfers.submitted.values.min())/10**9))
    offsets = ((cut.submitted - xfers.submitted.min()).values/10**9).astype(int)
    for time, offset in zip(((cut.started - cut.submitted).values/10**9).astype(int), offsets):
        if time > 0:
            que = np.array([1]*time)
            queued[offset:len(que)+offset] += que
    return queued

# actives
def calculate_actives(cut):
    actives = np.array([0]*int((xfers.ended.values.max() - xfers.submitted.values.min())/10**9))
    offsets = ((cut.started - xfers.submitted.min()).values/10**9).astype(int)
    for time, offset in zip(((cut.ended - cut.started).values/10**9).astype(int), offsets):
        if time > 0:
            act = np.array([1]*time)
            actives[offset:len(act)+offset] += act
    return actives

print('Loading dataset...')
xfers = pd.read_csv('data/transfers-2018-04-15.csv', usecols=['activity', 'bytes', 'created_at', 'source_rse_id','dest_rse_id', 'id', 'src_name', 'dst_name','started_at', 'submitted_at', 'transferred_at', 'updated_at'])

xfers['submitted'] = pd.to_datetime(xfers.submitted_at)
xfers['started'] = pd.to_datetime(xfers.started_at)
xfers['ended'] = pd.to_datetime(xfers.transferred_at)
xfers['SIZE'] = xfers.bytes
xfers['NTIME'] = ((xfers.ended - xfers.started).values / 10**9).astype(int)
xfers['QTIME'] = ((xfers.started - xfers.submitted).values / 10**9).astype(int)
xfers['RATE'] = xfers.SIZE/xfers.NTIME
#xfers['src_cat'] = pd.Categorical(xfers.src_name,ordered=False)
#xfers['dst_cat'] = pd.Categorical(xfers.dst_name,ordered=False)
#xfers['src_cat_code'] = xfers.src_cat.cat.codes
#xfers['dst_cat_code'] = xfers.dst_cat.cat.codes

print('Creating extra observables...')
total_actives = calculate_actives(xfers) 
avg_actives = [] 
xfers['started_offset'] = ((xfers.started - xfers.submitted.min()).values/10**9).astype(int)
for transfer in xfers.itertuples():
    avg_actives.append(np.mean(total_actives[transfer.started_offset:transfer.started_offset+transfer.NTIME]))
#xfers.pop('started_offset')
xfers['avg_actives'] = avg_actives

src = 'MWT2'
dst = 'AGLT2'

print ('Removing outliers...')
SD_link = xfers.where(xfers.src_name == src).dropna()
SD_link = SD_link.where(SD_link.dst_name == dst).dropna()

DS_link = xfers.where(xfers.src_name == dst).dropna()
DS_link = DS_link.where(DS_link.dst_name == src).dropna()

S_egressing = xfers.where(xfers.src_name == src).dropna()
S_egressing = S_egressing.where(S_egressing.dst_name != dst).dropna()
D_egressing = xfers.where(xfers.src_name == dst).dropna()
D_egressing = D_egressing.where(D_egressing.dst_name != src).dropna()
S_ingressing = xfers.where(xfers.dst_name == src).dropna()
S_ingressing = S_ingressing.where(S_ingressing.src_name != dst).dropna()
D_ingressing = xfers.where(xfers.dst_name == dst).dropna()
D_ingressing = D_ingressing.where(D_ingressing.src_name != src).dropna()


queued_SD_link = calculate_queued(SD_link)
queued_DS_link = calculate_queued(DS_link)
queued_S_egressing = calculate_queued(S_egressing)
queued_D_egressing = calculate_queued(D_egressing)
queued_S_ingressing = calculate_queued(S_ingressing)
queued_D_ingressing = calculate_queued(D_ingressing)

actives_SD_link = calculate_actives(SD_link)
actives_DS_link = calculate_actives(DS_link)
actives_S_egressing = calculate_actives(S_egressing)
actives_D_egressing = calculate_actives(D_egressing)
actives_S_ingressing = calculate_actives(S_ingressing)
actives_D_ingressing = calculate_actives(D_ingressing)


avg_actives_SD_link = []
avg_actives_DS_link = []
avg_actives_S_egress = []
avg_actives_D_egress = []
avg_actives_S_ingress = []
avg_actives_D_ingress = []

print ('Removing data that doesn\'t fit...')
for i, t in SD_link.iterrows():
    duration = int((t.ended - t.started).total_seconds())
    idx = int((t.started - xfers.submitted.min()).total_seconds())
    avg_actives_SD_link.append(np.mean(actives_SD_link[idx:idx+duration]))
    avg_actives_DS_link.append(np.mean(actives_DS_link[idx:idx+duration]))
    avg_actives_S_egress.append(np.mean(actives_S_egressing[idx:idx+duration]))
    avg_actives_D_egress.append(np.mean(actives_D_egressing[idx:idx+duration]))
    avg_actives_S_ingress.append(np.mean(actives_S_ingressing[idx:idx+duration]))
    avg_actives_D_ingress.append(np.mean(actives_D_ingressing[idx:idx+duration]))

print('Plotting...')
SD_link['NTIME'] = ((SD_link.ended - SD_link.started).values / 10**9).astype(int)
SD_link['RATE'] = SD_link.SIZE/SD_link.NTIME

SD_link['ACT_SD_LINK'] = avg_actives_SD_link
SD_link['ACT_DS_LINK'] = avg_actives_DS_link
SD_link['ACT_S_EGRES'] = avg_actives_S_egress
SD_link['ACT_D_EGRES'] = avg_actives_D_egress
SD_link['ACT_S_INGRES'] = avg_actives_S_ingress
SD_link['ACT_D_INGRES'] = avg_actives_D_ingress
SD_link['ACT_LINK_LOAD'] = SD_link['ACT_SD_LINK'] + SD_link['ACT_S_EGRES'] + SD_link['ACT_D_INGRES']


scatter_matrix(SD_link)
SD_link.plot(kind='scatter',x='SIZE', y='RATE', s=SD_link.NTIME, c='ACT_LINK_LOAD',colormap='brg')
SD_link.plot(kind='scatter',x='SIZE', y='RATE', s=SD_link.NTIME, c='ACT_SD_LINK',colormap='brg')
SD_link.plot(kind='scatter',x='SIZE', y='RATE', s=SD_link.NTIME, c='started_offset',colormap='brg')
xfers.plot(kind='scatter',x='NTIME', y='RATE', s=4, c='SIZE',colormap='brg')
xfers.plot(kind='scatter',x='avg_actives', y='QTIME', s=4, c='NTIME',colormap='brg')

