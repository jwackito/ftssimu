import numpy as np
import pandas as pd

xfers = pd.read_hdf('data/transfers.h5', 'table') 

xfers.submited = pd.to_datetime(xfers.submited)
xfers.started = pd.to_datetime(xfers.started)
xfers.ended = pd.to_datetime(xfers.ended)

# actives
actives = np.array([0]*int((max(xfers.ended)-min(xfers.started)).total_seconds()))
offsets = (xfers.started - xfers.started.min()).values/10**9
for time,offset in zip((xfers.ended - xfers.started).values/10**9, offsets):
    act = np.array([1]*int(time), dtype=int)
    actives[int(offset):len(act)+int(offset)] += act
# queued
queued = np.array([0]*int((max(xfers.ended)-min(xfers.submited)).total_seconds()))
offsets = (xfers.submited - xfers.submited.min()).values/10**9
for time,offset in zip((xfers.started - xfers.submited).values/10**9, offsets):
    que = np.array([1]*int(time), dtype=int)
    queued[int(offset):len(que)+int(offset)] += que
# qo
qo = np.array([0]*int((max(xfers.ended)-min(xfers.started)).total_seconds()))
offsets = (xfers.started - xfers.started.min()).values/10**9
for time,offset in zip((xfers.ended - xfers.started).values/10**9, offsets):
    qo[int(offset)] += 1
# no
no = np.array([0]*int((max(xfers.ended)-min(xfers.started)).total_seconds()))
offsets = (xfers.ended - xfers.ended.min()).values/10**9
for time,offset in zip((xfers.ended - xfers.started).values/10**9, offsets):
    no[int(offset)] += 1
# rates
#rates = np.array([0.]*int((max(xfers.ended)-min(xfers.started)).total_seconds()))
#offsets = (xfers.started - xfers.started.min()).values/10**9
#for i in range(len(xfers)):
#    transfer = xfers.loc[i]
#    offset = (transfer.started - xfers.started.min()).total_seconds()
#    duration = (transfer.ended - transfer.started).total_seconds()
#    rate = [int(transfer['size'])/max(duration,1)] * int(duration)
#    rates[int(offset):len(rate)+int(offset)] += rate



queued_absolut = np.array([0]*int((max(xfers.ended)-min(xfers.submited)).total_seconds()))
actives_absolut = np.array([0]*int((max(xfers.ended)-min(xfers.submited)).total_seconds()))
#rates_absolut = np.array([0]*int((max(xfers.ended)-min(xfers.submited)).total_seconds()))
qo_absolut = np.array([0]*int((max(xfers.ended)-min(xfers.submited)).total_seconds()))
no_absolut = np.array([0]*(int((max(xfers.ended)-min(xfers.submited)).total_seconds())+6))

queued_absolut[:len(queued)] = queued
started_offset = int((xfers.started.min() - xfers.submited.min()).total_seconds())
ended_offset = int((xfers.ended.min() - xfers.submited.min()).total_seconds())
actives_absolut[started_offset:len(actives)+started_offset] = actives
#rates_absolut[started_offset:len(rates)+started_offset] = rates
qo_absolut[started_offset:len(qo)+started_offset] = qo
no_absolut[ended_offset:len(no)+ended_offset] = no

# smarter rates ;)
# get slices
slices = []
no_zero_index = np.flatnonzero(actives_absolut)
ini = no_zero_index[0]
previous = ini
i = 1
while i < len(no_zero_index):
    current = no_zero_index[i]
    if current - previous != 1:
        end = previous
        slices.append((ini,end))
        ini = current
    previous = current
    i += 1
offset = xfers.submited.min()
rates_absolut = [0]*len(queued_absolut)
for s, e in slices:
    s1 = offset + datetime.timedelta(seconds=int(s-1))
    e1 = offset + datetime.timedelta(seconds=int(e+1))  
    cut = xfers.where(xfers.started >= s1).dropna()
    cut = cut.where(cut.ended <= e1).dropna()
    average_mbps = sum(np.array(cut['size'],dtype=int))/len(cut)
    rates_absolut[s-5:e+5] = [average_mbps]*((e+5)-(s-5))


#plt.plot(queued_absolut, label='# queued', alpha=0.5)
#plt.plot(actives_absolut, label='# active', alpha=0.5)
#plt.plot(qo_absolut, label='q output', alpha=0.5)
#plt.plot(no_absolut, label='n output', alpha=0.5)
#plt.legend()
#plt.xlabel('Seconds since first submition')

real_active = actives_absolut
real_queued = queued_absolut
real_qoutput = qo_absolut
real_noutput = no_absolut

rates = pd.DataFrame(rates_absolut, columns=['rate'])
rates.to_csv('data/rates.csv',index=False)
actives = pd.DataFrame(real_active, columns=[ 'active'])
actives.to_csv('data/actives.csv',index=False)
