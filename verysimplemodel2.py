import numpy as np
import pandas as pd
import datetime
bandwidth = (250*10**6)/8.
max_active = 100
current_active = 0


xfers = pd.read_hdf('/home/jwackito/frames/yrep.h5', 'table') 
xfers.submited = pd.to_datetime(xfers.submited)
transfers = []
for tr in xfers.itertuples():
    transfers.append((tr[1], int(tr[2])))

transfers = dict(transfers)
active = {} 
qo = []
no = []
act = []
transfers_stats = {}
step_nr = 0
transfers_remaining = len(xfers)
current_time = min(xfers.submited)
processed = [False] * len(xfers)
xfers.processed = processed

xfers_index = 0
while transfers_remaining > 0 or current_active > 0:
    # add transfers to the queue and the netlink
    current_qoutput = 0
    current_noutput = 0
    next_queued = xfers.where(xfers.submited < current_time + datetime.timedelta(seconds=1).dropna()
    next_queued = next_queued.where(next_queued.submited > current_time - datetime.timedelta(seconds=1))
    for tid, size, submited, _, _ , _ in next_queued.values:
        if current_active < max_active:
            active[tid] = size
            current_active += 1
            current_qoutput += 1
        else:
            break
    qo.append(current_qoutput)
    # simulate one step
    act.append(current_active)
    print('STEP: %d' % step_nr)
    print('active %d' % current_active)
    print('bandwidth/current_active = %f' % (bandwidth/current_active))

    step_nr += 1
    xfers_todelete = []
    for k in active.keys():
        active[k] -= bandwidth/current_active
        if active[k] <= 0:
            xfers_todelete.append(k)
    for k in xfers_todelete:
        active.pop(k)
        current_active -= 1
        current_noutput += 1
    no.append(current_noutput)
    current_time += datetime.timedelta(seconds=1)
plt.plot(qo)
plt.plot(no)
plt.plot(act)
