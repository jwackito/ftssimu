##################
# MODEL 11 (Using rate prediction from a function)
##################

import numpy as np
import pandas as pd
import datetime
from scipy.optimize import least_squares
from sklearn.metrics import r2_score as r2


max_actives = pd.read_csv('data/actives.csv')['active'].values
current_active = 0
current_queued = 0

xfers = pd.read_hdf('data/transfers.h5', 'table')
#xfers = xfers.drop(columns=['started', 'ended'])
xfers.submited = pd.to_datetime(xfers.submited)
xfers.started = pd.to_datetime(xfers.started)
xfers.ended = pd.to_datetime(xfers.ended)

# FIT A FUNCTION FOR THE RATE
def objective(vars, x, data):
    rate = vars[0]
    overhead = vars[1]
    diskrw_limit = vars[2]
    model = x/((x/rate)+overhead)
    model[model>float(diskrw_limit)] = diskrw_limit
    return data - model

xfers['SIZE'] = np.array(xfers['size'],dtype=int)
xfers['N_RATE'] = xfers.SIZE/np.array((xfers.ended - xfers.started).values/10**9, dtype=int)
cut = xfers.where(xfers.N_RATE < np.inf).dropna()
vars = [xfers.N_RATE.median(), 1., 100.0]
seed = 43
rate = 1
overhead = 1
diskrw = 1
cut['N_PRED'] = cut.SIZE/((cut.SIZE/rate)+overhead)
cut['N_PRED'][cut['N_PRED']>diskrw]=diskrw
while r2(cut.N_RATE, cut.N_PRED) < 0.5:
    np.random.seed(seed)
    sample = cut.sample(500)
    out = least_squares(objective, vars, args=(sample.SIZE, sample.N_RATE),bounds=(0,np.inf))
    rate = out.x[0]
    overhead = out.x[1]
    diskrw = out.x[2]
    cut['N_PRED'] = cut.SIZE/((cut.SIZE/rate)+overhead)
    cut['N_PRED'][cut['N_PRED']>diskrw]=diskrw
    seed +=1
#print(seed-1)
#print(rate, overhead, diskrw)
#print(r2(cut.N_RATE, cut.N_PRED))
xfers['N_PRED'] = xfers.SIZE/((xfers.SIZE/rate)+overhead)
xfers['N_PRED'][xfers['N_PRED']>diskrw]=diskrw
do_plot = False
if do_plot:
    plt.plot(xfers.SIZE/(1024*1024), xfers.N_RATE/(1024*1024),'.', label='rate')
    plt.plot(xfers.SIZE/(1024*1024), xfers.N_PRED/(1024*1024),'.', label='pred')
    #plt.plot(cut.SIZE/(1024*1024), cut.N_PRED/(1024*1024),'.', label='pred')
    plt.xlabel('size in MB')
    plt.ylabel('rate in MB/s')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.title('rate as a function of the size')

current_time = min(xfers.submited)
offset_time = min(xfers.submited)
xfers_index = 0
queued_transfers = {}
active_transfers = {}
ended_transfers = {}
curr_xfer = xfers.loc[xfers_index]

queue_output = []
network_output = []
n_queued = []
n_active = []

step_nr = 0
while current_time < xfers.loc[len(xfers)-1].submited or current_active > 0:
    # add transfers to the queue and the netlink
    current_qoutput = 0
    current_noutput = 0
    while curr_xfer.submited <= current_time and xfers_index < len(xfers):
        queued_transfers[curr_xfer.id] = {'submited': curr_xfer.submited,
                                          'size': int(curr_xfer['size']),
                                          'real_rate': curr_xfer['N_RATE'],
                                          'pred_rate': curr_xfer['N_PRED'],
                                          'remaining': int(curr_xfer['size']),
                                          'state': 'queued'}
        curr_xfer = xfers.loc[xfers_index]
        xfers_index += 1
        current_queued += 1
    n_queued.append(len(queued_transfers))

    unqueue = []
    try:
        current_maxactive = max_actives[int((current_time - offset_time).total_seconds())]
        max_active = current_maxactive
    except IndexError:
        pass
    for xfer in queued_transfers:
        if current_active < max_active:
            active_transfers[xfer] = queued_transfers[xfer]
            active_transfers[xfer]['state'] = 'active'
            active_transfers[xfer]['started'] = current_time
            unqueue.append(xfer)
            current_active += 1
            current_qoutput += 1
            current_queued -= 1
        else:
            break
    for t in unqueue:
        queued_transfers.pop(t)
    queue_output.append(current_qoutput)

    current_time += datetime.timedelta(seconds=1)
    
    # simulate one step
    n_active.append(current_active)
    try:
        current_bw = rates[int((current_time - offset_time).total_seconds())]
        bandwidth = current_bw
    except IndexError:
        pass
    #print('STEP: %d' % step_nr)
    #print('active %d' % current_active)
    #print('bandwidth/current_active = %f' % (bandwidth/max(current_active,1)))

    step_nr += 1
    deactive = []
    for xfer in active_transfers:
        active_transfers[xfer]['remaining'] -= active_transfers[xfer]['pred_rate']
        if active_transfers[xfer]['remaining'] <= 0:
            ended_transfers[xfer] = active_transfers[xfer]
            ended_transfers[xfer]['state'] = 'finished'
            ended_transfers[xfer]['ended'] = current_time
            deactive.append(xfer)
    for t in deactive:
        active_transfers.pop(t)
        current_active -= 1
        current_noutput += 1
    network_output.append(current_noutput)

model11_active = n_active
model11_queued = n_queued
model11_qoutput = queue_output
model11_noutput = network_output

