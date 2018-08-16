##################
# MODEL 12 (Using rate prediction from a function)
#           Using min(1,actives)
##################

import numpy as np
import pandas as pd
import datetime
from scipy.optimize import least_squares
from sklearn.metrics import r2_score as r2
from read import readxfers

max_actives = pd.read_csv('data/actives.csv')['active'].values
current_active = 0
current_queued = 0

#xfers = pd.read_hdf('data/transfers.h5', 'table')
#xfers = xfers.drop(columns=['started', 'ended'])
#xfers.submitted = pd.to_datetime(xfers.submitted)
#xfers.started = pd.to_datetime(xfers.started)
#xfers.ended = pd.to_datetime(xfers.ended)
xfers = readxfers()
xfers.index = range(len(xfers))

# FIT A FUNCTION FOR THE RATE
def objective(vars, x, data):
    rate = vars[0]
    overhead = vars[1]
    diskrw_limit = vars[2]
    model = x/((x/rate)+overhead)
    model[model>float(diskrw_limit)] = diskrw_limit
    return data - model

#xfers['SIZE'] = np.array(xfers['SIZE'],dtype=int)
#xfers['RATE'] = xfers.SIZE/np.array((xfers.ended - xfers.started).values/10**9, dtype=int)
cut = xfers.where(xfers.RATE < np.inf).dropna()
vars = [xfers.RATE.median(), 1., 100.0]
seed = 43
rate = 1
overhead = 1
diskrw = 1
cut['NPRED'] = cut.SIZE/((cut.SIZE/rate)+overhead)
cut['NPRED'][cut['NPRED']>diskrw]=diskrw
while r2(cut.RATE, cut.NPRED) < 0.28:
    np.random.seed(seed)
    sample = cut.sample(500)
    out = least_squares(objective, vars, args=(sample.SIZE, sample.RATE),bounds=(0,np.inf))
    rate = out.x[0]
    overhead = out.x[1]
    diskrw = out.x[2]
    cut['NPRED'] = cut.SIZE/((cut.SIZE/rate)+overhead)
    cut['NPRED'][cut['NPRED']>diskrw]=diskrw
    seed +=1
#print(seed-1)
#print(rate, overhead, diskrw)
#print(r2(cut.RATE, cut.NPRED))
xfers['NPRED'] = xfers.SIZE/((xfers.SIZE/rate)+overhead)
xfers['NPRED'][xfers['NPRED']>diskrw]=diskrw
do_plot = False
if do_plot:
    plt.plot(xfers.SIZE/(1024*1024), xfers.RATE/(1024*1024),'.', label='rate')
    plt.plot(xfers.SIZE/(1024*1024), xfers.NPRED/(1024*1024),'.', label='pred')
    #plt.plot(cut.SIZE/(1024*1024), cut.NPRED/(1024*1024),'.', label='pred')
    plt.xlabel('SIZE in MB')
    plt.ylabel('rate in MB/s')
    plt.legend()
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    plt.title('rate as a function of the SIZE')

current_time = min(xfers.submitted)
offset_time = min(xfers.submitted)
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
while current_time < xfers.loc[len(xfers)-1].submitted or current_active > 0 or len(queued_transfers) > 0:
    # add transfers to the queue and the netlink
    current_qoutput = 0
    current_noutput = 0
    while curr_xfer.submitted <= current_time and xfers_index < len(xfers):
        queued_transfers[curr_xfer.id] = {'submitted': curr_xfer.submitted,
                                          'SIZE': int(curr_xfer['SIZE']),
                                          'real_rate': curr_xfer['RATE'],
                                          'pred_rate': curr_xfer['NPRED'],
                                          'remaining': int(curr_xfer['SIZE']),
                                          'state': 'queued'}
        curr_xfer = xfers.loc[xfers_index]
        xfers_index += 1
        current_queued += 1
    n_queued.append(len(queued_transfers))

    unqueue = []
    try:
        current_maxactive = max_actives[int((current_time - offset_time).total_seconds())]
        max_active = max(1,current_maxactive)
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

modelD_active = n_active
modelD_queued = n_queued
modelD_qoutput = queue_output
modelD_noutput = network_output
modelD_ended_transfers = ended_transfers