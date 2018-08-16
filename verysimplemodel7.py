import numpy as np
import pandas as pd
import datetime
rates = pd.read_csv('/home/jwackito/frames/rates_avg.csv')['rate'].values
rates = pd.rolling_window(rates, window=35, win_type='boxcar', center=True)
max_actives = pd.read_csv('/home/jwackito/frames/actives.csv')['active'].values
#max_active = 7 
current_active = 0
current_queued = 0

xfers = pd.read_hdf('/home/jwackito/frames/yrep.h5', 'table')
xfers = xfers.drop(columns=['started', 'ended'])
xfers.submited = pd.to_datetime(xfers.submited)

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
while current_time < xfers.loc[20000-1].submited or current_active > 0:
    # add transfers to the queue and the netlink
    current_qoutput = 0
    current_noutput = 0
    while curr_xfer.submited <= current_time and xfers_index < 20000:
        queued_transfers[curr_xfer.id] = {'submited': curr_xfer.submited, 'size': int(curr_xfer['size']), 'remaining': int(curr_xfer['size']), 'state': 'queued'}
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
        active_transfers[xfer]['remaining'] -= bandwidth
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

#plt.plot(n_queued, label='# queued', alpha=0.5)
#plt.plot(n_active, label='# active', alpha=0.5)
#plt.plot(queue_output, label='q output', alpha=0.5)
#plt.plot(network_output, label='n output', alpha=0.5)
#plt.legend()

#print('mean qo: ', np.mean(queue_output))
#print('mean no: ', np.mean(network_output))

model7_active = n_active
model7_queued = n_queued
model7_qoutput = queue_output
model7_noutput = network_output

plt.plot(model7_queued,label='model 7 queued')
#plt.plot(real_queued, label='real queued')
plt.plot(model7_active, label='model 7 active')
#plt.plot(real_active, label='real active')
plt.legend()
plt.xlabel('seconds since first submition')
plt.ylabel('number of transfers')
plt.title('Model 7')
