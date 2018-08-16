import numpy as np
import pandas as pd
bandwidth = (250*10**6)/8.
max_active = 100
current_active = 0


xfers = pd.read_hdf('/home/jwackito/frames/xyrep.h5', 'table') 
transfers = []
i = 0
for tr in xfers.itertuples():
    transfers.append((tr[1], int(tr[2])))
#    if i > 1000:
#        break
#    else:
#        i+=1
transfers = dict(transfers)
active = {} 
qo = []
no = []
act = []
transfers_stats = {}
step_nr = 0
while len(transfers) > 0 or current_active > 0:
    # add transfers to netlink
    current_qoutput = 0
    current_noutput = 0
    xfers_todelete = []
    for k in transfers.keys():
        if current_active < max_active:
            active[k] = transfers[k]
            xfers_todelete.append(k)
            current_active += 1
        else:
            break
    for k in xfers_todelete:
        transfers.pop(k)
        current_qoutput += 1
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
plt.plot(qo)
plt.plot(no)
plt.plot(act)
