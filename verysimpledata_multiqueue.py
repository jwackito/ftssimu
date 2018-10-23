import numpy as np
import pandas as pd
import datetime
#from read import readxfers


#xfers = pd.read_hdf('data/xfers_wlinks_CERN-BNL_20180505-20180509.h5', 'table')
def foo(xfers):
    total_seconds = int((max(xfers.ended)-min(xfers.created)).total_seconds())
    link_queues = {}
    i = 0
    print('Initializing queues per link')
    listlen = len(list(set(xfers.rselink)))
    for link in list(set(xfers.rselink)):
        print('filling link %s (%d/%d)'%(link, i, listlen),end='\r')
        cut = xfers[xfers.rselink == link]
        link_queues[link] = {'nxfers': len(cut),
                             'current_source_output': 0,
                             'current_dest_input': 0,
                             'current_link_actives': 0,
                             'max_src_output': cut.iloc[0].soamax,
                             'max_dst_input': cut.iloc[0].diamax,
                             'max_link': cut.iloc[0].lamax,
                             'min_link': cut.iloc[0].lamin
                             }
        # actives
        actives = np.zeros(total_seconds)
        offsets = (cut.started - xfers.created.min()).values/10**9
        for time,offset in zip((cut.ended - cut.started).values/10**9, offsets):
            act = np.ones(int(time), dtype=int)
            actives[int(offset):len(act)+int(offset)] += act
        # queued
        queued = np.zeros(total_seconds)
        offsets = (cut.submitted - xfers.created.min()).values/10**9
        for time,offset in zip((cut.started - cut.submitted).values/10**9, offsets):
            que = np.ones(int(time), dtype=int)
            queued[int(offset):len(que)+int(offset)] += que
        link_queues[link]['real_actives'] = actives
        link_queues[link]['real_queued'] = queued
        i += 1
    return link_queues
#actives = pd.DataFrame(real_active, columns=[ 'active'])
#actives.to_csv('data/actives.csv',index=False)
#queued = pd.DataFrame(queued, columns=['queued'])
#queued.to_csv('data/queued.csv',index=False)
