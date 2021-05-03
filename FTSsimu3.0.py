from streamxfers import XferStreamer
#from scipy.optimize import least_squares
#from sklearn.metrics import r2_score as r2
import time
import numpy as np
import pandas as pd

def runsimu(nrows=10000):
    print('Running FTS simu 3.0 for %d transfers'%nrows)
    xs = XferStreamer('data/transfers-FTSBNL-20190606-20190731.csv', nrows=nrows)

    activities = list(set(xs.xfers.activity))
    activities.sort()
    links = list(set(xs.xfers.link))
    links[0] = 'nan'
    links.sort()
    links[links.index('nan')] = np.nan
    print(activities,file=open('data/activities_list.data','w'))
    print(links,file=open('data/links_list.data','w'))
    # can't save dates in np array of floats
    event_timestamps = []
    columns = [
        'event_time',
        'fts_queued',
        'fts_bytes_queued',
        'fts_thr_bps',
        'fts_submitted-ended',
        'fts_last_submitted',
        'fts_last_ended',
        'fts_active_links',
        'fts_avg_qtime',
        'fts_submitted_xfers',
        'fts_ended_xfers',
        'fts_failed_xfers',
        'link_queued',
        'link_bytes_queued',
        'link_thr_bps',
        'link_submitted-ended',
        'link_last_submitted',
        'link_last_ended',
        'link_active_activities',
        'link_avg_qtime',
        'link_submitted_xfers',
        'link_ended_xfers',
        'link_failed_xfers',
        'activity_queued',
        'activity_bytes_queued',
        'activity_thr_bps',
        'activity_submitted-ended',
        'activity_last_submitted',
        'activity_last_ended',
        'activity_avg_qtime',
        'activity_submitted_xfers',
        'activity_ended_xfers',
        'activity_failed_xfers',
        'link_offset',
        'activity_offset',
        'event_type'
    ]
# TRY TO COUNT THE DIFFERENT THINGS ACTIVE ( LINKS + LINKS*ACTIVITIES)
# MEANING: HOW MANY ACTIVE LINKS, AND FOR EACH ACTIVE LINK, HOW MANY ACTIVITIES.
# 
    print('Initializing FTS state...')
    FTS = {}
    FTS['queued'] = 0
    FTS['bytes_queued'] = 0
    FTS['thr_bps'] = 0
    FTS['submitted-ended'] = 0
    FTS['last_submitted'] = xs.xfers.created.min()
    FTS['last_ended'] = xs.xfers.created.min()
    FTS['active_links'] = 0
    FTS['avg_qtime'] = 0
    FTS['submitted_xfers'] = 0
    FTS['ended_xfers'] = 0
    FTS['failed_xfers'] = 0
    # creating links
    for link in list(set(xs.xfers.link)):
        FTS[link] = {}
        FTS[link]['queued'] = 0
        FTS[link]['bytes_queued'] = 0
        FTS[link]['thr_bps'] = 0
        FTS[link]['submitted-ended'] = 0
        FTS[link]['last_submitted'] = xs.xfers.created.min()
        FTS[link]['last_ended'] = xs.xfers.created.min()
        FTS[link]['active_activities'] = 0
        FTS[link]['avg_qtime'] = 0
        FTS[link]['submitted_xfers'] = 0
        FTS[link]['ended_xfers'] = 0
        FTS[link]['failed_xfers'] = 0
        for act in activities:
            FTS[link][act] = {}
            FTS[link][act]['queued'] = 0
            FTS[link][act]['bytes_queued'] = 0
            FTS[link][act]['thr_bps'] = 0
            FTS[link][act]['submitted-ended'] = 0
            FTS[link][act]['last_submitted'] = xs.xfers.created.min()
            FTS[link][act]['last_ended'] = xs.xfers.created.min()
            FTS[link][act]['avg_qtime'] = 0
            FTS[link][act]['submitted_xfers'] = 0
            FTS[link][act]['ended_xfers'] = 0
            FTS[link][act]['failed_xfers'] = 0

    def submit_transfer(t):
        # save FTS state first, then...
        event_timestamps.append(t.submitted)
        FTSstate = np.zeros(36)
        FTSstate[0] = 0
        FTSstate[1] = FTS['queued']
        FTSstate[2] = FTS['bytes_queued']
        FTSstate[3] = FTS['thr_bps']
        FTSstate[4] = FTS['submitted-ended']
        FTSstate[5] = (t.submitted - FTS['last_submitted']).total_seconds()
        FTSstate[6] = (t.submitted - FTS['last_ended']).total_seconds()
        FTSstate[7] = FTS['active_links']
        FTSstate[8] = FTS['avg_qtime']
        FTSstate[9] = FTS['submitted_xfers']
        FTSstate[10] = FTS['ended_xfers']
        FTSstate[11] = FTS['failed_xfers']
        FTSstate[12] = FTS[t.link]['queued']
        FTSstate[13] = FTS[t.link]['bytes_queued']
        FTSstate[14] = FTS[t.link]['thr_bps']
        FTSstate[15] = FTS[t.link]['submitted-ended']
        FTSstate[16] = (t.submitted - FTS[t.link]['last_submitted']).total_seconds()
        FTSstate[17] = (t.submitted - FTS[t.link]['last_ended']).total_seconds()
        FTSstate[18] = FTS[t.link]['active_activities']
        FTSstate[19] = FTS[t.link]['avg_qtime']
        FTSstate[20] = FTS[t.link]['submitted_xfers']
        FTSstate[21] = FTS[t.link]['ended_xfers']
        FTSstate[22] = FTS[t.link]['failed_xfers']
        FTSstate[23] = FTS[t.link][t.activity]['queued']
        FTSstate[24] = FTS[t.link][t.activity]['bytes_queued']
        FTSstate[25] = FTS[t.link][t.activity]['thr_bps']
        FTSstate[26] = FTS[t.link][t.activity]['submitted-ended']
        FTSstate[27] = (t.submitted - FTS[t.link][t.activity]['last_submitted']).total_seconds()
        FTSstate[28] = (t.submitted - FTS[t.link][t.activity]['last_ended']).total_seconds()
        FTSstate[29] = FTS[t.link][t.activity]['avg_qtime']
        FTSstate[30] = FTS[t.link][t.activity]['submitted_xfers']
        FTSstate[31] = FTS[t.link][t.activity]['ended_xfers']
        FTSstate[32] = FTS[t.link][t.activity]['failed_xfers']
        FTSstate[33] = links.index(t.link)
        FTSstate[34] = activities.index(t.activity)
        FTSstate[35] = 0

        # calculate the new FTS state
        FTS['queued'] += 1
        FTS['bytes_queued'] += t.SIZE
        FTS['last_submitted'] = t.submitted
        FTS['submitted-ended'] += 1
        if FTS[t.link]['queued'] == 0:
            FTS['active_links'] += 1
            #FTS['unique_load'] += 1
        FTS['submitted_xfers'] += 1
        
        FTS[t.link]['queued'] += 1
        FTS[t.link]['bytes_queued'] += t.SIZE
        FTS[t.link]['last_submitted'] = t.submitted
        FTS[t.link]['submitted-ended'] += 1
        if FTS[t.link][t.activity]['queued'] == 0:
            FTS[t.link]['active_activities'] += 1
            #FTS['unique_load'] += 1
        FTS[t.link]['submitted_xfers'] += 1
        
        FTS[t.link][t.activity]['queued'] += 1
        FTS[t.link][t.activity]['bytes_queued'] += t.SIZE
        FTS[t.link][t.activity]['last_submitted'] = t.submitted
        FTS[t.link][t.activity]['submitted-ended'] += 1
        FTS[t.link][t.activity]['submitted_xfers'] += 1
        return FTSstate

    def end_transfer(t):
        FTS['queued'] -= 1
        FTS['bytes_queued'] -= t.SIZE
        FTS['thr_bps'] = (FTS['thr_bps'] * FTS['ended_xfers'] + t.RATE)/(FTS['ended_xfers'] + 1)
        FTS['submitted-ended'] -= 1
        FTS['last_ended'] = t.ended 
        if FTS[t.link]['queued'] == 1:
            FTS['active_links'] -= 1
            #FTS['unique_load'] -= 1
        FTS['avg_qtime'] = (FTS['avg_qtime'] * FTS['ended_xfers'] + t.QTIME)/(FTS['ended_xfers'] + 1)
        FTS['ended_xfers'] += 1
        
        FTS[t.link]['queued'] -= 1
        FTS[t.link]['bytes_queued'] -= t.SIZE
        FTS[t.link]['thr_bps'] = (FTS[t.link]['thr_bps'] * FTS[t.link]['ended_xfers'] + t.RATE)/(FTS[t.link]['ended_xfers'] + 1)
        FTS[t.link]['submitted-ended'] -= 1
        FTS[t.link]['last_ended'] = t.ended
        if FTS[t.link][t.activity]['queued'] == 1:
            FTS[t.link]['active_activities'] -= 1
            #FTS['unique_load'] -= 1
        FTS[t.link]['avg_qtime'] = (FTS[t.link]['avg_qtime'] * FTS[t.link]['ended_xfers'] + t.QTIME)/(FTS[t.link]['ended_xfers'] + 1)
        FTS[t.link]['ended_xfers'] += 1

        FTS[t.link][t.activity]['queued'] -= 1
        FTS[t.link][t.activity]['bytes_queued'] -= t.SIZE
        FTS[t.link][t.activity]['thr_bps'] = (FTS[t.link][t.activity]['thr_bps'] * FTS[t.link][t.activity]['ended_xfers']
                + t.RATE)/(FTS[t.link][t.activity]['ended_xfers'] + 1)
        FTS[t.link][t.activity]['submitted-ended'] -= 1 
        FTS[t.link][t.activity]['last_ended'] = t.ended
        FTS[t.link][t.activity]['avg_qtime'] = (FTS[t.link][t.activity]['avg_qtime'] * FTS[t.link][t.activity]['ended_xfers']
                + t.QTIME)/(FTS[t.link][t.activity]['ended_xfers'] + 1)
        FTS[t.link][t.activity]['ended_xfers'] += 1

        # save FTS state first, then...
        event_timestamps.append(t.ended)
        FTSstate = np.zeros(36)
        FTSstate[0] = 0  # reserved for event time...
        FTSstate[1] = FTS['queued']
        FTSstate[2] = FTS['bytes_queued']
        FTSstate[3] = FTS['thr_bps']
        FTSstate[4] = FTS['submitted-ended']
        FTSstate[5] = (t.ended - FTS['last_submitted']).total_seconds()
        FTSstate[6] = (t.ended - FTS['last_ended']).total_seconds()
        FTSstate[7] = FTS['active_links']
        FTSstate[8] = FTS['avg_qtime']
        FTSstate[9] = FTS['submitted_xfers']
        FTSstate[10] = FTS['ended_xfers']
        FTSstate[11] = FTS['failed_xfers']
        FTSstate[12] = FTS[t.link]['queued']
        FTSstate[13] = FTS[t.link]['bytes_queued']
        FTSstate[14] = FTS[t.link]['thr_bps']
        FTSstate[15] = FTS[t.link]['submitted-ended']
        FTSstate[16] = (t.ended - FTS[t.link]['last_submitted']).total_seconds()
        FTSstate[17] = (t.ended - FTS[t.link]['last_ended']).total_seconds()
        FTSstate[18] = FTS[t.link]['active_activities']
        FTSstate[19] = FTS[t.link]['avg_qtime']
        FTSstate[20] = FTS[t.link]['submitted_xfers']
        FTSstate[21] = FTS[t.link]['ended_xfers']
        FTSstate[22] = FTS[t.link]['failed_xfers']
        FTSstate[23] = FTS[t.link][t.activity]['queued']
        FTSstate[24] = FTS[t.link][t.activity]['bytes_queued']
        FTSstate[25] = FTS[t.link][t.activity]['thr_bps']
        FTSstate[26] = FTS[t.link][t.activity]['submitted-ended']
        FTSstate[27] = (t.ended - FTS[t.link][t.activity]['last_submitted']).total_seconds()
        FTSstate[28] = (t.ended - FTS[t.link][t.activity]['last_ended']).total_seconds()
        FTSstate[29] = FTS[t.link][t.activity]['avg_qtime']
        FTSstate[30] = FTS[t.link][t.activity]['submitted_xfers']
        FTSstate[31] = FTS[t.link][t.activity]['ended_xfers']
        FTSstate[32] = FTS[t.link][t.activity]['failed_xfers']
        FTSstate[33] = links.index(t.link)
        FTSstate[34] = activities.index(t.activity)
        FTSstate[35] = 1  # event type
        return FTSstate
    
    def fail_transfer(t):
        FTS['queued'] -= 1
        FTS['bytes_queued'] -= t.SIZE
        #FTS['thr_bps'] = (FTS['thr_bps'] * FTS['ended_xfers'] + t.RATE)/(FTS['ended_xfers'] + 1)
        #FTS['submitted-ended'] -= 1
        #FTS['last_ended'] = t.ended 
        if FTS[t.link]['queued'] == 1:
            FTS['active_links'] -= 1
            #FTS['unique_load'] -= 1
        FTS['avg_qtime'] = (FTS['avg_qtime'] * FTS['ended_xfers'] + t.QTIME)/(FTS['ended_xfers'] + 1)
        FTS['failed_xfers'] += 1
        
        FTS[t.link]['queued'] -= 1
        FTS[t.link]['bytes_queued'] -= t.SIZE
        #FTS[t.link]['thr_bps'] = (FTS[t.link]['thr_bps'] * FTS[t.link]['ended_xfers'] + t.RATE)/(FTS[t.link]['ended_xfers'] + 1)
        #FTS[t.link]['submitted-ended'] -= 1
        #FTS[t.link]['last_ended'] = t.ended
        if FTS[t.link][t.activity]['queued'] == 1:
            FTS[t.link]['active_activities'] -= 1
            #FTS['unique_load'] -= 1
        FTS[t.link]['avg_qtime'] = (FTS[t.link]['avg_qtime'] * FTS[t.link]['ended_xfers'] + t.QTIME)/(FTS[t.link]['ended_xfers'] + 1)
        FTS[t.link]['failed_xfers'] += 1

        FTS[t.link][t.activity]['queued'] -= 1
        #FTS[t.link][t.activity]['bytes_queued'] -= t.SIZE
        #FTS[t.link][t.activity]['thr_bps'] = (FTS[t.link][t.activity]['thr_bps'] * FTS[t.link][t.activity]['ended_xfers']
        #        + t.RATE)/(FTS[t.link][t.activity]['ended_xfers'] + 1)
        FTS[t.link][t.activity]['submitted-ended'] -= 1 
        #FTS[t.link][t.activity]['last_ended'] = t.ended
        FTS[t.link][t.activity]['avg_qtime'] = (FTS[t.link][t.activity]['avg_qtime'] * FTS[t.link][t.activity]['ended_xfers']
                + t.QTIME)/(FTS[t.link][t.activity]['ended_xfers'] + 1)
        FTS[t.link][t.activity]['failed_xfers'] += 1

        # save FTS state first, then...
        event_timestamps.append(t.ended)
        FTSstate = np.zeros(36)
        FTSstate[0] = 0  # reserved for event time...
        FTSstate[1] = FTS['queued']
        FTSstate[2] = FTS['bytes_queued']
        FTSstate[3] = FTS['thr_bps']
        FTSstate[4] = FTS['submitted-ended']
        FTSstate[5] = (t.ended - FTS['last_submitted']).total_seconds()
        FTSstate[6] = (t.ended - FTS['last_ended']).total_seconds()
        FTSstate[7] = FTS['active_links']
        FTSstate[8] = FTS['avg_qtime']
        FTSstate[9] = FTS['submitted_xfers']
        FTSstate[10] = FTS['ended_xfers']
        FTSstate[11] = FTS['failed_xfers']
        FTSstate[12] = FTS[t.link]['queued']
        FTSstate[13] = FTS[t.link]['bytes_queued']
        FTSstate[14] = FTS[t.link]['thr_bps']
        FTSstate[15] = FTS[t.link]['submitted-ended']
        FTSstate[16] = (t.ended - FTS[t.link]['last_submitted']).total_seconds()
        FTSstate[17] = (t.ended - FTS[t.link]['last_ended']).total_seconds()
        FTSstate[18] = FTS[t.link]['active_activities']
        FTSstate[19] = FTS[t.link]['avg_qtime']
        FTSstate[20] = FTS[t.link]['submitted_xfers']
        FTSstate[21] = FTS[t.link]['ended_xfers']
        FTSstate[22] = FTS[t.link]['failed_xfers']
        FTSstate[23] = FTS[t.link][t.activity]['queued']
        FTSstate[24] = FTS[t.link][t.activity]['bytes_queued']
        FTSstate[25] = FTS[t.link][t.activity]['thr_bps']
        FTSstate[26] = FTS[t.link][t.activity]['submitted-ended']
        FTSstate[27] = (t.ended - FTS[t.link][t.activity]['last_submitted']).total_seconds()
        FTSstate[28] = (t.ended - FTS[t.link][t.activity]['last_ended']).total_seconds()
        FTSstate[29] = FTS[t.link][t.activity]['avg_qtime']
        FTSstate[30] = FTS[t.link][t.activity]['submitted_xfers']
        FTSstate[31] = FTS[t.link][t.activity]['ended_xfers']
        FTSstate[32] = FTS[t.link][t.activity]['failed_xfers']
        FTSstate[33] = links.index(t.link)
        FTSstate[34] = activities.index(t.activity)
        FTSstate[35] = 2  # event type
        return FTSstate

    ftsstates = []
    i = 0
    for s, e in xs:
        st = time.time()
        # first process the ended transfers
        for t in e.itertuples():
            if t.state == 'D':
                ftsstates.append(end_transfer(t))
            if t.state == 'F':
                ftsstates.append(fail_transfer(t))
            i += 1
            if i%100000 == 0:
                print('Saving %d finished events'%i)
                f = pd.DataFrame(np.array(ftsstates), columns=columns)
                f['event_time'] = event_timestamps
                f['linkname'] = f.link_offset.map(lambda x: links[int(x)])
                f['activityname'] = f.activity_offset.map(lambda x: activities[int(x)])
                f.to_csv('data/ftsstate_stats_all_2019.csv', index=False)
        # then the new submissions
        # VERY FUCKING IMPORTANT: This way, FTSstate is the previous state for the new submission.
        for t in s.itertuples():
            ftsstates.append(submit_transfer(t))

        print(i, len(xs.xfers), '%0.4f'%(time.time() - st), end='\r')
       

    print('Saving ftsstate_stats_all.csv')
    f = pd.DataFrame(np.array(ftsstates), columns=columns)
    f['event_time'] = event_timestamps
    f.to_csv('data/ftsstate_stats_all_2019_%08d.csv'%nrows, index=False)

runsimu(30000000)
