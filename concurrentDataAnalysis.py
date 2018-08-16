import pandas as pd
import numpy as np
import multiprocessing
import datetime as dt
from functions import read_xfers

def get_context(t, xfers):
    cut = xfers[t.ended > xfers.submitted]
    cut = cut[t.submitted < cut.ended]
    return cut

def get_other_context(t, xfers):
    cut = xfers[xfers.ended > t.created]
    cut = cut[cut.created < t.ended]
    return cut

def get_yet_another_context(t, xfers):
    cut = xfers[xfers.created < t.submitted]
    cut = cut[cut.started > t.submitted]
    #cut.submitted = cut.submitted.map(lambda x: min(x, t.submitted))
    #cut.started = cut.started.map(lambda x: min(x, t.submitted))
    #cut.ended = cut.ended.map(lambda x: min(x, t.submitted))
    return cut

def get_no_context(t, xfers, delta=3):
    '''Get transfers outside the context, delta hours plus and minus'''
    cut1 = xfers[xfers.created > t.ended]
    cut1 = cut1[cut1.created < (t.ended + dt.timedelta(hours=delta))]
    cut2 = xfers[xfers.ended > (t.created - dt.timedelta(hours=delta))]
    cut2 = cut2[cut2.ended < t.created]
    return cut2.append(cut1)

def get_ancient_context(t, xfers, delta=3):
    '''Get transfers outside the context, delta hours plus and minus'''
    cut = xfers[xfers.ended < t.submitted]
    cut = cut[cut.ended > (t.created - dt.timedelta(hours=delta))]
    return cut

def generate_datasample(t, xfers, get_context_function, queue):
    context = get_context_function(t, xfers)
    context['tasq'] = (t.submitted - context.submitted).astype(int)//10**9
    context.tasq = context.tasq.clip_lower(0)
    context['tass'] = (t.created - context.submitted).astype(int)//10**9
    context.tass = context.tass.clip_lower(0)
    context['created'] = (context.created - t.created).astype(int)//10**9
    context['submitted'] = (context.submitted - t.created).astype(int)//10**9
    context['started'] = (context.started - t.created).astype(int)//10**9
    context['ended'] = (context.ended - t.created).astype(int)//10**9
    queue.put((t,context))


#src = 'MWT2'
#dst = 'AGLT2'
#src = 'BNL-ATLAS'
#dst = 'TRIUMF-LCG2'
#src = 'CERN-PROD'
#dst = 'MWT2'
#filepath = 'data/BNL_MWT2_20180313.csv'
#filepath = 'data/BNL_MWT2_20180405.csv'
#filepath = 'data/BNL_MWT2_20180618.csv'
#filepath = 'data/transfers-2018-04-15.csv'
filepath = 'data/transfers_20180616.csv'
#filepath = 'data/BNL_MWT2_20180619.csv'
#filepath = 'data/BNL_MWT2_20180515.csv'
#filepath = 'data/BNL_MWT2_20180522.csv' # failed due to high R²
#filepath = 'data/BNL_MWT2_20180429.csv'
#filepath = 'data/sample.csv'

def calculate_context_link_aware(get_context_name, get_context_function, xfers, src, dst, context_number):
    '''
    Calculate the stats takin the link into account
    context_number = 1: transfers going from src to dst only
        src_name == src and dst_name == dst
    context_number = 2: transfers going from dst to src only
        src_name == dst and dst_name == src
    context_number = 3: transfers exiting src while destination not dst nor src
        src_name == src and dst_name != dst and dst_name != src
    context_number = 4: transfers exiting dst while destination not src not dst
        src_name == dst and dst_name != src and dst_name != dst
    context_number = 5: transfers arriving dst but not from src nor dst
        dst_name == dst and src_name == src and src_name != dst
    context_number = 6: transfers arriving src but not from dst nor src
        dst_name == src and src_name == dst and src_name != src
    context_number = 7: transfers not going to nor from src nor dst
        src_name != src and dst_name 1= dst and dst_name != src and dst_name != dst
    context_number = 8: transfers that goes through fts-bnl
        external_host == 'https://fts.usatlas.bnl.gov:8446' 
    context_number = 9: transfers that doesn't go through fts-bnl
        external_host != 'https://fts.usatlas.bnl.gov:8446'
    '''
    xfers2 = xfers[xfers.submitted > (xfers.submitted.min() + dt.timedelta(hours=8))]
    xfers2 = xfers2[xfers2.src_name == src]
    xfers2 = xfers2[xfers2.dst_name == dst]
    if len(xfers2) > 8000:
        np.random.seed(42)
        xfers2 = xfers2.sample(8000)
    #xfers2 = xfers2[xfers2.external_host != 'https://fts.usatlas.bnl.gov:8446'].sample(5000)
    xfers2 = xfers2.set_index(np.arange(len(xfers2)))
    # get set for context calculation
    #Calculate the stats takin the link into account
    if context_number == 1:
        #context_number = 1: transfers going from src to dst only
        #print('src_name == src and dst_name == dst')
        print('src_name == {0} and dst_name == {1}'.format(src,dst))
        xfers3 = xfers[xfers.src_name == src]
        xfers3 = xfers3[xfers3.dst_name == dst]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 2:    
        #context_number = 2: transfers going from dst to src only
        #print('src_name == dst and dst_name == src')
        print('src_name == {1} and dst_name == {0}'.format(src,dst))
        xfers3 = xfers[xfers.src_name == dst]
        xfers3 = xfers3[xfers3.dst_name == src]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 3:    
        #context_number = 3: transfers exiting src while destination not dst nor src
        #print('src_name == src and dst_name != dst and dst_name != src')
        print('src_name == {0} and dst_name != {1} and dst_name != {0}'.format(src,dst))
        xfers3 = xfers[xfers.src_name == src]
        xfers3 = xfers3[xfers3.dst_name != dst]
        xfers3 = xfers3[xfers3.dst_name != src]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 4:
        #context_number = 4: transfers exiting dst while destination not src not dst
        #print('src_name == dst and dst_name != src and dst_name != dst')
        print('src_name == {1} and dst_name != {0} and dst_name != {1}'.format(src,dst))
        xfers3 = xfers[xfers.src_name == dst]
        xfers3 = xfers3[xfers3.dst_name != dst]
        xfers3 = xfers3[xfers3.dst_name != src]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 5:
        #context_number = 5: transfers arriving dst but not from src nor dst
        #print('dst_name == dst and src_name == src and src_name != dst')
        print('dst_name == {1} and src_name == {0} and src_name != {1}'.format(src,dst))
        xfers3 = xfers[xfers.dst_name == dst]
        xfers3 = xfers3[xfers3.src_name != dst]
        xfers3 = xfers3[xfers3.src_name != src]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 6:
        #context_number = 6: transfers arriving src but not from dst nor src
        print('dst_name == {0} and src_name == {1} and src_name != {0}'.format(src,dst))
        xfers3 = xfers[xfers.dst_name == src]
        xfers3 = xfers3[xfers3.src_name != dst]
        xfers3 = xfers3[xfers3.src_name != src]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
        #context_number = 1: src_name == src and dst_name == dst
    if context_number == 7:
        #context_number = 7: transfers not going to nor from src nor dst
        print('dst_name != {0} and src_name != {1} and src_name != {0} and dst_name != {1}'.format(src,dst))
        xfers3 = xfers[xfers.dst_name != src]
        xfers3 = xfers3[xfers3.src_name != dst]
        xfers3 = xfers3[xfers3.src_name != src]
        xfers3 = xfers3[xfers3.dst_name != dst]
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 8:
        #context_number = 8: transfers going through https://fts.usatlas.bnl.gov:8446
        print('external_host == https://fts.usatlas.bnl.gov:8446')
        xfers3 = xfers[xfers.external_host == 'https://fts.usatlas.bnl.gov:8446']
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
    if context_number == 9:
        #context_number = 9: transfers not going through https://fts.usatlas.bnl.gov:8446
        print('external_host != https://fts.usatlas.bnl.gov:8446')
        xfers3 = xfers[xfers.external_host != 'https://fts.usatlas.bnl.gov:8446']
        xfers3 = xfers3.set_index(np.arange(len(xfers3)))
        #context_number = 1: src_name == src and dst_name == dst
    print('len(xfers) (all the transfers):',len(xfers))
    print('len(xfers2) (transfers from src to dst):',len(xfers2))
    print('len(xfers3) (transfers in context):',len(xfers3))
    #print('len(link):',len(link))

    pnumber = 12
    target=[]; m=[]; s=[]; mini=[]; maxi=[]; i=0                                          
    for i in range(0,len(xfers2),pnumber):
        try:
            print(i, end='\r')
            jobs = []
            queue = multiprocessing.Queue()
            results = {}
            targets = {}
            for pn in range(pnumber):
                try:
                    process = multiprocessing.Process(target=generate_datasample, args=(xfers2.loc[i+pn], xfers3, get_context_function, queue))
        #        targets[pn] = xfers.loc[i+pn]
                    jobs.append(process)
                except KeyError:
                    print('pnumber to:', pn)
                    pnumber = pn
                    break
            for j in jobs:
                j.start()
            for pn in range(pnumber):
                results[pn] = queue.get()
            for j in jobs:
                j.join()
            for k in results:
                t, c = results[k]
                # t = targets[k]
                # c = generate_datasample(t, xfers)
                m.append([c.created.mean(), c.submitted.mean(), c.started.mean(), c.ended.mean(), c.tasq.mean(), c.tass.mean()])
                s.append([c.created.std(), c.submitted.std(), c.started.std(), c.ended.std(), c.tasq.std(), c.tass.std()])
                mini.append([c.created.min(), c.submitted.min(), c.started.min(), c.ended.min(), c.tasq.min(), c.tass.min()])
                maxi.append([c.created.max(), c.submitted.max(), c.started.max(), c.ended.max(), c.tasq.max(), c.tass.max()])
                target.append([t.id, len(c), t.RATE, t.RTIME, t.NTIME, t.QTIME, t.SIZE, t.created,t.submitted])
        except IndexError:
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
    df['mean_tasq'] = m[:,4]
    df['std_tasq'] = s[:,4]
    df['min_tasq'] = mini[:,4]
    df['max_tasq'] = maxi[:,4]
    df['mean_tass'] = m[:,5]
    df['std_tass'] = s[:,5]
    df['min_tass'] = mini[:,5]
    df['max_tass'] = maxi[:,5]

    df = df.sort_values('created')
    df = df.sort_values('submitted')
    #df.to_hdf('sample.h5','table')
    df.to_hdf('data/%s_stats_%s_%s_context%d_20180616.h5'%(get_context_name,src,dst, context_number),'table')

links = [
        'CERN-PROD__BNL-ATLAS', 'BNL-ATLAS__CERN-PROD',
        'CERN-PROD__FZK-LCG2', 'FZK-LCG2__CERN-PROD',
        'TRIUMF-LCG2__BNL-ATLAS', 'BNL-ATLAS__TRIUMF-LCG2',
        'CERN_PROD__MWT2', 'MWT2__CERN-PROD'
        ]

for link in links:
    src,dst = link.split('__')
    xfers, link2 = read_xfers(filepath, src, dst)
    #for i in range(1,10):
    #    print('Calculating context get_context', i)
    #    calculate_context_link_aware('get_context',get_context ,xfers, src, dst, i)
    #for i in range(1,10):
    #    print('Calculating context get_other_context', i)
    #    calculate_context_link_aware('get_other_context',get_other_context, xfers, src, dst, i)
    for i in range(1,10):
        print('Calculating context get_yet_another_context', i)
        calculate_context_link_aware('get_yet_another_context',get_yet_another_context, xfers, src, dst, i)
    #for i in range(1,10):
    #    print('Calculating context get_ancient_context', i)
    #    calculate_context_link_aware('get_ancient_context',get_ancient_context, xfers, src, dst, i)
    #for i in range(1,10):
    #    print('Calculating context get_no_context', i)
    #    calculate_context_link_aware('get_no_context',get_no_context, xfers, src, dst, i)
