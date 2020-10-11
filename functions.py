import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from keras.layers import Embedding, Flatten, LSTM, Dense, Dropout
#from keras.models import Sequential
#from keras.preprocessing.text import Tokenizer

def get_regression_metrics(ytrue, ypred):
    from sklearn.metrics import r2_score, median_absolute_error, mean_squared_log_error, mean_squared_error, mean_absolute_error, max_error, explained_variance_score
    metrics = {}
    metrics['r2_score'] = r2_score(ytrue, ypred)
    metrics['explained_variance'] = explained_variance_score(ytrue, ypred)
    metrics['mean_absolute_error'] = mean_absolute_error(ytrue, ypred)
    metrics['median_absolute_error'] = median_absolute_error(ytrue, ypred)
    metrics['mean_squared_error'] = mean_squared_error(ytrue, ypred)
    metrics['mean_squared_log_error'] = mean_squared_log_error(ytrue, ypred)
    metrics['max_error'] = max_error(ytrue, ypred)
    return metrics


rsemap = pd.read_csv('/home/jwackito/FTSSimu/rseid2name_map.csv')
rsetiermap = pd.read_csv('/home/jwackito/FTSSimu/rse2tier.csv')
def get_context(t, xfers):
    cut = xfers[t.ended > xfers.submitted]
    cut = cut[t.submitted < cut.ended]
    return cut

def convert_to_sequences(xfers, tokenizer):
    l = xfers.to_csv(header=None, index=False).split('\n')
    l1 = [[s.replace(' ','_').replace(':', '_').replace(',', ' '). replace('_', '').replace('-','')] for s in l]
    l2 = [s[0] for s in l1][:-1]
    return tokenizer.texts_to_sequences(l2)

def id2name(rse_id):
    return rsemap[rsemap.id == rse_id].name

def id2type(rse_id):
    return rsemap[rsemap.id == rse_id].type

def id2staging(rse_id):
    return rsemap[rsemap.id == rse_id].staging

def id2tier(rse_id):
    return rsetiermap[rsetiermap.id == rse_id].tier

def preprocess(t, context, tokenizer):
    data = []
    # delete spureous columns
    context.pop('started')
    context.pop('ended')
    context.pop('anomalous1')
    context.pop('anomalous2')
    context.pop('NTIME')
    context.pop('QTIME')
    context.pop('RATE')
    # make timestamps relative to t.submitted
    context['dcreated'] = (context.created - t.submitted).values.astype(int)//10**9
    context['dsubmitted'] = (context.submitted - t.submitted).values.astype(int)//10**9
    #context['dstarted'] = (context.started - t.submitted).values.astype(int)//10**9
    #context['dended'] = (context.ended - t.submitted).values.astype(int)//10**9
    for c in context.itertuples():
        l = [a[0] for a in tokenizer.texts_to_sequences([c.account, c.activity, c.src_name, c.dst_name, c.source_rse_id, c.dest_rse_id])]
        l.append(c.dcreated)
        l.append(c.dsubmitted)
        l.append(c.SIZE)
        data.append(l)
    # append target information
    dcreated = int((t.created - t.submitted).total_seconds())
    dsubmitted = int((t.submitted - t.submitted).total_seconds())
    l = [a[0] for a in tokenizer.texts_to_sequences([t.account, t.activity, t.src_name, t.dst_name, t.source_rse_id, t.dest_rse_id])]
    l.append(dcreated)
    l.append(dsubmitted)
    l.append(t.SIZE)
    data.append(l)
    # append target
    data.append([t.anomalous2])
    return data 

shares = {                                                                      
    "Data_Brokering": 0.3,                                                      
    "Data_Consolidation": 0.2,                                                  
    "Data_Rebalancing": 0.5,                                                    
    "Default": 0.02,                                                            
    "Express": 0.4,                                                       
    "Functional_Test": 0.2,           
    "Production": 0.5,                 
    "Production_Input": 0.25,        
    "Production_Output": 0.25,                                                                                                                                
    "Recovery": 0.4,                          
    "Staging": 0.5,            
    "T0_Export": 0.7,                                                                                                                                         
    "T0_Tape": 0.7,            
    "User_Subscriptions": 0.1  
  }

def read_xfers(path, columns='data/transfers-20190606-20190731.header', nrows=0, skiprows=0):
    columns = pd.read_csv(columns).columns
    if nrows == 0:
        xfers = pd.read_csv(path, names=columns)
    else:
        xfers = pd.read_csv(path, names=columns, nrows=nrows, skiprows=skiprows)

    #xfers['proto'] = pd.read_csv('data/transfers_protocols-FTSBNL-20181001-20181016.csv')['dest_url']
    xfers['activity'] = [s.replace(' ', '_') for s in xfers.activity.values]
    xfers['created'] = pd.to_datetime(xfers.created)
    xfers['submitted'] = pd.to_datetime(xfers.submitted)
    xfers['started'] = pd.to_datetime(xfers.started)
    xfers['ended'] = pd.to_datetime(xfers.ended)
    xfers_len = len(xfers)
    xfers = xfers[xfers.RATE != np.inf]
    cut = xfers[xfers.QTIME < 0]
    xfers = xfers[xfers.QTIME >= 0]
    xfers = xfers.drop_duplicates(['id'])
    xfers = xfers.sort_values('submitted')
    xfers.index = range(len(xfers))
    print('Ignoring %d anomalous transfers (%0.2f%%)'%(xfers_len - len(xfers),(xfers_len-len(xfers))*100/xfers_len))
    print(set(cut.external_host))
    xfers['share'] = xfers.activity.map(shares)
    #xfers['link'] = xfers.src_rse + '__' + xfers.dst_rse  ## Malo malo muy malo!!
    #SD_link = xfers[xfers.src_name == src]
    #SD_link = SD_link[SD_link.dst_name == dst]
    return xfers #SD_link

def read_xfers_old(path, include_rsenames=True, nrows=0, skiprows=1):
    if nrows == 0:
        xfers = pd.read_csv(path, usecols=['account', 'activity', 'id', 'rule_id', 'src_name', 'dst_name', 'source_rse_id','dest_rse_id', 'external_host', 'created_at', 'started_at', 'submitted_at', 'transferred_at', 'bytes'])
    else:
        xfers = pd.read_csv(path, names=['account', 'activity', 'bytes', 'created_at', 'dest_rse_id', 'dest_url',
       'dst_name', 'external_host', 'external_id', 'id', 'name', 'priority',
       'retry_count', 'rule_id', 'scope', 'source_rse_id', 'src_name',
       'started_at', 'submitted_at', 'submitter_id', 'transferred_at',
       'updated_at'],
       nrows=nrows, skiprows=skiprows, usecols=['account', 'activity', 'id', 'rule_id', 'src_name', 'dst_name', 'source_rse_id','dest_rse_id', 'external_host', 'created_at', 'started_at', 'submitted_at', 'transferred_at', 'bytes'])

    #xfers['proto'] = pd.read_csv('data/transfers_protocols-FTSBNL-20181001-20181016.csv')['dest_url']
    xfers['activity'] = [s.replace(' ', '_') for s in xfers.activity.values]
    xfers['created'] = pd.to_datetime(xfers.created_at)
    xfers['submitted'] = pd.to_datetime(xfers.submitted_at)
    xfers['started'] = pd.to_datetime(xfers.started_at)
    xfers['ended'] = pd.to_datetime(xfers.transferred_at)
    #xfers['isubmitted'] = (xfers.submitted - xfers.submitted.min()).values.astype(int)//10**9
    #xfers['istarted'] = (xfers.started - xfers.submitted.min()).values.astype(int)//10**9
    #xfers['iended'] = (xfers.ended - xfers.submitted.min()).values.astype(int)//10**9
    xfers['SIZE'] = xfers.bytes
    xfers.pop('created_at')
    xfers.pop('submitted_at')
    xfers.pop('started_at')
    xfers.pop('transferred_at')
    xfers.pop('bytes')
    xfers['RTIME'] = ((xfers.submitted - xfers.created).values / 10**9).astype(int)
    xfers['NTIME'] = ((xfers.ended - xfers.started).values / 10**9).astype(int)
    xfers['QTIME'] = ((xfers.started - xfers.submitted).values / 10**9).astype(int)
    xfers['RATE'] = xfers.SIZE/xfers.NTIME
    #xfers['anomalous1'] = np.logical_and((xfers.NTIME.values > 780), (xfers.QTIME.values > 4600)).astype(int)
    #xfers['anomalous2'] = (xfers.RATE/xfers.SIZE < 0.02).astype(int)
    xfers_len = len(xfers)
    xfers = xfers[xfers.RATE != np.inf]
    cut = xfers[xfers.QTIME < 0]
    xfers = xfers[xfers.QTIME >= 0]
    xfers = xfers.drop_duplicates(['id'])
    xfers = xfers.sort_values('submitted')
    xfers.index = range(len(xfers))
    print('Ignoring %d anomalous transfers (%0.2f%%)'%(xfers_len - len(xfers),(xfers_len-len(xfers))*100/xfers_len))
    print(set(cut.external_host))
    if include_rsenames:
        rsemap = pd.read_csv('rseid2name_map.csv')
        rsetiermap = pd.read_csv('rse2tier.csv')
        id2staging={}
        for rse in rsemap.itertuples():
            id2staging[rse.id] = rse.staging
        id2type={}
        for rse in rsemap.itertuples():
            id2type[rse.id] = rse.type
        id2name={}
        for rse in rsemap.itertuples():
            id2name[rse.id] = rse.name
        id2tier={}
        for rse in rsetiermap.itertuples():
            id2tier[rse.id] = rse.tier
        xfers['src_rse_type'] = xfers.source_rse_id.map(id2type)
        xfers['dst_rse_type'] = xfers.dest_rse_id.map(id2type)
        xfers['src_rse_name'] = xfers.source_rse_id.map(id2name)
        xfers['dst_rse_name'] = xfers.dest_rse_id.map(id2name)
        #xfers['src_rse_staging'] = xfers.source_rse_id.map(id2staging)
        #xfers['dst_rse_staging'] = xfers.dest_rse_id.map(id2staging)
        xfers['src_tier'] = xfers.source_rse_id.map(id2tier)
        xfers['dst_tier'] = xfers.dest_rse_id.map(id2tier)
        xfers['src_tier'] = xfers.src_tier.fillna(4).astype(int)
        #xfers['src_tier'] = xfers.src_tier.astype(int)
        xfers['dst_tier'] = xfers.dst_tier.astype(int)
    xfers['link'] = xfers.src_name + '__'+ xfers.dst_name
    xfers['typetype'] = xfers.src_rse_type+'_'+xfers.dst_rse_type
    xfers['rselink'] = xfers.source_rse_id +'__'+xfers.dest_rse_id
    xfers['tiertier'] = xfers['src_tier'].astype(str)+'->'+xfers['dst_tier'].astype(str)
    xfers['share'] = xfers.activity.map(shares)
    #SD_link = xfers[xfers.src_name == src]
    #SD_link = SD_link[SD_link.dst_name == dst]
    return xfers #SD_link

def cut_outliers(xfers, column='QTIME', filter_suspected=True):
    p25 = xfers[column].quantile(.25)
    p75 = xfers[column].quantile(.75)
    IQR = p75 - p25
    outliers = xfers[xfers[column] > (p75+3*IQR)]
    outliers = outliers.append(xfers[xfers[column] < -(p25+3*IQR)])
    suspected1 = xfers[xfers[column] < (p75+3*IQR)]
    suspected1 = suspected1[suspected1[column] > -(p25+3*IQR)]
    suspected = suspected1[suspected1[column] > (p75+1.5*IQR)]
    suspected = suspected.append(suspected1[suspected1[column] < -(p25+1.5*IQR)])
    if filter_suspected:
        distance = 1.5
    else:
        distance = 3
    nooutliers = xfers[xfers[column] < (p75+distance*IQR)]
    nooutliers = nooutliers[nooutliers[column] > -(p25+distance*IQR)]
    return nooutliers,suspected,outliers

def percent(xfers, key, value):
    return len(xfers[xfers[key] == value])*100/len(xfers)


def get_host(url):
    '''Extract host from dest_url field'''
    proto = url.split(':')[0]
    host = url.split('://')[1].split(':')[0]
    port = url.split('://')[1].split(':')[1].split('/')[0]
    return proto + '://' + host + ':' + port

def histobox(data, title='title', xlabel='xlabel', ylabel='ylabel', bins=100):
    f,g = plt.subplots(2, sharex=True,gridspec_kw={"height_ratios": (.10, .90)})
    g[0].boxplot(data, whis=[5,95], showmeans=True, sym='.', vert=False)
    g[0].set_yticks([])
    g[0].grid()
    g[1].hist(data, bins=bins)
    g[1].grid()
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return f, g

def re(y, yhat):
    return abs(y - yhat)/y

def fogp(y, yhat, thr):
    e = re(y, yhat)
    return sum((e < thr).astype(int))/len(y)
