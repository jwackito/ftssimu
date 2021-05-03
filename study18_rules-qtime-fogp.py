import multiprocessing
from joblib import Parallel, delayed
from statsmodels.tsa.ar_model import AutoReg
from functions import read_xfers

rules = pd.read_csv('data/xfers_per_rule_2.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
for d in ['min_created', 'min_submitted',
       'min_started', 'min_ended', 'max_created', 'max_submitted',
       'max_started', 'max_ended']:
    rules[d] = pd.to_datetime(rules[d])
rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9
rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)

#rules = rules[rules.createdatonce == 1]
# Tell pandas to treat the infinite as NaN
pd.set_option('use_inf_as_na', True)

xfers = read_xfers('data/xfers_sample.csv')

def re(y, yhat):
    return abs(y - yhat)/y

def fogp(y, yhat, thr):
    e = re(y, yhat)
    return sum((e < thr).astype(int))/len(y)


rttc = []
rqt = []
sumqt = []
meanqt = []
medianqt = []
for r in list(set(xfers.rule_id.values)):
    cut = xfers[xfers.rule_id == r]
    rttc.append(rules[rules.ruleid == r].ttc.values[0])
    rqt.append((cut.started.max() - cut.submitted.min()).total_seconds())
    sumqt.append(cut.QTIME.sum())
    meanqt.append(cut.QTIME.mean())
    medianqt.append(cut.QTIME.median()) 
rttc = np.array(rttc)
rqt = np.array(rqt)
sumqt = np.array(sumqt)
meanqt = np.array(meanqt)
medianqt = np.array(medianqt)

fogp1 = []
fogp2 = []
fogp3 = []
fogp4 = []
for tau in np.arange(0.01, 2, 0.01):
    fogp1.append(fogp(rttc,rqt,tau))
    fogp2.append(fogp(rttc,sumqt,tau))
    fogp3.append(fogp(rttc,meanqt,tau))
    fogp4.append(fogp(rttc,medianqt,tau))

plt.plot(np.arange(0.01, 2, 0.01), fogp1, label='$r_{QT}')
plt.plot(np.arange(0.01, 2, 0.01), fogp2, label='$r\prime_{QT} f=sum()')
plt.plot(np.arange(0.01, 2, 0.01), fogp3, label='$r\prime_{QT} f=mean()')
plt.plot(np.arange(0.01, 2, 0.01), fogp4, label='$r\prime_{QT} f=median()')
