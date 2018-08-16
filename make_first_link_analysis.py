import datetime as dt
import pandas as pd
import numpy as np

from functions import read_xfers

finput = 'data/transfers_20180616.csv'
foutput = 'data/transfers_20180616_first_analisys.h5'

xfers,_ =read_xfers(finput,'','')
xfers2 = xfers[xfers.submitted > (xfers.submitted.min() + dt.timedelta(hours=8))]

l = []                                                                           
for sd in set([s + '__' + d for s,d in zip(xfers2.src_name.values, xfers2.dst_name.values)]):
    s = sd.split('__')[0]
    d = sd.split('__')[1]
    cut = xfers2[xfers2.src_name == s]
    cut = cut[cut.dst_name == d]
    lenx = len(cut)
    minq = cut.QTIME.min()
    maxq = cut.QTIME.max()
    meanq = cut.QTIME.mean()
    stdq = cut.QTIME.std()
    print(sd, lenx, minq, maxq, meanq, stdq)
    l.append([sd, lenx, minq, maxq, meanq, stdq])

ll = np.array(l)
dff = pd.DataFrame(ll, columns=['link', 'ncount', 'minq', 'maxq', 'meanq', 'stdq'])
dff.ncount = dff.ncount.values.astype(int)
dff.minq = dff.minq.values.astype(int)
dff.maxq = dff.maxq.values.astype(int) 
dff.meanq = dff.meanq.values.astype(float)
dff.stdq = dff.stdq.values.astype(float)

dff.to_hdf(foutput,'table')
