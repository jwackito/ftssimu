
TTC = []

for transfer in xfers.itertuples():
    xfer = {}
    xfer['id'] = transfer.id
    xfer['SUB_REAL'] = transfer.submited
    xfer['STR_REAL'] = transfer.started
    xfer['END_REAL'] = transfer.ended
    xfer['TTC_REAL'] = (transfer.ended - transfer.submited).total_seconds()
    xfer['TTCQ_REAL'] = (transfer.started - transfer.submited).total_seconds()
    xfer['TTCN_REAL'] = (transfer.ended - transfer.started).total_seconds()
    try:
        transferA = model9_ended_transfers[transfer.id]
        transferB = model10_ended_transfers[transfer.id]
        transferC = model11_ended_transfers[transfer.id]
        transferD = model12_ended_transfers[transfer.id]
        transferE = model13_ended_transfers[transfer.id]
    except:
        print('WARNING: transfer %s not in dict' % transfer.id)
        continue
    xfer['SUB_A'] = transferA['submited']
    xfer['STR_A'] = transferA['started']
    xfer['END_A'] = transferA['ended']
    xfer['TTC_A'] = (transferA['ended']-transferA['submited']).total_seconds()
    xfer['TTCQ_A'] = (transferA['started']-transferA['submited']).total_seconds()
    xfer['TTCN_A'] = (transferA['ended']-transferA['started']).total_seconds()

    xfer['SUB_B'] = transferB['submited']
    xfer['STR_B'] = transferB['started']
    xfer['END_B'] = transferB['ended']
    xfer['TTC_B'] = (transferB['ended']-transferB['submited']).total_seconds()
    xfer['TTCQ_B'] = (transferB['started']-transferB['submited']).total_seconds()
    xfer['TTCN_B'] = (transferB['ended']-transferB['started']).total_seconds()

    xfer['SUB_C'] = transferC['submited']
    xfer['STR_C'] = transferC['started']
    xfer['END_C'] = transferC['ended']
    xfer['TTC_C'] = (transferC['ended']-transferC['submited']).total_seconds()
    xfer['TTCQ_C'] = (transferC['started']-transferC['submited']).total_seconds()
    xfer['TTCN_C'] = (transferC['ended']-transferC['started']).total_seconds()

    xfer['SUB_D'] = transferD['submited']
    xfer['STR_D'] = transferD['started']
    xfer['END_D'] = transferD['ended']
    xfer['TTC_D'] = (transferD['ended']-transferD['submited']).total_seconds()
    xfer['TTCQ_D'] = (transferD['started']-transferD['submited']).total_seconds()
    xfer['TTCN_D'] = (transferD['ended']-transferD['started']).total_seconds()

    xfer['SUB_E'] = transferE['submited']
    xfer['STR_E'] = transferE['started']
    xfer['END_E'] = transferE['ended']
    xfer['TTC_E'] = (transferE['ended']-transferE['submited']).total_seconds()
    xfer['TTCQ_E'] = (transferE['started']-transferE['submited']).total_seconds()
    xfer['TTCN_E'] = (transferE['ended']-transferE['started']).total_seconds()

    TTC.append(xfer)

TTC = pd.DataFrame(TTC[:300])
yindex = range(len(TTC))
TTC['OFFSETS_Q'] = ((TTC.SUB_REAL - min(TTC.SUB_REAL)).values/10**9).astype(int)
TTC['OFFSETS_N'] = ((TTC.SUB_REAL - min(TTC.SUB_REAL)).values/10**9).astype(int) + TTC.TTCQ_REAL
xq_real = [(ini, fin) for ini,fin in zip(TTC.OFFSETS_Q,TTC.TTCQ_REAL + TTC.OFFSETS_Q)]
xn_real = [(ini, fin) for ini,fin in zip(TTC.OFFSETS_N,TTC.TTCN_REAL + TTC.OFFSETS_N)]
xq_model = [(ini, fin) for ini,fin in zip(TTC.OFFSETS_Q,TTC.TTCQ_E + TTC.OFFSETS_Q)]
xn_model = [(ini, fin) for ini,fin in zip(TTC.OFFSETS_N,TTC.TTCN_E + TTC.OFFSETS_N)]

for x,y in zip(xq_real,yindex):
    ini,fin = x
    plt.plot([ini, fin], [y,y], '-vk')
for x,y in zip(xn_real,yindex):
    ini,fin = x
    plt.plot([ini, fin], [y,y], '-^k')
print('real done')
for x,y in zip(xq_model,yindex):
    ini,fin = x
    plt.plot([ini, fin], [y,y], '-xb')
for x,y in zip(xn_model,yindex):
    ini,fin = x
    plt.plot([ini, fin], [y,y], '-+r')

import matplotlib.lines as mlines
rq = mlines.Line2D([], [], color='k', marker='v', label='real queue time')
rn = mlines.Line2D([], [], color='k', marker='^', label='real network time')
pq = mlines.Line2D([], [], color='b', marker='x', label='pred queue time')
pn = mlines.Line2D([], [], color='r', marker='+', label='pred network time')
plt.legend(handles=[rq, rn, pq, pn])
