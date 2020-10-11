import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from functions import read_xfers

ftsstate = pd.read_csv('data/ftsstate_stats_all_2019.csv', nrows=30000000)
ftsstate['event_time'] = pd.to_datetime(ftsstate.event_time)

xfers = read_xfers('data/transfers-FTSBNL-20190606-20190731.csv', nrows=10000000)
rules = pd.read_csv('data/xfers_per_rule.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)
xfers['nxfers'] = xfers[['rule_id']].join(rules.set_index('ruleid'), on='rule_id').nxfers

mysample = xfers[(xfers.nxfers == 20) & (xfers.link.astype(str) != 'nan') & (xfers.state == 'D')]

def get_states(xferid):
    t = xfers[xfers.id == xferid].iloc[0]
    x2 = ftsstate[(ftsstate.event_time == t.submitted) & (ftsstate.linkname == t.link)]
    ftsqueued_mean = x2.fts_queued.mean()
    ftsqueued_median = np.median(x2.fts_queued)
    ftsqueued_95p = np.percentile(x2.fts_queued,95)
    ftsqueued_max = x2.fts_queued.max()
    ftsbqueued_mean = x2.fts_bytes_queued.mean()
    ftsbqueued_median = np.median(x2.fts_bytes_queued)
    ftsbqueued_95p = np.percentile(x2.fts_bytes_queued,95)
    ftsbqueued_max = x2.fts_bytes_queued.max()
    linkqueued_mean = x2.link_queued.mean()
    linkqueued_median = np.median(x2.link_queued)
    linkqueued_95p = np.percentile(x2.link_queued,95)
    linkqueued_max = x2.link_queued.max()
    linkbqueued_mean = x2.link_bytes_queued.mean()
    linkbqueued_median = np.median(x2.link_bytes_queued)
    linkbqueued_95p = np.percentile(x2.link_bytes_queued,95)
    linkbqueued_max = x2.link_bytes_queued.max()
    linksactive_max = x2.fts_active_links.max()
    linkmeanthr = x2.link_thr_bps.max()
    linkmeanqtime = x2.link_avg_qtime.max()
    return [ftsqueued_mean, ftsbqueued_mean, linkqueued_mean, linkbqueued_mean, ftsqueued_median, ftsbqueued_median, linkqueued_median, linkbqueued_median, ftsqueued_95p, ftsbqueued_95p, linkqueued_95p, linkbqueued_95p, ftsqueued_max, ftsbqueued_max, linkqueued_max, linkbqueued_max, linksactive_max, linkmeanthr, linkmeanqtime]
st = time.time()
statess = Parallel(n_jobs=12, backend='multiprocessing')(delayed(get_states)(xferid) for xferid in mysample.id.values)
print('Elapsed time for loop: %0.2f minutes'%((time.time() - st)/60))
statess = np.array(statess).T

mysample['ftsqueued_mean'] = statess[0]
mysample['ftsbqueued_mean'] = statess[1]
mysample['linkqueued_mean'] = statess[2]
mysample['linkbqueued_mean'] = statess[3]
mysample['ftsqueued_median'] = statess[4]
mysample['ftsbqueued_median'] = statess[5]
mysample['linkqueued_median'] = statess[6]
mysample['linkbqueued_median'] = statess[7]
mysample['ftsqueued_95p'] = statess[8]
mysample['ftsbqueued_95p'] = statess[9]
mysample['linkqueued_95p'] = statess[10]
mysample['linkbqueued_95p'] = statess[11]
mysample['ftsqueued_max'] = statess[12]
mysample['ftsbqueued_max'] = statess[13]
mysample['linkqueued_max'] = statess[14]
mysample['linkbqueued_max'] = statess[15]
mysample['linksactive_max'] = statess[16]
mysample['linkmeanthr'] = statess[17]
mysample['linkmeanqtime'] = statess[18]

mysample.to_csv('data/testpolydata_nxfers20_2019.csv', index=False)




# #single thread version...
#state_mean = []
#state_median = []
#state_95p = []
#state_max = []
#for t in mysample.itertuples():
#    x2 = ftsstate[(ftsstate.event_time == t.submitted) & (ftsstate.linkname == t.link)]
#    ftsqueued_mean = x2.fts_queued.mean()
#    ftsqueued_median = np.median(x2.fts_queued)
#    ftsqueued_95p = np.percentile(x2.fts_queued,95)
#    ftsqueued_max = x2.fts_queued.max()
#    ftsbqueued_mean = x2.fts_bytes_queued.mean()
#    ftsbqueued_median = np.median(x2.fts_bytes_queued)
#    ftsbqueued_95p = np.percentile(x2.fts_bytes_queued,95)
#    ftsbqueued_max = x2.fts_bytes_queued.max()
#    linkqueued_mean = x2.link_queued.mean()
#    linkqueued_median = np.median(x2.link_queued)
#    linkqueued_95p = np.percentile(x2.link_queued,95)
#    linkqueued_max = x2.link_queued.max()
#    linkbqueued_mean = x2.link_bytes_queued.mean()
#    linkbqueued_median = np.median(x2.link_bytes_queued)
#    linkbqueued_95p = np.percentile(x2.link_bytes_queued,95)
#    linkbqueued_max = x2.link_bytes_queued.max()
#    state_mean.append([ftsqueued_mean, ftsbqueued_mean, linkqueued_mean, linkbqueued_mean])
#    state_median.append([ftsqueued_median, ftsbqueued_median, linkqueued_median, linkbqueued_median])
#    state_95p.append([ftsqueued_95p, ftsbqueued_95p, linkqueued_95p, linkbqueued_95p])
#    state_max.append([ftsqueued_max, ftsbqueued_max, linkqueued_max, linkbqueued_max])


