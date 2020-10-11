import pickle
from sklearn.metrics import r2_score as r2

dname = 'data/testpolydata_nxfers20_2019.csv'

#rmodel_name = 'data/polymodels3/model_RTIME_polydeg1_lss_trainedwith_link-agnostic_3110__linksactive_max-linkmeanqtime-retry_count-ftsqueued_median__.model'
rmodel_name = 'data/polymodels4/model_RTIME_polydeg1_lss_trainedwith_link-agnostic_21474__linkqueued_max-linksactive_max-linkmeanthr-ftsqueued_mean-ftsbqueued_median-ftsqueued_95p__.model'
#qmodel_name = 'data/polymodels3/model_QTIME_polydeg1_lss_trainedwith_link-agnostic_2174__linkqueued_median-linkmeanthr-ftsqueued_median-ftsbqueued_median__.model'
qmodel_name = 'data/polymodels4/model_QTIME_polydeg1_lss_trainedwith_link-agnostic_14294__linkbqueued_mean-linkqueued_median-linkbqueued_95p-linkqueued_max-linkmeanthr-SIZE__.model'
#nmodel_name = 'data/polymodels3/model_NTIME_polydeg1_lss_trainedwith_link-agnostic_3093__linksactive_max-linkmeanthr-retry_count-SIZE__.model'
nmodel_name = 'data/polymodels4/model_NTIME_polydeg1_lss_trainedwith_link-agnostic_9360__linkmeanthr-linkmeanqtime-SIZE-ftsbqueued_median-ftsqueued_95p__.model'

rmodel = pickle.load(open(rmodel_name,'rb'))
qmodel = pickle.load(open(qmodel_name,'rb'))
nmodel = pickle.load(open(nmodel_name,'rb'))
rmodel_columns = rmodel_name.split('__')[1].split('-')
qmodel_columns = qmodel_name.split('__')[1].split('-')
nmodel_columns = nmodel_name.split('__')[1].split('-')

df = pd.read_csv(dname)
df['stimes'] = pd.to_datetime(df.submitted)
df.created = pd.to_datetime(df.created)
df.submitted = pd.to_datetime(df.submitted)
df.started = pd.to_datetime(df.started)
df.ended = pd.to_datetime(df.ended)

rX = pd.DataFrame(df, columns=rmodel_columns).values
qX = pd.DataFrame(df, columns=qmodel_columns).values
nX = pd.DataFrame(df, columns=nmodel_columns).values

rpred = rmodel.predict(rX)
qpred = qmodel.predict(qX)
npred = nmodel.predict(nX)

df['rpred'] = rpred
df['qpred'] = qpred
df['npred'] = npred
df['pred'] = qpred

dtrainname = 'data/trainpolydata_1Million_xfers_2019.csv'

df_train = pd.read_csv(dtrainname)
df_train['stimes'] = pd.to_datetime(df_train.submitted) #index
stimes_train = df_train.stimes

df_test = pd.read_csv(dtrainname)
df_test['stimes'] = pd.to_datetime(df_test.submitted) #index
stimes_test = df_test.stimes

plt.subplots()
plt.plot(df.submitted, df.RTIME,'.', label='observed')
plt.plot(df.submitted, df.rpred,'.', label='predicted')
plt.plot(df_train.stimes, df_train.RTIME,'.', label='train')
plt.legend()
plt.title('Prediction of Rucio Queue time for poly model 9172 (R²: 0.18) linkbqueued_max-linksactive_max-linkmeanqtime-retry_count-ftsqueued_median')
plt.ylabel('Queue Time')
plt.xlabel('Submission Time')
print(r2(df.RTIME, df.rpred))

plt.subplots()
plt.plot(df.submitted, df.QTIME,'.', label='observed')
plt.plot(df.submitted, df.qpred,'.', label='predicted')
plt.plot(df_train.stimes, df_train.QTIME,'.', label='train')
plt.legend()
plt.title('Prediction of FTS Queue time for poly model 7341 (R²: 0.3106) linkqueued_median-linkmeanthr-linkmeanqtime-ftsqueued_median-ftsbqueued_median')
plt.ylabel('Queue Time')
plt.xlabel('Submission Time')
print(r2(df.QTIME, df.qpred))

plt.subplots()
plt.plot(df.submitted, df.NTIME,'.', label='observed')
plt.plot(df.submitted, df.npred,'.', label='predicted')
plt.plot(df_train.stimes, df_train.NTIME,'.', label='train')
plt.legend()
plt.title('Prediction of FTS Queue time for poly model 9293 (R²: 0.2823) linksactive_max-linkmeanthr-retry_count-SIZE-ftsqueued_95p')
plt.ylabel('Network Time')
plt.xlabel('Submission Time')
print(r2(df.NTIME, df.npred))

plt.subplots()
plt.plot(df.submitted, df.RTIME + df.QTIME + df.NTIME,'.', label='observed')
plt.plot(df.submitted, df.pred,'.', label='predicted')
plt.plot(df_train.stimes, df_train.QTIME,'.', label='train')
plt.legend()
plt.title('Prediction of TTC for poly model 14294__linkbqueued_mean-linkqueued_median-linkbqueued_95p-linkqueued_max-linkmeanthr-SIZE (R²: 0.6424)')
plt.ylabel('Time')
plt.xlabel('Creation Time')
print(r2(df.NTIME, df.npred))
