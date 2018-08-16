import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

xfers = pd.read_csv('data/transfers-2018-04-15.csv', usecols=['activity', 'bytes', 'created_at', 'source_rse_id','dest_rse_id', 'id', 'src_name', 'dst_name','started_at', 'submitted_at', 'transferred_at', 'updated_at'])

xfers['submitted'] = pd.to_datetime(xfers.submitted_at)
xfers['started'] = pd.to_datetime(xfers.started_at)
xfers['ended'] = pd.to_datetime(xfers.transferred_at)
xfers['SIZE'] = xfers.bytes
xfers['N_TIME'] = ((xfers.ended - xfers.started).values / 10**9).astype(int)
xfers['Q_TIME'] = ((xfers.started - xfers.submitted).values / 10**9).astype(int)
xfers['RATE'] = xfers.SIZE/xfers.N_TIME
xfers['src_cat'] = pd.Categorical(xfers.src_name,ordered=False)
xfers['dst_cat'] = pd.Categorical(xfers.dst_name,ordered=False)
xfers['src_cat_code'] = xfers.src_cat.cat.codes
xfers['dst_cat_code'] = xfers.dst_cat.cat.codes

src = 'MWT2'
dst = 'AGLT2'

SD_link = xfers.where(xfers.src_name == src).dropna()
SD_link = SD_link.where(SD_link.dst_name == dst).dropna()

DS_link = xfers.where(xfers.src_name == dst).dropna()
DS_link = DS_link.where(DS_link.dst_name == src).dropna()

S_egressing = xfers.where(xfers.src_name == src).dropna()
S_egressing = S_egressing.where(S_egressing.dst_name != dst).dropna()
D_egressing = xfers.where(xfers.src_name == dst).dropna()
D_egressing = D_egressing.where(D_egressing.dst_name != src).dropna()
S_ingressing = xfers.where(xfers.dst_name == src).dropna()
S_ingressing = S_ingressing.where(S_ingressing.src_name != dst).dropna()
D_ingressing = xfers.where(xfers.dst_name == dst).dropna()
D_ingressing = D_ingressing.where(D_ingressing.src_name != src).dropna()

