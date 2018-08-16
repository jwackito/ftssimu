import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

src = 'CERN-PROD'
dst = 'BNL-ATLAS'

hot_matrix = []
for i in range(1,10):
     df = pd.read_hdf('data/get_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
     hot_matrix.append(df.corr().qtime.values)
fig, ax = plt.subplots()
im = ax.imshow(np.array(hot_matrix), cmap='coolwarm', vmax=1, vmin=-1)
ax.set_xticks(np.arange(len(hot_matrix[0])))
ax.set_yticks(np.arange(len(hot_matrix)))
ax.set_yticklabels(['Set %d'%i for i in range(1,10)])
ax.set_xticklabels(df.corr().qtime.keys())
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for j in range(9):
    for i in range(len(df.corr().qtime.keys())):
        text = ax.text(i, j, '%.2f'%(hot_matrix[j][i]), ha="center", va="center", color="k")
plt.title('%s --> %s: get_context'%(src, dst))
hot_matrix = []
for i in range(1,10):
     df = pd.read_hdf('data/get_other_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
     hot_matrix.append(df.corr().qtime.values)
fig, ax = plt.subplots()
im = ax.imshow(np.array(hot_matrix), cmap='coolwarm', vmax=1, vmin=-1)
ax.set_xticks(np.arange(len(hot_matrix[0])))
ax.set_yticks(np.arange(len(hot_matrix)))
ax.set_yticklabels(['Set %d'%i for i in range(1,10)])
ax.set_xticklabels(df.corr().qtime.keys())
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.title('%s --> %s: get_other_context'%(src, dst))
for j in range(9):
    for i in range(len(df.corr().qtime.keys())):
        text = ax.text(i, j, '%.2f'%(hot_matrix[j][i]), ha="center", va="center", color="k")
hot_matrix = []
for i in range(1,10):
     df = pd.read_hdf('data/get_no_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
     hot_matrix.append(df.corr().qtime.values)
fig, ax = plt.subplots()
im = ax.imshow(np.array(hot_matrix), cmap='coolwarm', vmax=1, vmin=-1)
ax.set_xticks(np.arange(len(hot_matrix[0])))
ax.set_yticks(np.arange(len(hot_matrix)))
ax.set_yticklabels(['Set %d'%i for i in range(1,10)])
ax.set_xticklabels(df.corr().qtime.keys())
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.title('%s --> %s: get_no_context'%(src, dst))
for j in range(9):
    for i in range(len(df.corr().qtime.keys())):
        text = ax.text(i, j, '%.2f'%(hot_matrix[j][i]), ha="center", va="center", color="k")
