import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

src = 'CERN-PROD'
dst = 'BNL-ATLAS'

x = 'max_submitted'
y = 'qtime'
c = 'ncount'
xset = 6
for i in range(xset, xset+1):
    df = pd.read_hdf('data/get_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
    df = df[df.qtime > 0]
    #df = df[df.max_submitted > 0]
    df.plot(kind='scatter', x=x, y=y, c=c, s=2, cmap='brg')
    #plt.xscale('log')
    plt.yscale('log')
    plt.title('%s --> %s: Set %d'%(src,dst, i))
    plt.text(2,df.qtime.max(), "Transfers with qtime < 1000 sec: %0.2f%%"%(len(df[df.qtime < 1000])*100/ len(df)), bbox=dict(facecolor='red', alpha=0.5))
    df = pd.read_hdf('data/get_other_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
    df = df[df.qtime > 0]
    #df = df[df.max_submitted > 0]
    df.plot(kind='scatter', x=x, y=y, c=c, s=2, cmap='brg')
    #plt.xscale('log')
    plt.yscale('log')
    plt.title('%s --> %s: Set %d'%(src,dst, i))
    plt.text(2,df.qtime.max(), "Transfers with qtime < 1000 sec: %0.2f%%"%(len(df[df.qtime < 1000])*100/ len(df)), bbox=dict(facecolor='red', alpha=0.5))
    df = pd.read_hdf('data/get_no_context_stats_%s_%s_context%d_20180616.h5'%(src,dst,i),'table')
    df = df[df.qtime > 0]
    #df = df[df.max_submitted > 0]
    df.plot(kind='scatter', x=x, y=y, c=c, s=2, cmap='brg')
    #plt.xscale('log')
    plt.yscale('log')
    plt.title('%s --> %s: Set %d'%(src,dst, i))
    plt.text(2,df.qtime.max(), "Transfers with qtime < 1000 sec: %0.2f%%"%(len(df[df.qtime < 1000])*100/ len(df)), bbox=dict(facecolor='red', alpha=0.5))
