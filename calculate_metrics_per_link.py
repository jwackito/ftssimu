import numpy as np
import pandas as pd
import datetime
from read import readxfers


def calculate_actives(xfers, absolute_offset=None):
    if absolute_offset is None:
        absolute_offset = xfers.created.min()
    actives = np.zeros(int((xfers.ended.max() - absolute_offset).total_seconds()))
    offsets = (xfers.started - absolute_offset).values/10**9
    for time,offset in zip((xfers.ended - xfers.started).values/10**9, offsets):
        act = np.array([1]*int(time), dtype=int)
        actives[int(offset):len(act)+int(offset)] += act
    return np.array(actives, dtype=int)


def calculate_queued(xfers, absolute_offset=None):
    if absolute_offset is None:
        absolute_offset = xfers.created.min()
    queued = np.zeros(int((xfers.ended.max() - absolute_offset).total_seconds()))
    queued_bytes = np.zeros(int((xfers.ended.max() - absolute_offset).total_seconds()))
    offsets = (xfers.submitted - absolute_offset).values/10**9
    for time,offset,size in zip((xfers.started - xfers.submitted).values/10**9, offsets, xfers.SIZE):
        q = np.array([1]*int(time), dtype=int)
        s = np.array([size]*int(time), dtype=int)
        queued[int(offset):len(q)+int(offset)] += q
        queued_bytes[int(offset):len(s)+int(offset)] += s
    return queued,queued_bytes


src = 'CERN-PROD'
dst = 'BNL-ATLAS'
print ('Analizing link %s --> %s...'%(src, dst))
print ('reading transfers')
#xfers,link = read_xfers('data/transfers_20180616.csv', src, dst)
#xfers['link'] = xfers.src_name + '__'+ xfers.dst_name
link = src + '__' + dst
#times = np.zeros(int((xfers.ended.max() - xfers.created.min()).total_seconds()))
offset = xfers.created.min()

xfers_in_link = xfers[xfers.link == link]
xfers_exiting_src = xfers[xfers.src_name == src]
xfers_arriving_dst = xfers[xfers.dst_name == dst]

print('Calculating actives/queued for link based on %d transfers'%(len(xfers_in_link)))
link_actives = calculate_actives(xfers_in_link, offset)
link_queued,link_queued_bytes = calculate_queued(xfers_in_link, offset)
print('Calculating actives/queued for src based on %d transfers'%(len(xfers_exiting_src) - len(xfers_in_link)))
src_actives = calculate_actives(xfers_exiting_src, offset)
minlen = min(len(link_actives),len(src_actives))
src_actives[:minlen] -= link_actives
src_queued,src_queued_bytes = calculate_queued(xfers_exiting_src, offset)
minlen = min(len(link_actives),len(dst_actives))
src_queued[:minlen] -= link_queued
minlen = min(len(link_actives),len(dst_actives))
src_actives[:minlen] -= link_actives
print('Calculating actives/queued for dst based on %d transfers'%(len(xfers_arriving_dst) - len(xfers_in_link)))
dst_actives = calculate_actives(xfers_arriving_dst, offset)
minlen = min(len(link_actives),len(dst_actives))
dst_actives[:minlen] -= link_actives
dst_queued,dst_queued_bytes = calculate_queued(xfers_arriving_dst, offset)
minlen = min(len(link_actives),len(dst_actives))
dst_queued[:minlen] -= link_queued
minlen = min(len(link_actives),len(dst_actives))
dst_actives[:minlen] -= link_actives

plt.hist(src_actives,bins=50)
plt.hist(dst_actives,bins=50)
plt.hist(link_actives,bins=50)
