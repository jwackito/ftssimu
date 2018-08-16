from functions import read_xfers

def get_context(t, xfers):
    cut = xfers[t.ended > xfers.submitted]
    cut = cut[t.submitted < cut.ended]
    cut.created = (cut.created - t.created).astype(int)//10**9
    cut.submitted = (cut.submitted - t.created).astype(int)//10**9
    cut.started = (cut.started - t.created).astype(int)//10**9
    cut.ended = (cut.ended - t.created).astype(int)//10**9
    return cut

def get_other_context(t, xfers):
    cut = xfers[xfers.submitted <= t.ended]
    cut = cut[cut.ended >= t.submitted]
    cut.created = (cut.created - t.created).astype(int)//10**9
    cut.submitted = (cut.submitted - t.created).astype(int)//10**9
    cut.started = (cut.started - t.created).astype(int)//10**9
    cut.ended = (cut.ended - t.created).astype(int)//10**9
    return cut

def get_yet_another_context(t, xfers):
    cut = xfers[xfers.created < t.submitted]
    cut = cut[cut.ended > t.submitted]
    cut.submitted = cut.submitted.map(lambda x: min(x, t.submitted))
    cut.started = cut.started.map(lambda x: min(x, t.submitted))
    cut.ended = cut.ended.map(lambda x: min(x, t.submitted))
    cut.created = (cut.created - t.created).astype(int)//10**9
    cut.submitted = (cut.submitted - t.created).astype(int)//10**9
    cut.started = (cut.started - t.created).astype(int)//10**9
    cut.ended = (cut.ended - t.created).astype(int)//10**9
    #cut.submitted = cut.submitted.map(lambda x: min(x, t.RTIME))
    #cut.started = cut.started.map(lambda x: min(x, t.RTIME))
    #cut.ended = cut.ended.map(lambda x: min(x, t.RTIME))
    return cut

def get_no_context(t, xfers):
    cut1 = xfers[xfers.submitted > t.ended]
    cut2 = xfers[xfers.ended < t.submitted]
    cut = cut1.append(cut2)
    cut.created = (cut.created - t.created).astype(int)//10**9
    cut.submitted = (cut.submitted - t.created).astype(int)//10**9
    cut.started = (cut.started - t.created).astype(int)//10**9
    cut.ended = (cut.ended - t.created).astype(int)//10**9
    return cut

def get_tasq_context(t, xfers):                
    cut = xfers[xfers.submitted < t.submitted]
    cut = cut[cut.ended > t.submitted]
    cut['tasq'] = (cut.submitted - t.submitted).astype(int)//10**9
    cut.created = (cut.created - t.created).astype(int)//10**9
    cut.submitted = (cut.submitted - t.created).astype(int)//10**9
    cut.started = (cut.started - t.created).astype(int)//10**9
    cut.ended = (cut.ended - t.created).astype(int)//10**9
    return cut

#xfers,_ =read_xfers('data/transfers_20180616.csv','','')
target = xfers.loc[10]
context = get_tasq_context(target,xfers)
print('len(context):', len(context))
times = []                                                  
for c in context.itertuples():
    created = c.created
    submitted = c.submitted
    started = c.started
    ended = c.ended
    tasq = c.tasq
    times.append([created,submitted, started, ended, tasq])
# colors
rucio_time = 'SeaGreen'
queue_time = 'FireBrick'
net_time = 'Turquoise'
purpleish = 'Indigo'
blakish = 'black'
ms = 10
for time in times:
    if time[0] == 0:
        # skip transfers created at the same time as the target
        continue
    plt.plot(time[0],time[0],'.', ms=ms, color=rucio_time)
    plt.plot(np.arange(time[0], time[1]),[time[0]]*len(np.arange(time[0], time[1])), '-', color=rucio_time)
    plt.plot(time[1],time[0],'.', ms=ms, color=queue_time)
    plt.plot(np.arange(time[1], time[2]),[time[0]]*len(np.arange(time[1], time[2])), '-', color=queue_time)
    plt.plot(time[2],time[0],'.', ms=ms, color=net_time)
    plt.plot(np.arange(time[2], time[3]),[time[0]]*len(np.arange(time[2], time[3])), '-', color=net_time)
    plt.plot(time[3],time[0],'.', ms=ms, color=blakish)
    plt.plot(np.arange(time[1], time[1]+abs(time[4])),[time[0]]*len(np.arange(time[1], time[1]+abs(time[4]))), '--', color=blakish, alpha=0.5)
# plot target
time = []
time.append((target.created - target.created).total_seconds())
time.append((target.submitted - target.created).total_seconds())
time.append((target.started - target.created).total_seconds())
time.append((target.ended - target.created).total_seconds())
plt.plot(time[0],time[0],'.-', ms=ms, color=rucio_time, label='Rucio time')
plt.plot(time[1],time[0],'.-', ms=ms, color=queue_time, label='FTS queue time')
plt.plot(time[2],time[0],'.-', ms=ms, color=net_time, label='Network time')
plt.plot(time[3],time[0],'.', ms=ms, color=blakish, label='End time')
plt.plot(np.arange(time[0], time[1]),[time[0]]*len(np.arange(time[0], time[1])), '-', color=purpleish, label='Target Transfer')
plt.plot(np.arange(time[1], time[2]),[time[0]]*len(np.arange(time[1], time[2])), '-', color=purpleish)
plt.plot(np.arange(time[2], time[3]),[time[0]]*len(np.arange(time[2], time[3])), '-', color=purpleish)

plt.legend()
plt.title('Target transfer and it\'s context')
plt.xlabel('Time in secods (offset-ed to Target creation time)')
plt.ylabel('Transfer created at time in seconds (offset-ed to Target creation time)')

draw_limits = True
if draw_limits == True:
    plt.plot([time[1]]*200, np.arange(-100,100), '--', color='DarkGray')
    plt.plot([time[3]]*200, np.arange(-100,100), '--', color='DarkGray')
