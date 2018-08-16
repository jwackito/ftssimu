from functions import read_xfers

src = 'BNL-ATLAS'
dst = 'MWT2'
filepath = 'data/BNL_MWT2_20180624.csv'

xfers, _ = read_xfers(filepath, src, dst)

# separate transfers by FTS server
servers = set(xfers.external_host)
dflist = []
for server in servers:
    dflist.append(xfers[xfers.external_host == server])

stats = {}
for df,server in zip(dflist,servers):
    # number of queued per second in df
    print('Processing server: %s'%server)
    queued = np.zeros(int((max(df.ended)-min(df.submitted)).total_seconds()))
    offsets = (df.submitted - df.submitted.min()).values/10**9
    for time,offset in zip((df.started - df.submitted).values/10**9, offsets):
        que = np.array([1]*int(time), dtype=int)
        queued[int(offset):len(que)+int(offset)] += que
    # number of actives per second in df
    actives = np.zeros(int((max(df.ended)-min(df.submitted)).total_seconds()))
    offsets = (df.started - df.submitted.min()).values/10**9
    for time,offset in zip((df.ended - df.started).values/10**9, offsets):
        act = np.array([1]*int(time), dtype=int)
        actives[int(offset):len(act)+int(offset)] += act
    stats[server] = {}
    stats[server]['queued'] = np.array(queued)
    stats[server]['active'] = np.array(actives)
    # measure the reaction time
    q0 = stats[server]['queued'] == 0
    a0 = stats[server]['active'] == 0
    q0a0 = np.logical_and(q0,a0)
    # there are queued but nor actives in
    queued_but_noactive = (q0a0^a0).astype(int)
    
    samples = []
    partial = 0
    for run in queued_but_noactive:
        if run == 0 and partial != 0:
            samples.append(partial)
            partial = 0
        if run == 1:
            partial += 1
    stats[server]['delays'] = np.array(samples)
    print('Count: %d'%len(samples))
    print('Mean delay: %f'%mean(samples))
    print('STD delay: %f'%std(samples))
    print('MIN delay: %f'%min(samples))
    print('MAX delay: %f'%max(samples))
