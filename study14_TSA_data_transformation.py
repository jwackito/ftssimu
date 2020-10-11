
# Time Series from Rules Dataset transformation

rules = pd.read_csv('data/xfers_per_rule_2.csv') 
rules.nxfers = rules.nxfers.astype(int) 
rules.bytes = rules.bytes.astype(float) 
for d in ['min_created', 'min_submitted', 
       'min_started', 'min_ended', 'max_created', 'max_submitted', 
       'max_started', 'max_ended']: 
    rules[d] = pd.to_datetime(rules[d]) 
rules['ttc'] = (rules.max_ended - rules.min_created).astype(int)/10**9 
rules['createdatonce'] = (rules.min_created == rules.max_created).values.astype(int)

rules = rules[rules.createdatonce == 1]

rho = 30

x = pd.date_range(rules.min_created.min(), rules.min_created.max(), freq='%d'%rho+'s')
totaldates = len(x)

dates = [] 
minttc = [] 
maxttc = [] 
meanttc = [] 
medianttc = [] 
i = 0 
for s, e in zip(x,x[1:]): 
    cut = rules[(rules.min_created > s) & (rules.min_created < e)] 
    dates.append(d) 
    minttc.append(cut.ttc.min()) 
    maxttc.append(cut.ttc.max()) 
    meanttc.append(cut.ttc.mean()) 
    medianttc.append(cut.ttc.median()) 
    i += 1 
    print('%d/%d'%(i,totaldates), end='\r')
df = pd.DataFrame(np.array([x[:-1], minttc, medianttc, meanttc, maxttc]).T, columns=['min_created', 'minttc','medianttc','meanttc','maxttc'])
df = df.fillna(method='ffill')
df.to_csv('data/rules_timeseries_real_rho%d.csv'%rho,index=False)
df[(df.minttc >= 0) == False]


