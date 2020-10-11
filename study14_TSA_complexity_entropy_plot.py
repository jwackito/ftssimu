from itertools import permutations
from fbm import FBM

def get_symbol(x):
    '''
    Returns the symbol of x, as the indexes of the sorted elements
        [1,2,3] -> [1,2,3] = [0,1,2]
        [3,2,1] -> [1,2,3] = [2,1,0]
        [2,3,1] -> [1,2,3] = [1,2,0]
    '''
    x1 = x.copy()
    x1.sort()
    return tuple([x1.index(i) for i in x ])

def get_symbol2(x):
    t = get_symbol(x)
    if len(t) == len(set(t)):
        return t
    else:
        return ()

def bandt_pompe_pattern_probability_distribution(x, order=4, step='one'):
    #x = list(x + random.random(len(x))*0.00001)
    x = list(x)
    if step == 'one':
        step = 1
    elif step == 'half':
        step = int(floor(order/2))
    elif step == 'full':
        step = order
    else:
        raise ValueError('step must be one of one|half|full.')
    idx = 0
    symbols = []
    totalsteps = 0
    while (idx+order) <= len(x):
        sym = get_symbol2(x[idx:idx+order])
        if sym != ():
            symbols.append(sym)
            totalsteps += 1
        idx += step
    smap = set(symbols)
    if len(smap) > factorial(order):
        print('Warning: more symbols than order factorial')
    if len(smap) < factorial(order):
        print('Warning: will be empty bins')
    counts = []
    for t in smap:
        counts.append(symbols.count(t))
    counts = np.array(counts)
    prob = counts/len(symbols)
    return prob

def S(P):
    '''Shannon entropy'''
    return -sum(P*log(P))

def S_t(P,q):
    '''Tsallis entropy for entropic index q'''
    return (1/(q-1))*sum(P-(P**q))

def Hs(P):
    return S(P)/log(len(P))

def J(P, Pe):
    return S((P + Pe)/2) - S(P)/2 - S(Pe)/2

def Qj(P, Pe):
    M = len(P)
    Q0 = -2 * ( ((M+1)/M) * log(M+1) - 2 * log(2*M) + log(M))**-1
    return J(P,Pe)*Q0

def Cjs(P):
    Pe=np.ones(len(P))/len(P)
    hs = Hs(P)
    return Qj(P,Pe)*hs, hs

def jointplot(x, y):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(3, 3)
    ax_main = plt.subplot(gs[1:3, :2])
    ax_xDist = plt.subplot(gs[0, :2],sharex=ax_main)
    ax_yDist = plt.subplot(gs[1:3, 2],sharey=ax_main)

    ax_main.scatter(x,y,marker='.')
    ax_main.set(xlabel="x data", ylabel="y data")

    ax_xDist.hist(x,bins=100,align='mid')
    ax_xDist.set(ylabel='count')
    ax_xCumDist = ax_xDist.twinx()
    ax_xCumDist.hist(x,bins=100,cumulative=True,histtype='step',normed=True,color='r',align='mid')
    ax_xCumDist.tick_params('y', colors='r')
    ax_xCumDist.set_ylabel('cumulative',color='r')

    ax_yDist.hist(y,bins=100,orientation='horizontal',align='mid')
    ax_yDist.set(xlabel='count')
    ax_yCumDist = ax_yDist.twiny()
    ax_yCumDist.hist(y,bins=100,cumulative=True,histtype='step',normed=True,color='r',align='mid',orientation='horizontal')
    ax_yCumDist.tick_params('x', colors='r')
    ax_yCumDist.set_xlabel('cumulative',color='r')

    plt.show()

rho = 30
rulesttc = pd.read_csv('data/rules_timeseries_real_rho%d.csv'%rho)
rulesttc['created'] = pd.to_datetime(rulesttc.min_created)
mu = 'medianttc'
x = rulesttc[mu].values.astype(float)
bporder = 2
for order in range(4,5):
    #plt.subplots()
    for hurst, fmt in zip([.01, .05, .25, .5, ((.5+.75)/2), .75, .85], ['r.','g.', 'b.','k.', 'm.','y.', 'c.']):
        f = FBM(n=5000, hurst=hurst, length=1, method='daviesharte')
        hss = []
        cjss = []
        for i in range(50):
            cjs,hs = Cjs(bandt_pompe_pattern_probability_distribution(f.fbm(),order))
            hss.append(hs)
            cjss.append(cjs)
        plt.plot(hss, cjss, fmt, alpha=0.25,label='fbm h=%0.2f'%hurst)
    #for hurst, fmt in zip([.05, .25, .5, .62, .65, .75, .85], ['r*','g*', 'b*','k*', 'm*','y*','c*']):
    #    f = FBM(n=len(x)*2, hurst=hurst, length=1, method='daviesharte')
    #    hss = []
    #    cjss = []
    #    for i in range(50):
    #        cjs,hs = Cjs(bandt_pompe_pattern_probability_distribution(f.fgn(),order))
    #        hss.append(hs)
    #        cjss.append(cjs)
    #    plt.plot(hss, cjss, fmt, alpha=0.25,label='fgn h=%0.2f'%hurst)
    hss = []
    cjss = []
    for s, e in zip(range(0,len(x),1000), range(1000,len(x), 1000)):
        cjs,hs = Cjs(bandt_pompe_pattern_probability_distribution(x[s:e],order))
        hss.append(hs)
        cjss.append(cjs)
    plt.plot(hss, cjss,'x', alpha=0.95,label='obs %d'%order)
    hss = []
    cjss = []
    for s, e in zip(range(0,len(x),1000), range(1000,len(x), 1000)):
        cjs,hs = Cjs(bandt_pompe_pattern_probability_distribution(x[s:e],order,'half'))
        hss.append(hs)
        cjss.append(cjs)
    plt.plot(hss, cjss,'x', alpha=0.95,label='obs %d'%order)
    hss = []
    cjss = []
    for s, e in zip(range(0,len(x),1000), range(1000,len(x), 1000)):
        cjs,hs = Cjs(bandt_pompe_pattern_probability_distribution(x[s:e],order,'full'))
        hss.append(hs)
        cjss.append(cjs)
    plt.plot(hss, cjss,'x', alpha=0.95,label='obs %d'%order)
    plt.xlim((0.5,1))
    plt.ylim((0,.5))
    plt.legend()
    plt.title('Complexity-Entropy (BP) Causality Plane for %s of order %d'%(mu, order))
    plt.ylabel('$C_{js}$')
    plt.xlabel('$H_s$')
