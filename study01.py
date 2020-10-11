# Percentage of rules with X transfers vs Percentage contribution to the total bytes transferred
#from functions import read_xfers

#xfers = read_xfers('data/transfers-FTSBNL-20190606-20190731.csv', nrows=10000000)

rules = pd.read_csv('data/xfers_per_rule.csv')
rules.nxfers = rules.nxfers.astype(int)
rules.bytes = rules.bytes.astype(float)

n = []
for i in range(1001):
    n.append(len(rules.nxfers[rules.nxfers == i])/len(rules))

n2 = []
for i in range(1001):
    n2.append(rules.bytes[rules.nxfers == i].sum()/rules.bytes.sum())

plt.plot(np.array(n)*100, label='Percentage of rules with X transfers')
plt.plot(np.array(n2)*100,'-*', label='Percentage of contribution to the total bytes transferred')
plt.legend()
plt.xlabel('Number of transfers in the rule')
plt.ylabel('%% respect to total transfers')
plt.title('Contribution of the rules to trafic')

print('%22s %5s %5s %5s %5s'%('activity', 'tot', 'size',' qt', 'nt'))
for act in list(set(xfers.activity)):                                                                                                                                                                    
    print('%22s %0.3f %0.3f %0.3f %0.3f'%(act, len(xfers[(xfers.account == 'panda')&(xfers.activity == act)])/len(xfers),xfers[(xfers.account == 'panda')&(xfers.activity == act)].SIZE.sum()/xfers.SIZE.sum(), xfers[(xfers.account == 'panda')&(xfers.activity == act)].QTIME.sum()/xfers.QTIME.sum(), xfers[(xfers.account == 'panda')&(xfers.activity == act)].NTIME.sum()/xfers.NTIME.sum()))
    print('%22s %0.3f %0.3f %0.3f %0.3f'%('total', len(xfers[(xfers.account == 'panda')])/len(xfers),xfers[(xfers.account == 'panda')].SIZE.sum()/xfers.SIZE.sum(), xfers[(xfers.account == 'panda')].QTIME.sum()/xfers.QTIME.sum(), xfers[(xfers.account == 'panda')].NTIME.sum()/xfers.NTIME.sum()))
