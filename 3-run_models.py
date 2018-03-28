%load verysimpledata.py
%load verysimplemodel5.py
%load verysimplemodel9.py
plt.plot(real_queued, '-ok',label='real queued', alpha=0.5)
plt.plot(model5_queued,label='model C (original) queued')
plt.plot(model9_queued,label='model C (improved) queued')
#plt.plot(real_active, label='real active')
#plt.plot(model5_active, label='model 5 active')
#plt.plot(model5_active, label='model 5 active')
plt.legend()
plt.xlabel('seconds since first submition')
plt.ylabel('number of transfers')
plt.title('Queue Behaviour Comparison')
