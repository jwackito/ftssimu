minlen = min(len(real_queued), len(model9_queued))
ylimits = (-225,115)
plt.subplot(231)
plt.hist((real_queued[:minlen] - model9_queued)/real_queued[:minlen], label='Model A', bins=50)
plt.grid()
plt.legend()
plt.xlim(ylimits)
plt.yscale('log')
plt.xlabel('Number of queued ((real - pred)')
plt.ylabel('Freq')

plt.subplot(232)
plt.hist((real_queued[:minlen] - model10_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model10_queued)/real_queued[:minlen], label='Model B', bins=50)
plt.grid()
plt.legend()
plt.xlim(ylimits)
plt.yscale('log')
plt.xlabel('Number of queued ((real - pred)')
plt.ylabel('Freq')

plt.subplot(233)
plt.hist((real_queued[:minlen] - model11_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model11_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model11_queued)/real_queued[:minlen], label='Model C', bins=50)
plt.grid()
plt.legend()
plt.xlim(ylimits)
plt.yscale('log')
plt.xlabel('Number of queued ((real - pred)')
plt.ylabel('Freq')

plt.subplot(223)
plt.hist((real_queued[:minlen] - model12_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model12_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model12_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model12_queued)/real_queued[:minlen], label='Model D', bins=50)
plt.grid()
plt.legend()
plt.xlim(ylimits)
plt.yscale('log')
plt.xlabel('Number of queued ((real - pred)')
plt.ylabel('Freq')

plt.subplot(224)
plt.hist((real_queued[:minlen] - model13_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model13_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model13_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model13_queued)/real_queued[:minlen], bins=50)
plt.hist((real_queued[:minlen] - model13_queued)/real_queued[:minlen], label='Model E', bins=50)
plt.grid()
plt.legend()
plt.xlim(ylimits)
plt.yscale('log')
plt.xlabel('Number of queued ((real - pred)')
plt.ylabel('Freq')
