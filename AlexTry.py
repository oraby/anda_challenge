plt.figure(figsize=(10, 10))
for d in all_data:
    h = plt.hist(all_data[d]['isi'], histtype='step', density=1, cumulative=False, bins=10000, label=d)
plt.semilogy()
plt.xlim(0, 0.05)
plt.legend(loc='lower right')



plt.figure(figsize=(10, 10))
for d in all_data:
    h = plt.hist(all_data[d]['isi'], histtype='step', density=1, cumulative=True, bins=100000, label=d)
plt.semilogy()
plt.xlim(0, 0.05)
plt.legend(loc='lower right')
