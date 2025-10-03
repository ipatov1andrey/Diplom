import matplotlib.pyplot as plt

islands = [40, 60, 80, 100]
original_50 = [0.0349, 0.0939, 0.2392, 0.5884]
flow_50 = [0.0399, 0.1234, 0.4542, 1.4327]
modified_50 = [0.0407, 0.0987, 0.2558, 0.6753]

plt.figure(figsize=(8,5))
plt.plot(islands, original_50, marker='o', label='Original')
plt.plot(islands, flow_50, marker='s', label='Flow')
plt.plot(islands, modified_50, marker='^', label='Modified')
plt.yscale('log')
plt.xlabel('Число островов')
plt.ylabel('Среднее время решения (сек)')
plt.title('Время решения vs число островов (16x16, 50% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('islands_vs_time_50.png', dpi=300)
plt.show() 