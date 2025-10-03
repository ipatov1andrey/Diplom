import matplotlib.pyplot as plt

islands = [40, 60, 80, 100]
original_75 = [0.0304, 0.0691, 0.1529, 0.3007]
flow_75 = [0.0428, 0.1085, 0.2197, 0.7841]
modified_75 = [0.0325, 0.0666, 0.1452, 0.3515]

plt.figure(figsize=(8,5))
plt.plot(islands, original_75, marker='o', label='Original')
plt.plot(islands, flow_75, marker='s', label='Flow')
plt.plot(islands, modified_75, marker='^', label='Modified')
plt.yscale('log')
plt.xlabel('Число островов')
plt.ylabel('Среднее время решения (сек)')
plt.title('Время решения vs число островов (16x16, 75% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('islands_vs_time_75.png', dpi=300)
plt.show() 