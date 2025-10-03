import matplotlib.pyplot as plt

sizes = ['7x7', '10x10', '15x15', '16x16', '24x24']
original_75 = [0.0091, 0.0173, 0.0509, 0.1529, 0.8202]
flow_75 = [0.0107, 0.0208, 0.0568, 0.2197, 3.6611]
modified_75 = [0.0076, 0.0159, 0.0477, 0.1452, 0.9445]

plt.figure(figsize=(8,5))
plt.plot(sizes, original_75, marker='o', label='Original')
plt.plot(sizes, flow_75, marker='s', label='Flow')
plt.plot(sizes, modified_75, marker='^', label='Modified')
plt.yscale('log')
plt.xlabel('Размер сетки')
plt.ylabel('Среднее время решения (сек)')
plt.title('Время решения vs размер сетки (75% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('grid_vs_time_75.png', dpi=300)
plt.show() 