import matplotlib.pyplot as plt

sizes = ['7x7', '10x10', '15x15', '16x16', '24x24']
original_50 = [0.0098, 0.0179, 0.0597, 0.2392, 1.4967]
flow_50 = [0.0112, 0.0202, 0.0717, 0.4542, 4.7305]
modified_50 = [0.0082, 0.0152, 0.0482, 0.2558, 1.5458]

plt.figure(figsize=(8,5))
plt.plot(sizes, original_50, marker='o', label='Original')
plt.plot(sizes, flow_50, marker='s', label='Flow')
plt.plot(sizes, modified_50, marker='^', label='Modified')
plt.yscale('log')
plt.xlabel('Размер сетки')
plt.ylabel('Среднее время решения (сек)')
plt.title('Время решения vs размер сетки (50% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('grid_vs_time_50.png', dpi=300)
plt.show() 