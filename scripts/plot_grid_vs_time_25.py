import matplotlib.pyplot as plt

sizes = ['7x7', '10x10', '15x15', '16x16', '24x24']
original_25 = [0.0097, 0.0201, 0.0673, 0.4065, 3.4356]
flow_25 = [0.0115, 0.0225, 0.1004, 0.9184, 9.0847]
modified_25 = [0.0083, 0.0162, 0.0654, 0.3932, 3.5442]

plt.figure(figsize=(8,5))
plt.plot(sizes, original_25, marker='o', label='Original')
plt.plot(sizes, flow_25, marker='s', label='Flow')
plt.plot(sizes, modified_25, marker='^', label='Modified')
plt.yscale('log')
plt.xlabel('Размер сетки')
plt.ylabel('Среднее время решения (сек)')
plt.title('Время решения vs размер сетки (25% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('grid_vs_time_25.png', dpi=300)
plt.show() 