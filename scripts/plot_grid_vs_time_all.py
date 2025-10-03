import matplotlib.pyplot as plt

sizes = ['7x7', '10x10', '15x15', '16x16', '24x24']

# Original
original_25 = [0.0097, 0.0201, 0.0673, 0.4065, 3.4356]
original_50 = [0.0098, 0.0179, 0.0597, 0.2392, 1.4967]
original_75 = [0.0091, 0.0173, 0.0509, 0.1529, 0.8202]

# Flow
flow_25 = [0.0115, 0.0225, 0.1004, 0.9184, 9.0847]
flow_50 = [0.0112, 0.0202, 0.0717, 0.4542, 4.7305]
flow_75 = [0.0107, 0.0208, 0.0568, 0.2197, 3.6611]

# Modified
modified_25 = [0.0083, 0.0162, 0.0654, 0.3932, 3.5442]
modified_50 = [0.0082, 0.0152, 0.0482, 0.2558, 1.5458]
modified_75 = [0.0076, 0.0159, 0.0477, 0.1452, 0.9445]

plt.figure(figsize=(10,6))
for data, label, style in [
    (original_25, 'Original 25%', 'o-'),
    (original_50, 'Original 50%', 'o--'),
    (original_75, 'Original 75%', 'o:'),
    (flow_25, 'Flow 25%', 's-'),
    (flow_50, 'Flow 50%', 's--'),
    (flow_75, 'Flow 75%', 's:'),
    (modified_25, 'Modified 25%', '^-'),
    (modified_50, 'Modified 50%', '^--'),
    (modified_75, 'Modified 75%', '^:'),
]:
    plt.plot(sizes, data, style, label=label)

plt.yscale('log')
plt.xlabel('Размер сетки')
plt.ylabel('Среднее время решения (сек)')
plt.title('Зависимость времени решения от размера сетки для разных моделей и плотностей')
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig('grid_vs_time_all.png', dpi=300)
plt.show() 