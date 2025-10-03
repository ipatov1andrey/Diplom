import matplotlib.pyplot as plt

islands = [40, 60, 80, 100]

# Original
original_25 = [0.0337, 0.1084, 0.4065, 1.5542]
original_50 = [0.0349, 0.0939, 0.2392, 0.5884]
original_75 = [0.0304, 0.0691, 0.1529, 0.3007]

# Flow
flow_25 = [0.0450, 0.1081, 0.9184, 2.0741]
flow_50 = [0.0399, 0.1234, 0.4542, 1.4327]
flow_75 = [0.0428, 0.1085, 0.2197, 0.7841]

# Modified
modified_25 = [0.0352, 0.1751, 0.3932, 1.4385]
modified_50 = [0.0407, 0.0987, 0.2558, 0.6753]
modified_75 = [0.0325, 0.0666, 0.1452, 0.3515]

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
    plt.plot(islands, data, style, label=label)

plt.yscale('log')
plt.xlabel('Число островов')
plt.ylabel('Среднее время решения (сек)')
plt.title('Влияние числа островов на время решения (16x16) для разных моделей и плотностей')
plt.legend(ncol=3)
plt.tight_layout()
plt.savefig('islands_vs_time_all.png', dpi=300)
plt.show() 