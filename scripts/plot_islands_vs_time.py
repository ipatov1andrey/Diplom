import matplotlib.pyplot as plt

islands = [40, 60, 80, 100]
original_25 = [0.0337, 0.1084, 0.4065, 1.5542]
flow_25 = [0.0450, 0.1081, 0.9184, 2.0741]
modified_25 = [0.0352, 0.1751, 0.3932, 1.4385]

plt.figure(figsize=(8,5))
plt.plot(islands, original_25, marker='o', label='Original')
plt.plot(islands, flow_25, marker='o', label='Flow')
plt.plot(islands, modified_25, marker='o', label='Modified')
plt.yscale('log')
plt.xlabel('Число островов')
plt.ylabel('Среднее время решения (сек)')
plt.title('Влияние числа островов на время решения (16x16, 25% двойных мостов)')
plt.legend()
plt.tight_layout()
plt.savefig('islands_vs_time.png', dpi=300)
plt.show() 