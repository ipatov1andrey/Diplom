import pandas as pd
import matplotlib.pyplot as plt

file_path = 'examples/generated/15x15_50_50_stats.txt'
df = pd.read_csv(file_path, sep='|', header=None, engine='python')
df.columns = ['filename', 'islands', 'single_bridges', 'double_bridges', 'total_bridges', 'double_percent', 'attempts']
df['double_percent'] = df['double_percent'].str.replace('%', '').astype(float)

plt.figure(figsize=(5, 6))
plt.boxplot([df['double_percent']], vert=True, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))
plt.ylabel('Процент двойных мостов')
plt.title('Разброс процента двойных мостов (15x15, 50 островов, 50%)')
plt.xticks([1], ['50%'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('double_bridge_percent_boxplot_50.png', dpi=300)
plt.show()