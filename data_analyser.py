import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('annotated_data.csv')

corr_matrix = df.corr()

print(corr_matrix)

# Визуализация корреляционной матрицы
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Dataframe feature correlation matrix')
plt.show()