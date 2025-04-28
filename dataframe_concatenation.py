import pandas as pd

df1 = pd.read_csv('annotated_data.csv')
df2 = pd.read_csv('annotated_data_28_04_25.csv')

df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)

print(df_combined)

df_combined.to_csv('annotated_data_CONC.csv', index=False)