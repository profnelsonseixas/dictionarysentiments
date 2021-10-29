import pandas as pd

csv1 = pd.read_csv('minute232-currently.csv')
csv2 = pd.read_csv('minute97-199.csv')
csv3 = pd.read_csv('minute82-96.csv')
csv4 = pd.read_csv('minute33-81.csv')
csv5 = pd.read_csv('minute21-32.csv')

minutes_concat = pd.concat([csv1, csv2, csv3, csv4, csv5], axis=0)
df = pd.DataFrame(minutes_concat)
df.columns = ['Minutes']
df.to_csv('minutes_concat.csv', index=False)

print(pd.read_csv('minutes_concat.csv'))
