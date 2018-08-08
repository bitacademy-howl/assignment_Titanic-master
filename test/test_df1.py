import pandas as pd

df = pd.DataFrame()

df["1"] = [1,2,3,4,5]
df["2"] = ['a', 'b', 'c', 'd', 'e']
df['asd'] = [None,2,3,'a','b']


# df.info()
#
#
# print(pd.isna(df['3']), type(pd.isna(df['3'])))
print(df[pd.isna(df['asd'])])

# print(df[df['as'] == 'a'], type(df[df['as'] == 'a']))

# df['asd'] = df[df['asd'] == 'a']
#
# print(df)