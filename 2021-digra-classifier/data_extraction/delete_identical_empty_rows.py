import pandas as pd
import re

df=pd.read_csv("kotor1_nonpersuade_sents.csv",index_col=[0])
print(len(df))
df = df.dropna()
df = df.drop_duplicates(subset=['en', 'de', 'es', 'fr', 'it'], keep=False)
print(len(df))
df.to_csv("nwn_nonpersuade_sents_filtered.csv",mode='wb',encoding='utf-8')