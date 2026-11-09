import pandas as pd
df_lang = pd.read_parquet("phase1_language_flags.parquet")
print(df_lang.head())
print(df_lang['lang'].value_counts())