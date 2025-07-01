##-----------Required modules
import pandas as pd

##-----------Load e Extraction
df = pd.read_csv(r'data\raw\AdSmartABdata - AdSmartABdata.csv')

##-----------First look into the data
print(df.head(10))
print(df.describe())
print(df.info())
print(df.nunique())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.dtypes)

print(df['auction_id'].apply(type).value_counts())
print(df['experiment'].apply(type).value_counts())
print(df['device_make'].apply(type).value_counts())
print(df['browser'].apply(type).value_counts())

###-----------Data type conversion
df['auction_id'] = df['auction_id'].astype(pd.StringDtype())
df['experiment'] = df['experiment'].astype(pd.StringDtype())
df['device_make'] = df['device_make'].astype(pd.StringDtype())
df['browser'] = df['browser'].astype(pd.StringDtype())
df['date'] = pd.to_datetime(df['date'])

print(df.dtypes) #checking the data types after conversion

##-----------Save the cleaned data
df.to_csv('cleaned_data.csv', index=False)
