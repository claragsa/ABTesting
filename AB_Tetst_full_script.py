##-----------Required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
df.to_parquet(r'data\clean\cleaned_data.parquet')

df = pd.read_parquet(r'data\clean\cleaned_data.parquet')
print('\n', df.dtypes) #checking the data types after saving

##-----------Exploratory Data Analysis
#Number of users in each experiment group
print('\nNumber of users in each experiment group:')
print(df.groupby('experiment').size())

#Percentage of users in each experiment group
percent_control = df[df['experiment'] == 'control'].shape[0]/df['experiment'].shape[0] * 100
percent_exposed = df[df['experiment']== 'exposed'].shape[0]/df['experiment'].shape[0]*100
print(f'Percentage of control group: {percent_control:.2f}%')
print(f'Percentage of exposed group: {percent_exposed:.2f}%')

#Conversion rate by experiment group
df['responded'] = df[['yes', 'no']].sum(axis=1)
df['responded'] = df['responded'].apply(lambda x: 1 if x >=  1 else 0)
responded_conversion = df.groupby('experiment')['responded'].mean()*100
print('Conversion rate by experiment group (users selected yes or no):')
print(responded_conversion)

df['not_responded'] = df['responded'].apply(lambda x: 1 if x == 0 else 0)
not_responded_conversion = df.groupby('experiment')['not_responded'].mean()*100
print('\nUsers that did not respond to the survey:')
print(not_responded_conversion)

yes_conversion = df.groupby('experiment')['yes'].mean()*100
print('\nUsers that respondend and selected yes:')
print(yes_conversion)

no_conversion = df.groupby('experiment')['no'].mean()*100
print('\nUsers that responded and selected no:')
print(no_conversion)

#Conversion rate visualization
groups = ['control', 'exposed']

total_rate = [responded_conversion.loc[group] for group in groups]
yes_rate = [yes_conversion.loc[group] for group in groups]
no_rate = [no_conversion.loc[group] for group in groups]

x = np.arange(len(groups))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x-width, total_rate, width, label= 'Responded')
ax.bar(x, yes_rate, width, label= 'Yes')
ax.bar(x+width, no_rate, width, label= 'No')

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('COnversion Rate (%)')
ax.set_xlabel('Experiment Group')
ax.set_title('Conversion Rate by Experiment Group')
ax.legend()
plt.tight_layout()
plt.show()
