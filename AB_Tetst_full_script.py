##---------------------Required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest, proportion_confint

##---------------------Load e Extraction
df = pd.read_csv(r'data\raw\AdSmartABdata - AdSmartABdata.csv')

##---------------------First look into the data
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

##---------------------Data type conversion
df['auction_id'] = df['auction_id'].astype(pd.StringDtype())
df['experiment'] = df['experiment'].astype(pd.StringDtype())
df['device_make'] = df['device_make'].astype(pd.StringDtype())
df['browser'] = df['browser'].astype(pd.StringDtype())
df['date'] = pd.to_datetime(df['date'])

print(df.dtypes) #checking the data types after conversion

##---------------------Save the cleaned data
df.to_parquet(r'data\clean\cleaned_data.parquet')

df = pd.read_parquet(r'data\clean\cleaned_data.parquet')
print('\n', df.dtypes) #checking the data types after saving

##---------------------Exploratory Data Analysis
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

ax.bar(x-width, total_rate, width, label= 'Responded', color = "#54ADF6")
ax.bar(x, yes_rate, width, label= 'Yes', color = '#4CAF50')
ax.bar(x+width, no_rate, width, label= 'No', color = "#ED5F55")

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Conversion Rate (%)')
ax.set_xlabel('Experiment Group')
ax.set_title('Conversion Rate by Experiment Group')
ax.legend()
plt.tight_layout()
plt.show()

##---------------------Statistical Tests
#---------Data preparation---------
df['responded'] = df[['yes', 'no']].sum(axis=1)
df['responded'] = df['responded'].apply(lambda x: 1 if x >=  1 else 0)

#group counts
control = df[df['experiment'] == 'control']
exposed = df[df['experiment'] == 'exposed']

#conversions and users per group
conversions = [exposed['responded'].sum(), control['responded'].sum() ]
nobs = [len(exposed), len(control)]

#conversion rates
control_cr = conversions[1]/nobs[1]
exposed_cr = conversions[0]/nobs[0]

print(f'Conversion Rate Control: {control_cr:.2%}')
print(f'Conversion Rate Exposed: {exposed_cr:.2%}')
print(f'Absolute Difference: {exposed_cr - control_cr:.2%}')

#---------Unilateral Testing---------
z_stat, p_value = proportions_ztest(count = conversions, nobs = nobs, alternative = 'larger')
print(f'\nZ-statistic:{z_stat:.4f}')
print(f'P-value: {p_value:.4f}')

if p_value < 0.05:
    print('Result: Sucess to reject the null hypothesis. There is a significant difference in conversions.')
else:
    print('Result: Fail to reject the null hypothesis. There is no significant difference in conversions.')

#---------Statistical Power---------
effect_size_responded = proportion_effectsize(exposed_cr, control_cr)
analysis = NormalIndPower()
power = analysis.power(effect_size = effect_size_responded, nobs1=nobs[0], alpha = 0.05, ratio = nobs[1]/nobs[0], alternative= 'larger')

print(f'\nCohen\'s h (effect size): {effect_size_responded:.4f}')
print(f'Statistical Power: {power:.4f}')

if power >= 0.80:
    print('The statistical power is sufficient (greater than 80%)')
else:
    print('The statistical power is insufficient (less than 80%)')

#---------Conversion Rate Visualization---------
lower, upper = proportion_confint(conversions, nobs, alpha=0.05, method='wilson')

groups = ['Exposed', 'Control']
conversion_rates = [exposed_cr, control_cr]
error_lower = conversion_rates - lower
error_upper = upper - conversion_rates
error = [error_lower, error_upper]

plt.figure(figsize=(8,5))
plt.bar(groups, conversion_rates, yerr=error, capsize =5, color = ['#4CAF50', '#2196F3'])
plt.ylabel('Conversion Rate')
plt.title('Conversion Rates by Group (95% CI)')
plt.ylim(0, max(upper) + 0.05)

for i, rate in enumerate(conversion_rates):
    plt.text(i, rate +0.015,
