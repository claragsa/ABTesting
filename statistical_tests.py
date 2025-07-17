import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

df = pd.read_parquet(r'data\clean\cleaned_data.parquet')

#---------Samples sizes---------
print(df.groupby('experiment').size())

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


#---------Unilateral Testing---------
z_stat, p_value = proportions_ztest(count = conversions, nobs = nobs, alternative = 'larger')
print(f'\nZ-statistic:{z_stat:.4f}')
print(f'P-value: {p_value:.4f}')

if p_value < 0.05:
    print('Result: Sucess to reject the null hypothesis. There is a significant difference in conversions.')
else:
    print('Result: Fail to reject the null hypothesis. There is no significant difference in conversions.')
