import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

df= pd.read_parquet(r'data\clean\cleaned_data.parquet')

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
    plt.text(i, rate +0.015, f'{rate:.2%}', ha='center',fontsize = 10)
plt.show()