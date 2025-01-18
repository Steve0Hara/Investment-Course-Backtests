import pandas as pd
import numpy as np

df_factors = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Factors')
df_factors['YearMonth'] = pd.to_datetime(df_factors['YearMonth'], format='%Y%m')

# holdout period
df_factors_holdout = df_factors[(df_factors['YearMonth'] >= pd.to_datetime('201801', format='%Y%m')) & (df_factors['YearMonth'] <= pd.to_datetime('202312', format='%Y%m'))].copy()
df_factors_holdout['Mkt-RF'] = df_factors_holdout['Mkt-RF'] / 100
df_factors_holdout['RF'] = df_factors_holdout['RF'] / 100

# total market return
df_factors_holdout['Market_Total_Return'] = df_factors_holdout['Mkt-RF'] + df_factors_holdout['RF']

# excess return
df_factors_holdout['Excess_Return'] = df_factors_holdout['Mkt-RF']

monthly_mean_excess_return = df_factors_holdout['Excess_Return'].mean()  
average_annual_excess_return = monthly_mean_excess_return * 12  # annualized

# std
monthly_std_excess_return = df_factors_holdout['Excess_Return'].std()
annual_std_excess_return = monthly_std_excess_return * np.sqrt(12)  # annualized

sharpe_ratio = average_annual_excess_return / annual_std_excess_return

print(f"Average Annual Excess Return: {average_annual_excess_return}")
print(f"Annualized Standard Deviation of Excess Returns: {annual_std_excess_return}")
print(f"Sharpe Ratio: {sharpe_ratio}")

df_factors_holdout.to_csv('ex5e_Market_Performance_Stats.csv', index=False)
