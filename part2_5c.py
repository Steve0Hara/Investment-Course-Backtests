import pandas as pd
import numpy as np
from plots import *
from utility_functions import *

industry_dict = {
    11: "Hlth",
    2: "Food",
    26: "Guns",
    30: "Oil",
    12: "MedEq",
    46: "RlEst",
    35: "Comps",
    41: "Whlsl",
    47: "Fin",
    7: "Fun",
}



df_industry_returns = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Return')
df_factors = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Factors')

df_industry_returns['YearMonth'] = df_industry_returns['YearMonth'].astype(str)
df_factors['YearMonth'] = df_factors['YearMonth'].astype(str)
df_combined_orig = df_industry_returns.merge(df_factors, on='YearMonth', how='left')


# ----------- Define Data ----------------
df_holdout_sample5 = df_combined_orig[(df_combined_orig['YearMonth'] >= '201701') & (df_combined_orig['YearMonth'] <= '202312')].copy() # 2017 because its needed to calculate the first 12 month trailing stats

df_holdout_sample5['YearMonth'] = pd.to_datetime(df_holdout_sample5['YearMonth'], format='%Y%m')

industry_columns = [col for col in df_holdout_sample5.columns if col not in ['YearMonth', 'Year', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']]
df_holdout_sample5[industry_columns] = df_holdout_sample5[industry_columns] / 100
df_holdout_sample5['RF'] = df_holdout_sample5['RF'] / 100

# --------------- Calculate Monthly Excess Returns -------------
df_excess_returns = df_holdout_sample5[industry_columns].sub(df_holdout_sample5['RF'], axis=0)
df_excess_returns['YearMonth'] = df_holdout_sample5['YearMonth']

# --------------- Calculate Trailing 12-Month Cumulative Excess Return -------------
def trailing_cumulative_excess_return(x):
    return np.prod(1 + x) - 1

# trailing cum excess returns
df_trailing_cum_excess_returns = df_excess_returns[industry_columns].shift(1).rolling(window=12).apply(trailing_cumulative_excess_return, raw=True)
df_trailing_cum_excess_returns['YearMonth'] = df_excess_returns['YearMonth']

# -------------- Calculate Trailing 12-Month Standard Deviation ------------------
df_trailing_std = df_excess_returns[industry_columns].shift(1).rolling(window=12).std()
df_trailing_std['YearMonth'] = df_excess_returns['YearMonth']


# --------------- Merging Results -------------
df_trailing_analysis = df_trailing_cum_excess_returns.merge(df_trailing_std, on='YearMonth', suffixes=('_cum_return', '_std'))

# filtering out 2017
df_trailing_analysis = df_trailing_analysis[df_trailing_analysis['YearMonth'] >= '2018-01-01']

df_trailing_analysis.to_csv('ex5c_Trailing_Data.csv', index=False)

# print("Trailing 12-Month Cumulative Excess Returns and Standard Deviations by Month:")
# print(df_trailing_analysis)

selected_industries = list(industry_dict.values())[:10]
df_excess_returns['YearMonth'] = df_holdout_sample5['YearMonth']

df_momentum = df_excess_returns.merge(df_trailing_analysis, on='YearMonth', suffixes=('', '_trailing'))

scaling_factor = 0.40  # 40% scaling factor

# function for calculating TSMOM returns
def calculate_tsmom_returns(row, industry):
  
    trailing_cum_ret = row[f"{industry}_cum_return"]

    trailing_std = row[f"{industry}_std"]
    
    monthly_excess_ret = row[industry]
    
    # weight in the formula
    weight = (scaling_factor / trailing_std) 
    
    sign_factor = np.sign(trailing_cum_ret)  # +1 if positive, -1 if negative
    
    # tsmom return formula
    tsmom_return = sign_factor * weight * monthly_excess_ret

    return tsmom_return

# applying the tsmom formula to each industry
for industry in selected_industries:
    df_momentum[f"{industry}_tsmom"] = df_momentum.apply(
        lambda row: calculate_tsmom_returns(row, industry), axis=1
    )

# calculating the mean of all tsmom returns for the tsmom portfolio return
industry_tsmom_columns = [f"{industry}_tsmom" for industry in selected_industries]
print(industry_tsmom_columns)
df_momentum['TSMOM_Portfolio_Return'] = df_momentum[industry_tsmom_columns].mean(axis=1)

df_tsmom_result = df_momentum[['YearMonth', 'TSMOM_Portfolio_Return'] + industry_tsmom_columns]
print(df_tsmom_result)

df_tsmom_result.to_csv('ex5c_TSMOM_Portfolio_Data.csv', index=False)

# ---------- CALCULATE STATS ------------

# average excess return (annualized)
average_monthly_return = df_tsmom_result['TSMOM_Portfolio_Return'].mean()
average_annual_return = (average_monthly_return)*12

# standard deviation (annualized)
monthly_std_dev = df_tsmom_result['TSMOM_Portfolio_Return'].std()
annualized_std_dev = monthly_std_dev * (12**0.5)

# sharpe ratio 
sharpe_ratio = average_annual_return / annualized_std_dev # riskfree rate is not subtracted again because the tsmom returns are already excess returns

print(f'Avg monthly excess return: {average_monthly_return}, Avg annual excess return: {average_annual_return}, Avg annual std: {annualized_std_dev}, Annual sharpe ratio: {sharpe_ratio}')


