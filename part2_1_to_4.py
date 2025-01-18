import pandas as pd
import numpy as np
from plots import *
from utility_functions import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # this has nothing to do with the calculations

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
    18: "Cnstr"
}


# ------------------ Importing Data and making it usable -----------------------

# import data from excel
df_industry_returns = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Return')
df_factors = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Factors')

df_industry_returns['YearMonth'] = df_industry_returns['YearMonth'].astype(str)
df_factors['YearMonth'] = df_factors['YearMonth'].astype(str)

# merging the 2 excel sheets into one df
df_combined_orig = df_industry_returns.merge(df_factors, on='YearMonth', how='left')

# in-sample period
df_in_sample = df_combined_orig[(df_combined_orig['YearMonth'] >= '199501') & (df_combined_orig['YearMonth'] <= '201712')].copy()

# out-of-sample period
df_out_of_sample = df_combined_orig[(df_combined_orig['YearMonth'] >= '201801') & (df_combined_orig['YearMonth'] <= '202312')].copy()

df_in_sample.loc[:, 'YearMonth'] = pd.to_datetime(df_in_sample['YearMonth'], format='%Y%m')
df_out_of_sample.loc[:, 'YearMonth'] = pd.to_datetime(df_out_of_sample['YearMonth'], format='%Y%m')



""" 1) a) Plot the cumulative return of your principal industry and the market portfolio. """ 

# cum return for health industry
df_in_sample['Hlth_Cumulative_Return'] = (1 + df_in_sample['Hlth'] / 100).cumprod() # don't forget to put Hlth back!!!!!!!!!!!!
df_in_sample['Hlth_Cumulative_Return'] = df_in_sample['Hlth_Cumulative_Return'] -1

# total market return as sum of excess mkt-rf and rf
df_in_sample['Market_Total_Return'] = df_in_sample['Mkt-RF'] / 100 + df_in_sample['RF'] / 100

# cum return for total market
df_in_sample['Market_Cumulative_Return'] = (1 + df_in_sample['Market_Total_Return']).cumprod()
df_in_sample['Market_Cumulative_Return'] = df_in_sample['Market_Cumulative_Return'] - 1

df_in_sample.to_csv("ex_1a.csv") 

plot_cumulative_returns(df_in_sample, title="Cumulative Return of the Principal Industry (Health) and the Market Portfolio")



""" 1) c) Calculate the CAPM beta for your ten industries and discuss their systematic risk, 
    while considering the fundamental characteristics of the industries. """

print('\n\n----------- Exercise 1 c) -------------')

betas = []

# defining a function for calculating capm betas
def calculate_capm_beta(df, asset_returns_col, riskfree_rate='RF'):

    # asset excess return
    df_in_sample['Excess_Asset_Returns'] = df[asset_returns_col] - df[riskfree_rate]

    # cov of asset excess return and market excess return
    covariance = df_in_sample['Excess_Asset_Returns'].cov(df_in_sample['Mkt-RF'])

    # market excess return variance
    market_variance = df_in_sample['Mkt-RF'].var()
    # print(f'covariance {covariance}')
    # print(f'market variance {market_variance}')

    # beta formula
    beta = covariance / market_variance

    return beta

for industry_id, industry_ticker in industry_dict.items():

    beta = calculate_capm_beta(df_in_sample, asset_returns_col=industry_ticker)
    
    print(f"CAPM Beta for {industry_ticker} (Industry ID {industry_id}): {beta}")

    betas.append(beta)

industry_tickers = list(industry_dict.values())
plot_capm_betas(industry_tickers, betas)



""" 2) For this question use data from the in-sample period. Plot the mean-variance efficient 
    frontier of the ten industries portfolios assigned to your group in a chart. Also present the 
    Capital Allocation Line in the same chart. """

print('\n\n----------- Exercise 2 -------------')

# ---------- Get Data ------------

industries = list(industry_dict.values())[:-1] # all industries minus the extra one (Cnstr)
returns_data = df_in_sample[industries] / 100  

# mean monthly industry returns
mean_returns = returns_data.mean()
print('Mean Returns of the industries (Monthly):')
print(mean_returns)

# annualized industry mean returns
mean_returns_annual = mean_returns * 12
print('Annualized Mean Returns of the industries:')
print(mean_returns_annual)

# annualized industry cov matrix
cov_matrix = returns_data.cov() * 12
print('Covariance Matrix of the industries (Annual):')
print(cov_matrix)

# convert to arrays
expected_returns = mean_returns_annual.values 
cov_matrix = cov_matrix.values 

# annualizing riskfree asset returns
rf_monthly = df_in_sample['RF'] / 100  
rf_yearly_mean = rf_monthly.mean() * 12

# -------- Defining Parameters for the Efficient Frontier ---------

# target returns
target_returns = np.linspace(0, 0.25, 100)
target_returns = np.round(target_returns, 8) # Rounding to 8 decimals

# weights sum to 1
constraints = (
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
)

# bounds on weights 
bounds = [(-1, 1) for _ in range(10)]

portfolio_volatilities_2, portfolio_returns_2 = compute_efficient_frontier(expected_returns, cov_matrix, target_returns, constraints, bounds, method='SLSQP')

# --------- Defining Parameters for the Tangency Portfolio ----------

tangency_volatility_2, tangency_return_2, tangency_weights_2, tangency_sharpe_2 = compute_tangency_portfolio(expected_returns, cov_matrix, rf_yearly_mean, constraints, bounds, method='SLSQP')
print(f'2) tangency return {tangency_return_2}, tangency volatility {tangency_volatility_2}, tangency weights {tangency_weights_2}')

# ---------- Getting data for industry dots --------

industry_returns = mean_returns_annual.values
industry_volatilities = np.sqrt(np.diag(cov_matrix))
industry_labels = list(industry_dict.values())[:-1]

# ---------- Plot --------------

plot_efficient_frontier(portfolio_volatilities_2, portfolio_returns_2, tangency_volatility_2, tangency_return_2, rf_yearly_mean, industry_volatilities, industry_returns, industry_labels, 'Exercise_2')



""" 3) Using data from the in-sample period, what is the optimal portfolio for a mean-variance 
    investor that desires an expected return of 8% per year? Do this for each of following three 
    investment universes:  

    a. Ten industries and the risk-free asset (11 total assets), long and short positions 
    allowed. 

    b. Ten industries and the risk-free asset (11 total assets), only long positions 
    allowed. No short-selling. 

    c. Ten industries and the risk-free asset (11 total assets), with long and short 
    positions allowed, but each industry's weight must be constrained to between -
    25% and 25% of the total portfolio. 

    d. Ten industries, the risk-free asset, and the “Extra Industry” (12 assets), long and 
    short positions allowed. 
    
    Then discuss: 
    e. How do the weights on the assets vary across the four different portfolios? Why do 
    they vary, and what do you learn from the differences? """

print('\n\n-------------------- Exercise 3) -------------------')

target_return = 0.08
rf_mean_annual_in_sample = (df_in_sample['RF'] / 100).mean() * 12

# 3) a) long and short positions allowed
assets_3abc = list(industry_dict.values())[:-1] + ['RF']
bounds_3a = [(-1, 1) for _ in assets_3abc]

portfolio_weights_3a = run_portfolio_optimization(df_in_sample, assets_3abc, target_return, bounds_3a, rf_mean_annual_in_sample, 'Exercise 3a', 'Exercise_3a')

# 3) b) only long positions allowed
bounds_3b = [(0, 1) for _ in assets_3abc]

portfolio_weights_3b = run_portfolio_optimization(df_in_sample, assets_3abc, target_return, bounds_3b, rf_mean_annual_in_sample, 'Exercise 3b', 'Exercise_3b')

# 3) c) weights on industry assets constrained between -25% and 25% / riskfree asset bound is not constrained
industry_bounds = [(-0.25, 0.25) for _ in assets_3abc[:-1]]  # all assets except 'RF'
riskfree_bound = [(-np.inf, np.inf)]

bounds_3c = industry_bounds + riskfree_bound # cap does not apply for riskfree asset!

portfolio_weights_3c = run_portfolio_optimization(df_in_sample, assets_3abc, target_return, bounds_3c, rf_mean_annual_in_sample, 'Exercise 3c', 'Exercise_3c')

# 3) d) including the extra industry
assets_3d = list(industry_dict.values()) + ['RF']
bounds_3d = [(-1, 1) for _ in assets_3d]

portfolio_weights_3d = run_portfolio_optimization(df_in_sample, assets_3d, target_return, bounds_3d, rf_mean_annual_in_sample, 'Exercise 3d', 'Exercise_3d')



""" Form and analyze an industry time-series momentum (Hurst, Ooi, Pedersen (2013)) long-short  portfolio  that  selects  
    industries  based  not  only  on  returns  but  also  on  their  recent volatility. For this question use data from the in-sample period: 

    a. For  each  industry,  calculate  its  trailing  12-month  standard  deviation and 
    trailing 12-month cumulative excess return. For instance, in January 2024, 
    the trailing 12-month standard deviation will use data from the 12 months starting 
    January 2023 and ending December 2023. """

print('\n\n---------------- Exercise 4) ---------------')

# ----------- Define Data ----------------
# creating new df for a fresh start
df_in_sample_ex4 = df_combined_orig[(df_combined_orig['YearMonth'] >= '199401') & (df_combined_orig['YearMonth'] <= '201712')].copy() # 1994 included for now, to calculate Jan 1995

df_in_sample_ex4['YearMonth'] = pd.to_datetime(df_in_sample_ex4['YearMonth'], format='%Y%m')

industry_columns = [col for col in df_in_sample_ex4.columns if col not in ['YearMonth', 'Year', 'RF', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RMW', 'CMA']]

df_in_sample_ex4[industry_columns] = df_in_sample_ex4[industry_columns] / 100
df_in_sample_ex4['RF'] = df_in_sample_ex4['RF'] / 100

# --------------- Calculate Monthly Excess Returns -------------
# subtracting rf from all monthly asset returns
df_excess_returns = df_in_sample_ex4[industry_columns].sub(df_in_sample_ex4['RF'], axis=0)
df_excess_returns['YearMonth'] = df_in_sample_ex4['YearMonth']

# --------------- Calculate Trailing 12-Month Cumulative Excess Return -------------
def trailing_cumulative_excess_return(x):
    return np.prod(1 + x) - 1

# applying the function always to the trailing 12-months. shift(1) to not include the target month itself in the calculation
df_trailing_cum_excess_returns = df_excess_returns[industry_columns].shift(1).rolling(window=12).apply(trailing_cumulative_excess_return, raw=True)
df_trailing_cum_excess_returns['YearMonth'] = df_excess_returns['YearMonth']

# -------------- Calculate Trailing 12-Month Standard Deviation ------------------
# same for the std
df_trailing_std = df_excess_returns[industry_columns].shift(1).rolling(window=12).std()
df_trailing_std['YearMonth'] = df_excess_returns['YearMonth']

# --------------- Merging Results -------------
df_trailing_analysis = df_trailing_cum_excess_returns.merge(df_trailing_std, on='YearMonth', suffixes=('_cum_return', '_std'))

# filtering out 1994 now
df_trailing_analysis = df_trailing_analysis[df_trailing_analysis['YearMonth'] >= '1995-01-01'] # 1995!!!

print("Trailing 12-Month Cumulative Excess Returns and Standard Deviations by Month:")
print(df_trailing_analysis)

df_trailing_analysis.to_csv('ex_4a_trailing_data.csv', index=False)



""" 4) b)"""
selected_industries = list(industry_dict.values())[:10]
df_excess_returns['YearMonth'] = df_in_sample_ex4['YearMonth']

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
df_momentum['TSMOM_Portfolio_Return'] = df_momentum[industry_tsmom_columns].mean(axis=1)

# keeping only relevant columns
df_tsmom_result = df_momentum[['YearMonth', 'TSMOM_Portfolio_Return'] + industry_tsmom_columns]

print("TSMOM Portfolio Returns:")
print(df_tsmom_result)

df_tsmom_result.to_csv('ex4b_tsmom_results.csv', index=False)

# average excess return (annualized)
average_monthly_return = df_tsmom_result['TSMOM_Portfolio_Return'].mean()
average_annual_return = (average_monthly_return)*12

# standard deviation (annualized)
monthly_std_dev = df_tsmom_result['TSMOM_Portfolio_Return'].std()
annualized_std_dev = monthly_std_dev * (12**0.5)

# sharpe ratio 
sharpe_ratio = average_annual_return / annualized_std_dev # riskfree rate is not subtracted again because the tsmom returns are already excess returns

print(f'Avg monthly excess return: {average_monthly_return}, Avg annual excess return: {average_annual_return}, Avg annual std: {annualized_std_dev}, Annual sharpe ratio: {sharpe_ratio}')



""" 4) c) """
import statsmodels.api as sm
import pandas as pd
import numpy as np

df_regression_data = df_momentum[['YearMonth', 'TSMOM_Portfolio_Return']].merge(df_in_sample_ex4[['YearMonth', 'Mkt-RF', 'SMB', 'HML', 'Mom', 'RF']], on='YearMonth')

#columns_to_scale = [ 'Mkt-RF', 'SMB', 'HML', 'Mom']
df_regression_data[['Mkt-RF', 'SMB', 'HML', 'Mom']] = df_regression_data[['Mkt-RF', 'SMB', 'HML', 'Mom']] / 100

# dropping any missing data just in case
df_regression_data.dropna(inplace=True)

# creating dependent variable
Y = df_regression_data['TSMOM_Portfolio_Return']

# creating explanatory variables
X_capm = sm.add_constant(df_regression_data['Mkt-RF'])  
capm_model = sm.OLS(Y, X_capm).fit()

# creating explanatory variables for FF4 model
X_ff4 = sm.add_constant(df_regression_data[['Mkt-RF', 'SMB', 'HML', 'Mom']])  
ff4_model = sm.OLS(Y, X_ff4).fit()

df_regression_data.to_csv('ex_4c_regression_data.csv', index=False)

print("CAPM Model Results:")
print(capm_model.summary())
print("\nFama-French 4-Factor Model Results:")
print(ff4_model.summary())

# information ratio
def information_ratio(model):
    alpha = model.params['const']  # alpha is intercept
    tracking_error = model.resid.std() 
    return alpha / tracking_error

capm_ir = information_ratio(capm_model)
ff4_ir = information_ratio(ff4_model)

print(f"\nCAPM Information Ratio: {capm_ir}")
print(f"Fama-French 4-Factor Model Information Ratio: {ff4_ir}")


