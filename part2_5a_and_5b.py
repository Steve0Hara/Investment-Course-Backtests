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


# weights taken from 3a and 3c
weights_dict = {
    "5a": [-0.0767, 0.1797, 0.1343, 0.0594, 0.2285, -0.0809, 0.0238, -0.1238, 0.0512, 0.0143, 0.5903],
    "5b": [-0.0767, 0.1797, 0.1343, 0.0594, 0.2285, -0.0809, 0.0238, -0.1238, 0.0512, 0.0143, 0.5903]
}

df_industry_returns = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Return')
df_factors = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Factors')

df_industry_returns['YearMonth'] = df_industry_returns['YearMonth'].astype(str)
df_factors['YearMonth'] = df_factors['YearMonth'].astype(str)
df_combined_orig = df_industry_returns.merge(df_factors, on='YearMonth', how='left')

industry_cols = list(industry_dict.values())  
df_combined_orig[industry_cols] = df_combined_orig[industry_cols] / 100
df_combined_orig['RF'] = df_combined_orig['RF'] / 100

# holdout period now
df_holdout_sample5 = df_combined_orig[(df_combined_orig['YearMonth'] >= '201801') & (df_combined_orig['YearMonth'] <= '202312')].copy()

df_holdout_sample5['YearMonth'] = pd.to_datetime(df_holdout_sample5['YearMonth'], format='%Y%m')
industry_columns = [industry_dict[key] for key in industry_dict]
df_industry_returns = df_holdout_sample5[['YearMonth', 'RF'] + industry_columns]

# calculating portfolio variance for starting weights
def calculate_portfolio_performance(df, weights, exercise_name):

    weights = np.array(weights)
    
    asset_returns = df[industry_columns + ['RF']].values
    portfolio_returns = np.dot(asset_returns, weights)  
    df['Portfolio_Return'] = portfolio_returns
    
    # calculate excess portfolio returns
    df['Excess_Portfolio_Return'] = df['Portfolio_Return'] - df['RF']

    average_annual_excess_return = df['Excess_Portfolio_Return'].mean() * 12  # annualized excess return
    std_annual_excess_return = df['Excess_Portfolio_Return'].std() * np.sqrt(12)  # annualized std
    sharpe_ratio = average_annual_excess_return / std_annual_excess_return  # sharpe 

    print(f"\nExercise {exercise_name}:")
    print(f"Average Annual Excess Return: {average_annual_excess_return:.4f}")
    print(f"Annualized Standard Deviation of Excess Returns: {std_annual_excess_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    
    df.to_csv(f'ex{exercise_name}_PortfolioPerformance.csv', index=False)

# Run calculations for each exercise
for ex, weights in weights_dict.items():
    calculate_portfolio_performance(df_industry_returns.copy(), weights, ex)