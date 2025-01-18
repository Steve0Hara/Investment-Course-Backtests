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


# same stuff as always
df_industry_returns = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Return')
df_factors = pd.read_excel('Assignment Investments 2024 data clean.xlsx', sheet_name='Factors')

df_industry_returns['YearMonth'] = df_industry_returns['YearMonth'].astype(str)
df_factors['YearMonth'] = df_factors['YearMonth'].astype(str)
df_combined_orig = df_industry_returns.merge(df_factors, on='YearMonth', how='left')

industry_cols = list(industry_dict.values()) 
df_combined_orig[industry_cols] = df_combined_orig[industry_cols] / 100
df_combined_orig['Mkt-RF'] = df_combined_orig['Mkt-RF'] / 100
df_combined_orig['RF'] = df_combined_orig['RF'] / 100

df_holdout_sample5 = df_combined_orig[(df_combined_orig['YearMonth'] >= '201801') & (df_combined_orig['YearMonth'] <= '202312')].copy()
industry_columns = [industry_dict[key] for key in industry_dict]
df_industry_returns = df_holdout_sample5[['YearMonth', 'RF'] + industry_columns]

df_industry_returns['YearMonth'] = pd.to_datetime(df_industry_returns['YearMonth'], format='%Y%m')

# Ffunction for the equal-weigthed portfolio return
def calculate_equal_weighted_portfolio(df, industry_cols):
    df = df.sort_values('YearMonth').copy()

    # initial weigths
    num_industries = len(industry_cols)
    initial_weight = 1 / num_industries

    df['Portfolio_Return'] = 0.0

    df['Year'] = df['YearMonth'].dt.year
    grouped = df.groupby('Year')

    for year, group in grouped:
        # resetting weigths yearly
        weights = np.full(num_industries, initial_weight)

        for idx, row in group.iterrows():
            # portfolio return for current month
            industry_returns = row[industry_cols].values  
            monthly_return = np.dot(weights, industry_returns)
            df.at[idx, 'Portfolio_Return'] = monthly_return

            # updating weigths
            weights = weights * (1 + industry_returns)
            weights = weights / weights.sum()  # normalizing weights to sum to 1

    return df

df_with_portfolio = calculate_equal_weighted_portfolio(df_industry_returns, industry_columns)

# excess returns
df_with_portfolio['Excess_Portfolio_Return'] = df_with_portfolio['Portfolio_Return'] - df_with_portfolio['RF']

average_excess_return = df_with_portfolio['Excess_Portfolio_Return'].mean() * 12  # Annualized

std_excess_return = df_with_portfolio['Excess_Portfolio_Return'].std() * np.sqrt(12)  # Annualized

sharpe_ratio = average_excess_return / std_excess_return

print(f"Average Excess Return (Annualized): {average_excess_return}")
print(f"Standard Deviation of Excess Returns (Annualized): {std_excess_return}")
print(f"Sharpe Ratio: {sharpe_ratio}")

df_with_portfolio.to_csv('ex5d_EqualWeightedPortfolio.csv', index=False)