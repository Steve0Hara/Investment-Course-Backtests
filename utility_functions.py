import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from plots import *

# function to calculate portfolio return and variance
def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# function to calculate the negative Sharpe ratio
def negative_sharpe_ratio(weights, expected_returns, cov_matrix, rf):
    ret = portfolio_return(weights, expected_returns)
    vol = np.sqrt(portfolio_variance(weights, cov_matrix))
    sharpe_ratio = (ret - rf) / vol
    return -sharpe_ratio  # negative because we minimize

# function to compute the efficient frontier
def compute_efficient_frontier(expected_returns, cov_matrix, target_returns, constraints, bounds, method='SLSQP'):
    num_assets = len(expected_returns)
    # print(f'Anzahl Assets {num_assets}')

    # initial equal guess for weigths
    init_guess = np.repeat(1/num_assets, num_assets)

    portfolio_returns = []
    portfolio_volatilities = []

    for target_return in target_returns:
        
        constraints_with_target = constraints + (
            {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, expected_returns) - target_return},
        )
        
        # minimize portfolio variance for the target return
        result = minimize(
            portfolio_variance,
            init_guess,
            args=(cov_matrix,),
            method=method,
            constraints=constraints_with_target,
            bounds=bounds
        )
        
        if result.success:
            weights = result.x
            ret = portfolio_return(weights, expected_returns)
            vol = np.sqrt(portfolio_variance(weights, cov_matrix))
            portfolio_returns.append(ret)
            portfolio_volatilities.append(vol)

        else:
            # print("Optimization failed for target return:", target_return)
            pass

    return portfolio_volatilities, portfolio_returns

# function to compute the tangency portfolio
def compute_tangency_portfolio(expected_returns, cov_matrix, rf, constraints, bounds, method='SLSQP'):
    num_assets = len(expected_returns)

    # initial wigth guess
    init_guess = np.repeat(1/num_assets, num_assets)

    result_tangent = minimize(
        negative_sharpe_ratio,
        init_guess,
        args=(expected_returns, cov_matrix, rf),
        method=method,
        constraints=constraints,
        bounds=bounds
    )

    if result_tangent.success:
        tangency_weights = result_tangent.x
        tangency_return = portfolio_return(tangency_weights, expected_returns)
        tangency_volatility = np.sqrt(portfolio_variance(tangency_weights, cov_matrix))
        tangency_sharpe = (tangency_return - rf) / tangency_volatility
        return tangency_volatility, tangency_return, tangency_weights, tangency_sharpe
    
    else:
        # print("Optimization failed for the tangency portfolio")
        return None
    
# function to minimize portfolio variance for a given target return
def compute_target_return_portfolio(expected_returns, cov_matrix, rf, constraints, bounds, method='SLSQP'):
    num_assets = len(expected_returns)
    init_guess = np.repeat(1 / num_assets, num_assets)

    # minimize portfolio variance
    result = minimize(
        portfolio_variance,
        init_guess,
        args=(cov_matrix,),
        method=method,
        constraints=constraints,
        bounds=bounds
    )

    if result.success:
        optimal_weights = result.x
        optimal_return = portfolio_return(optimal_weights, expected_returns)
        optimal_volatility = np.sqrt(portfolio_variance(optimal_weights, cov_matrix))
        optimal_sharpe = (optimal_return - rf) / optimal_volatility
        return optimal_volatility, optimal_return, optimal_weights, optimal_sharpe
    
    else:
        # print("Optimization failed for the target return portfolio")
        return None, None, None

def get_portfolio_data(df_sample, assets):
    returns_data = df_sample[assets] / 100  

    mean_returns = returns_data.mean() * 12  

    cov_matrix = returns_data.cov() * 12

    # setting last row and column to zero because the risk free rate should have 0 variance!!!
    cov_matrix.iloc[-1, :] = 0
    cov_matrix.iloc[:, -1] = 0

    print(f'{cov_matrix}')

    return mean_returns.values, cov_matrix.values, returns_data

def run_portfolio_optimization(df_sample, assets, target_return, bounds, rf_mean_annual, plot_title, file_suffix):
 
    mean_returns, cov_matrix, returns_data = get_portfolio_data(df_sample, assets)

    # weights sum to 1 and portfolio return equals target return
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, mean_returns) - target_return}
    )

    # optimal investor portfolio
    optimal_volatility, optimal_return, optimal_weights, optimal_sharpe = compute_target_return_portfolio(mean_returns, cov_matrix, rf_mean_annual, constraints, bounds)
    optimal_weights_dict = {asset: weight for asset, weight in zip(assets, optimal_weights)}

    # exclude the risk-free asset for efficient frontier computation
    mean_returns_frontier = mean_returns[:-1]
    cov_matrix_frontier = cov_matrix[:-1, :-1]

    # target_returns_efficient_frontier = np.linspace(mean_returns_frontier.min(), mean_returns_frontier.max(), 100)
    target_returns_efficient_frontier = np.linspace(0, 0.25, 1000)
    target_returns_efficient_frontier = np.round(target_returns_efficient_frontier, 8)

    constraints_frontier = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},)
    bounds_frontier = bounds[:-1]

    # compute efficient frontier
    portfolio_volatilities, portfolio_returns = compute_efficient_frontier(mean_returns_frontier, cov_matrix_frontier, target_returns_efficient_frontier, constraints_frontier, bounds_frontier)

    # compute tangency portfolio
    tangency_volatility, tangency_return, tangency_weights, tangency_sharpe = compute_tangency_portfolio(mean_returns_frontier, cov_matrix_frontier, rf_mean_annual, constraints_frontier, bounds_frontier)

    asset_returns = mean_returns
    asset_volatilities = np.sqrt(np.diag(cov_matrix))
    asset_labels = assets

    # plot everything
    plot_efficient_frontier_plus_extra_portfolio(portfolio_volatilities, portfolio_returns, tangency_volatility, tangency_return, 
                                                 optimal_volatility, optimal_return, rf_mean_annual, asset_volatilities, asset_returns, 
                                                 asset_labels, file_suffix)

    # print optimal portfolio results
    print(f'\n{plot_title}: Optimal Portfolio for a mean-variance investor with a desired expected return of {target_return}:')
    for asset, weight in optimal_weights_dict.items():
        print(f'{asset}: {weight:.4f}')
    print(f'Return: {optimal_return:.4f}, Volatility: {optimal_volatility:.4f}, Sharpe Ratio: {optimal_sharpe:.4f}')
    
    return optimal_weights_dict, optimal_return, optimal_volatility, optimal_sharpe
