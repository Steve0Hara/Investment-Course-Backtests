import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")
cmap = sns.color_palette("crest", as_cmap=True) 

plt.rcParams['axes.facecolor'] = '#f4f4f8'  

# 1 a) cumulative returns 
def plot_cumulative_returns(df, title="Cumulative Return of Industry vs Market Portfolio"):
    plt.figure(figsize=(12, 7))

    sns.lineplot(data=df, x='YearMonth', y='Hlth_Cumulative_Return', label='Health Industry Cumulative Return', color='gray')
    sns.lineplot(data=df, x='YearMonth', y='Market_Cumulative_Return', label='Market Portfolio Cumulative Return', color=cmap(0.4))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    
    plt.title(title, fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('Exercise_1a.png', dpi=300, bbox_inches='tight')
    

# 1 c) Function to plot CAPM betass
def plot_capm_betas(industry_tickers, betas, title="CAPM Beta for Each Industry"):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=industry_tickers, y=betas, palette="crest")
    
    plt.xlabel("Industry Ticker")
    plt.ylabel("CAPM Beta")
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('Exercise_1c.png', dpi=300, bbox_inches='tight')
   

# 2) function to plot the efficient frontier, CAL, and individual assets
def plot_efficient_frontier(portfolio_volatilities, portfolio_returns, tangency_volatility, tangency_return, rf, industry_volatilities, industry_returns, industry_labels, filename):
    plt.figure(figsize=(10, 7))
    
    plt.plot(portfolio_volatilities, portfolio_returns, color=cmap(0.7), linestyle='-', label='Efficient Frontier', linewidth=2)
    plt.plot(tangency_volatility, tangency_return, 'o', color='#d92b6a', label='Tangency Portfolio')

    cal_x = [0, max(portfolio_volatilities)*1.5]
    cal_slope = (tangency_return - rf) / tangency_volatility
    cal_y = [rf, rf + cal_slope * max(portfolio_volatilities)*1.5]
    plt.plot(cal_x, cal_y, color='#6baed6', linestyle='-', label='Capital Allocation Line', linewidth=2)

    plt.scatter(industry_volatilities, industry_returns, color='#fc9272', label='Individual Assets')
    for i, label in enumerate(industry_labels):
        plt.annotate(label, (industry_volatilities[i], industry_returns[i]), textcoords="offset points", xytext=(5,5), ha='center')

    plt.xlim(0, 0.3)
    plt.ylim(0, 0.25)
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Expected Return')
    plt.title('In-sample Period Efficient Frontier, CAL and Tangency Portfolio', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(filename) + '.png', dpi=300, bbox_inches='tight')

# 3) function to plot the efficient frontier, CAL, tangency portfolio, optimal portfolio, and individual assets
def plot_efficient_frontier_plus_extra_portfolio(portfolio_volatilities, portfolio_returns, tangency_volatility, tangency_return, 
                            optimal_volatility, optimal_return, rf, industry_volatilities, industry_returns, 
                            industry_labels, filename):
    plt.figure(figsize=(10, 7))

    plt.plot(portfolio_volatilities, portfolio_returns, color=cmap(0.7), linestyle='-', label='Efficient Frontier', linewidth=2)

    # calculating CAL
    cal_slope = (tangency_return - rf) / tangency_volatility
    cal_x = [0, max(portfolio_volatilities)*1.5]
    cal_y = [rf, rf + cal_slope * max(portfolio_volatilities)*1.5]
    plt.plot(cal_x, cal_y, color='#6baed6', linestyle='-', label='Capital Allocation Line', linewidth=2)

    plt.plot(optimal_volatility, optimal_return, 'o', color='#fdb863', label='Optimal Portfolio', markeredgewidth=2)

    plt.annotate('Investor Portfolio', 
                 (optimal_volatility, optimal_return), 
                 textcoords="offset points", 
                 xytext=(60,0), 
                 ha='center', 
                 fontsize=12, 
                 fontweight='bold', 
                 color='#fdb863')
    
    plt.plot(tangency_volatility, tangency_return, 'o', color='#d92b6a', label='Tangency Portfolio')
    plt.scatter(industry_volatilities, industry_returns, color='#fc9272', label='Individual Assets')

    for i, label in enumerate(industry_labels):
        plt.annotate(label, (industry_volatilities[i], industry_returns[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=12)

    plt.xlim(0, 0.3)
    plt.ylim(0, 0.25)
    plt.yticks(np.arange(0, 0.25, 0.02))  
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('Portfolio Standard Deviation')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier, CAL, Investor & Tangency Portfolio', fontsize=16)
    plt.legend(loc="upper left") 
    plt.tight_layout()
    plt.savefig(str(filename) + '.png', dpi=300, bbox_inches='tight')


  

