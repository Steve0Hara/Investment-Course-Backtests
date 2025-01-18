import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set(style="whitegrid")
plt.rcParams['axes.facecolor'] = '#f4f4f8'

portfolio_files = {
    "Portfolio of 3a)": "ex5a_PortfolioPerformance.csv",
    "Portfolio of 3c)": "ex5b_PortfolioPerformance.csv",
    "TSMOM Portfolio": "ex5c_TSMOM_Portfolio_Data.csv",
    "Equal-Weight Portolfio": "ex5d_EqualWeightedPortfolio.csv",
    "Market Portfolio": "ex5e_Market_Performance_Stats.csv"
}

required_columns = {
    "Portfolio of 3a)": "Excess_Portfolio_Return",
    "Portfolio of 3c)": "Excess_Portfolio_Return",
    "TSMOM Portfolio": "TSMOM_Portfolio_Return",
    "Equal-Weight Portolfio": "Excess_Portfolio_Return",
    "Market Portfolio": "Excess_Return"
}

cumulative_returns = {}

for label, file in portfolio_files.items():
  
    df = pd.read_csv(file)
    
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    
    required_col = required_columns[label]
    if required_col not in df.columns:
        print(f"Error: {required_col} is missing in {file}. Skipping.")
        continue
    
    df['Cumulative_Excess_Return'] = (1 + df[required_col]).cumprod() - 1

    cumulative_returns[label] = df[['YearMonth', 'Cumulative_Excess_Return']]

custom_colors = sns.color_palette("crest", 3) + sns.color_palette("tab10", 2)

plt.figure(figsize=(14, 8))

for idx, (label, data) in enumerate(cumulative_returns.items()):
    plt.plot(
        data['YearMonth'],
        data['Cumulative_Excess_Return'],
        label=f"{label}",
        color=custom_colors[idx],
        linewidth=2.5
    )

plt.title("Cumulative Excess Returns of Portfolios", fontsize=18, weight="bold", color="#333333")
plt.xlabel("Year", fontsize=14, weight="bold", color="#333333")
plt.ylabel("Cumulative Excess Return", fontsize=14, weight="bold", color="#333333")
plt.xticks(fontsize=12, rotation=45, color="#333333")
plt.yticks(fontsize=12, color="#333333")

plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
plt.legend(title="Portfolios", fontsize=12, title_fontsize=13, loc="lower left", frameon=True, edgecolor="gray")

plt.tight_layout()
plt.savefig("Exercise_5_cumulative_excess_returns_mixed_colors.png", dpi=300, bbox_inches='tight')
plt.show()
