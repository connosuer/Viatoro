# Quantitative Analysis Practice Repository

This repository contains a collection of practice projects and implementations focused on quantitative analysis, algorithmic trading, and financial modeling. Each implementation demonstrates different aspects of financial analysis and machine learning applications in finance.

## Repository Structure

### Trading Strategies (`/trading-strategies`)
- **nvidia_bollinger_rsi_macd_strategy.py**: Trading strategy implementation using technical indicators (Bollinger Bands, RSI, and MACD) specifically for NVIDIA stock
- **rsi_trading_simulation.py**: Trading simulation based on Relative Strength Index (RSI) signals
- **genetic_algo_tradeOp.py**: Implementation of a genetic algorithm for optimizing trading strategies
- **RL_agent_MarketMaking.ipynb**: Reinforcement Learning agent designed for market making strategies

### Machine Learning (`/machine-learning`)
- **clustering_stocks_practice.ipynb**: Stock clustering implementation using KMeans algorithm
- **kmeans.py**: Standalone KMeans algorithm implementation
- **Linear Regression Studies:**
  - `linearRegression_relationships_prac.ipynb`: Understanding relationships in linear regression models
  - `linear_regressionPractice1.ipynb`: Basic linear regression model practice
  - `linear_regression_retailSales_prac.ipynb`: Linear regression applied to retail chain sales data
  - `multiple_regression_analysis.ipynb`: Multiple regression analysis on South Korea's economic indicators

### Portfolio Optimization (`/portfolio-optimization`)
- **monte_carlo_portfolio_optimization.py**: Portfolio optimization using Monte Carlo simulation
- **VaR_CVaR_monteCarlo.py**: Risk analysis using Value at Risk (VaR) and Conditional Value at Risk (CVaR)
- **riskFree_marketPort.ipynb**: Optimal portfolio construction combining risk-free assets and market portfolio

### Price Prediction (`/price-prediction`)
- **BTC_price_prediction.ipynb** & **btc_price_prediction.py**: Bitcoin price prediction models
- **S&P500Predictor.ipynb**: S&P 500 index prediction implementation
- **TimeSeries_model.ipynb**: Time series analysis focused on Microsoft stock

### Miscellaneous (`/misc`)
- **expense_tracker.py**: Simple expense tracking application
- **sentiment_analysis.py**: Financial sentiment analysis implementation
- **py_Data_Analysis_Weather.ipynb**: Weather dataset analysis practice

## Technologies Used
- Python 3.x
- Key Libraries:
  - pandas: Data manipulation and analysis
  - numpy: Numerical computations
  - scikit-learn: Machine learning implementations
  - yfinance: Financial data acquisition
  - matplotlib/seaborn: Data visualization
  - tensorflow/pytorch: Deep learning (for specific implementations)

## Prerequisites
To run these implementations, you'll need:
```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn tensorflow
```
Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Usage
Each script/notebook is self-contained with its own documentation and comments. To use any implementation:

1. Navigate to the specific directory
2. Open the .ipynb file in Jupyter Notebook/Lab or run the .py file:
   ```bash
   python filename.py
   ```

## Learning Objectives
This repository demonstrates proficiency in:
- Financial data analysis and manipulation
- Machine learning applications in finance
- Trading strategy development and backtesting
- Portfolio optimization techniques
- Risk analysis and management
- Time series analysis and prediction

## Contributing
Feel free to fork this repository and submit pull requests for improvements. Areas for contribution:
- Bug fixes
- Documentation improvements
- New implementations
- Performance optimizations

## Note
These implementations are for educational purposes and should not be used for actual trading without proper validation and risk management.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Disclaimer: This repository contains practice implementations and should not be used for financial advice or real trading without proper validation.*
