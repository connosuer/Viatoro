import numpy as np
import pandas as pd #Used for handling time series data and dataframes.
import yfinance as yf
from deap import base, creator, tools, algorithms #library for rapid prototyping and testing
import random 
import matplotlib.pyplot as plt

class GeneticTradingStrategy: 
    def __init__(self, symbol, start_date, end_date): 
        #initializing strategy with stock symbols and date range
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._fetch_data()

    def _fetch_data(self):
        data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        data['Returns'] = data['Close'].pct_change()
        print(f"Downloaded {len(data)} rows of data")
        return data
    
    def _calculate_indicators(self): 
        #calculating technical indicators: short and long SMA's and RSI
        self.data['SMA_short'] = self.data['Close'].rolling(window=10).mean()
        self.data['SMA_long'] = self.data['Close'].rolling(window=30).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['Close'],14)

    def _calculate_rsi(self,prices,window):
        #calculating relative strength index
        delta = prices.diff()
        gain = (delta.where(delta > 0,0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0,0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100/(1+rs))
    
    def _evaluate_strategy(self, individual):
        try:
            sma_short = individual[0]
            sma_long = individual[1]
            rsi_lower = individual[2]
            rsi_upper = individual[3]

            self.data['SMA_short'] = self.data['Close'].rolling(window=sma_short).mean()
            self.data['SMA_long'] = self.data['Close'].rolling(window=sma_long).mean()
            self.data['RSI'] = self._calculate_rsi(self.data['Close'], 14)

            self.data['Position'] = 0
            self.data.loc[(self.data['SMA_short'] > self.data['SMA_long']) & (self.data['RSI'] < rsi_lower), 'Position'] = 1
            self.data.loc[(self.data['SMA_short'] < self.data['SMA_long']) & (self.data['RSI'] > rsi_upper), 'Position'] = -1

            self.data['Strategy_Returns'] = self.data['Position'].shift(1) * self.data['Returns']
            
            if self.data['Strategy_Returns'].std() == 0:
                return (-np.inf,)  # Return a very low fitness if there's no variation in returns
            
            sharpe_ratio = np.sqrt(252) * self.data['Strategy_Returns'].mean() / self.data['Strategy_Returns'].std()
            print(f"Evaluating: {individual}, Sharpe Ratio: {sharpe_ratio}")
            return (sharpe_ratio,)
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return (-np.inf,)  # Return a very low fitness in case of any error
        
    def run_optimization(self, population_size=50, generations= 50): 
        #setting up genetic algo components 
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        #defining genes: SMA periods and RSI thresholds 
        toolbox.register("attr_int", random.randint, 5, 50)
        toolbox.register("attr_rsi", random.randint, 1, 100)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_int, toolbox.attr_int, toolbox.attr_rsi, toolbox.attr_rsi), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # register genetic operators
        toolbox.register("evaluate", self._evaluate_strategy)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=5, up=100, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # create initial population
        population = toolbox.population(n=population_size)
        
        # set up statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # run the genetic algorithm
        final_pop, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, 
                                                 ngen=generations, stats=stats, verbose=True)

        # get and print the best individual
        best_individual = tools.selBest(final_pop, k=1)[0]
        print(f"Best strategy parameters: {best_individual}")
        print(f"Best Sharpe Ratio: {best_individual.fitness.values[0]}")
        

        return best_individual, logbook

    def plot_results(self, best_individual):
        # plot the performance of the optimized strategy vs buy-and-hold
        self._evaluate_strategy(best_individual)
        self.data['Cumulative_Returns'] = (1 + self.data['Returns']).cumprod()
        self.data['Strategy_Cumulative_Returns'] = (1 + self.data['Strategy_Returns']).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Cumulative_Returns'], label='Buy and Hold')
        plt.plot(self.data.index, self.data['Strategy_Cumulative_Returns'], label='Optimized Strategy')
        plt.title(f'Strategy Performance: {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.show()

def main():
    # get user input for stock symbol and date range
    symbol = input("Enter stock symbol (e.g., AAPL): ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    # create and run the genetic trading strategy
    strategy = GeneticTradingStrategy(symbol, start_date, end_date)
    best_individual, logbook = strategy.run_optimization()
    strategy.plot_results(best_individual)

# [All the existing code remains here]

if __name__ == "__main__":
    main()

"""
This script implements a genetic algorithm to optimize a trading strategy for stocks.

Key components and workflow:

1. Data Collection:
   - Uses yfinance to download historical stock data for a specified symbol and date range.

2. Trading Strategy:
   - Based on two technical indicators: Simple Moving Averages (SMA) and Relative Strength Index (RSI).
   - Goes long when the short-term SMA is above the long-term SMA and RSI is below a lower threshold.
   - Goes short when the short-term SMA is below the long-term SMA and RSI is above an upper threshold.

3. Genetic Algorithm Optimization:
   - Uses the DEAP library to implement a genetic algorithm.
   - Optimizes four parameters: short SMA period, long SMA period, RSI lower threshold, and RSI upper threshold.
   - Fitness function: Sharpe ratio of the strategy's returns.
   - Evolves the population over multiple generations to find the best combination of parameters.

4. Evaluation:
   - Calculates the strategy's returns based on the optimized parameters.
   - Computes the Sharpe ratio to measure risk-adjusted performance.

5. Visualization:
   - Plots the cumulative returns of the optimized strategy against a buy-and-hold approach.

How it works:
1. User inputs the stock symbol and date range.
2. The script downloads the stock data and initializes the genetic algorithm.
3. It runs the optimization process, evolving the population to find the best strategy parameters.
4. The best-performing strategy (highest Sharpe ratio) is identified.
5. Results are displayed, showing the optimized parameters and a performance plot.


Key libararies: 
deap (Distributed Evolutionary Algorithms in Python):
   - Evolutionary computation framework.
   - Provides tools for implementing genetic algorithms.
   - We import specific modules:
     * base: Basic structures.
     * creator: For creating custom classes (fitness and individuals).
     * tools: Genetic operators and utilities.
     * algorithms: Pre-implemented evolutionary algorithms.

random:
   - Python's built-in random number generator.
   - Used for generating random values in the genetic algorithm.

This approach combines financial analysis with machine learning techniques to develop and optimize
a quantitative trading strategy. It demonstrates the application of genetic algorithms in financial
modeling and automated strategy development.

Note: This is a simplified model for practice. 
"""
