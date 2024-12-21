import backtrader as bt
import yfinance as yf

class ImprovedStrategy(bt.Strategy):
    params = (
        ('fast_ema', 12),
        ('slow_ema', 26),
        ('signal_ema', 9),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('atr_period', 14),
        ('atr_multiplier', 2),
    )

    def __init__(self):
        self.dataclose = self.data.close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
        self.ema_fast = bt.indicators.EMA(self.data, period=self.p.fast_ema)
        self.ema_slow = bt.indicators.EMA(self.data, period=self.p.slow_ema)
        self.macd = bt.indicators.MACD(self.data, 
                                       period_me1=self.p.fast_ema, 
                                       period_me2=self.p.slow_ema, 
                                       period_signal=self.p.signal_ema)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
        self.sma50 = bt.indicators.SMA(self.data, period=50)
        self.sma200 = bt.indicators.SMA(self.data, period=200)

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.crossover > 0 and self.dataclose[0] > self.sma50[0] > self.sma200[0] and self.rsi[0] < self.p.rsi_oversold:
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')
                self.order = self.buy()
        else:
            if self.crossover < 0 or self.rsi[0] > self.p.rsi_overbought or self.dataclose[0] < self.sma50[0]:
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')
                self.order = self.sell()

        # Implement trailing stop
        if self.position:
            pclose = self.data.close[0]
            pstop = self.position.price - self.atr[0] * self.p.atr_multiplier

            if pclose < pstop:
                self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

def run_backtest():
    cerebro = bt.Cerebro()
    df = yf.download('NVDA', start='2023-01-01', end='2024-01-01')
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(ImprovedStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()

if __name__ == '__main__':
    run_backtest()
    
# ------------------ Explanation of the Strategy ------------------
# This strategy combines Bollinger Bands, RSI, and MACD indicators for trading Nvidia stock.
# - **Bollinger Bands**: Measures market volatility. The strategy buys when the price crosses the lower band upwards, and sells when it crosses the upper band downwards.
# - **RSI (Relative Strength Index)**: A momentum oscillator. Buy signals occur when RSI is below 30 (oversold), and sell signals occur when RSI is above 70 (overbought).
# - **MACD (Moving Average Convergence Divergence)**: Used to gauge momentum. The strategy buys when the MACD line crosses above the signal line and sells when it crosses below.
# 
# **Backtesting**: The code uses the `backtrader` library to backtest the strategy on historical Nvidia stock data. The initial portfolio is set to $10,000, with a commission of 0.1% per trade.
# 
# **Results**: After running the backtest, the strategy will display the starting and final portfolio values and plot the performance over time. The output provides insights into the strategy's effectiveness.
# ---------------------------------------------------------------
