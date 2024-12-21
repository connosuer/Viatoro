import random

def generate_price_data(days, start_price=100, volatility=0.02):
    prices = [start_price]
    for _ in range(1, days):
        change = random.gauss(0, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(max(0.01, new_price))  # Ensure price doesn't go negative
    return prices

def calculate_rsi(prices, period=14):
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gain = [d if d > 0 else 0 for d in deltas]
    loss = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gain[:period]) / period
    avg_loss = sum(loss[:period]) / period
    
    rsi = []
    for i in range(period, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gain[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i-1]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi.append(100 - (100 / (1 + rs)))
    
    return [None] * period + rsi

def simulate_trading(prices, rsi, cash=10000, shares=0):
    trades = []
    for i in range(len(prices)):
        if rsi[i] is not None:
            if rsi[i] < 30 and cash > prices[i]:  # Oversold - Buy
                buy_shares = cash // prices[i]
                cash -= buy_shares * prices[i]
                shares += buy_shares
                trades.append(('BUY', buy_shares, prices[i]))
            elif rsi[i] > 70 and shares > 0:  # Overbought - Sell
                cash += shares * prices[i]
                trades.append(('SELL', shares, prices[i]))
                shares = 0
    
    final_value = cash + shares * prices[-1]
    return trades, final_value

def main():
    days = 100
    prices = generate_price_data(days)
    rsi = calculate_rsi(prices)
    
    trades, final_value = simulate_trading(prices, rsi)
    
    print("Simulation Results:")
    print(f"Starting Price: ${prices[0]:.2f}")
    print(f"Ending Price: ${prices[-1]:.2f}")
    print(f"\nTrades:")
    for trade in trades:
        print(f"{trade[0]} {trade[1]} shares at ${trade[2]:.2f}")
    print(f"\nFinal Portfolio Value: ${final_value:.2f}")
    
    # Calculate and return
    initial_value = 10000  #starting cash
    returns = (final_value - initial_value) / initial_value * 100
    print(f"Total Return: {returns:.2f}%")

if __name__ == "__main__":
    main()