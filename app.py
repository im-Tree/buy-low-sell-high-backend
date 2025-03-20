import pandas as pd
import numpy as np
import yfinance as yf
import talib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

app = Flask(__name__)

def compute_strategy(stock_symbol, strategy, short_window, long_window, stop_loss_pct):
    # 下载股票数据
    stock = yf.download(stock_symbol, start="2023-01-01", end="2024-01-01")
    stock = stock.reset_index()  # 这样 'Date' 会变成普通列
    stock.columns = stock.columns.droplevel(1)  # 删除第 1 层索引

    # 计算选定的技术指标
    if strategy == "SMA":
        stock["Short_MA"] = stock["Close"].rolling(window=short_window).mean()
        stock["Long_MA"] = stock["Close"].rolling(window=long_window).mean()
        stock["Buy_Signal"] = stock["Short_MA"] > stock["Long_MA"]
        stock["Sell_Signal"] = stock["Short_MA"] < stock["Long_MA"]
    elif strategy == "EMA":
        stock["Short_MA"] = talib.EMA(stock["Close"], timeperiod=short_window)
        stock["Long_MA"] = talib.EMA(stock["Close"], timeperiod=long_window)
        stock["Buy_Signal"] = stock["Short_MA"] > stock["Long_MA"]
        stock["Sell_Signal"] = stock["Short_MA"] < stock["Long_MA"]
    elif strategy == "RSI":
        stock["RSI"] = talib.RSI(stock["Close"], timeperiod=short_window)
        stock["Buy_Signal"] = stock["RSI"] < 30
        stock["Sell_Signal"] = stock["RSI"] > 70
    elif strategy == "MACD":
        stock["MACD"], stock["MACD_Signal"], _ = talib.MACD(stock["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
        stock["Buy_Signal"] = stock["MACD"] > stock["MACD_Signal"]
        stock["Sell_Signal"] = stock["MACD"] < stock["MACD_Signal"]

    # 计算净值曲线（NAV）
    stock["Daily_Return"] = stock["Close"].pct_change()
    stock["Strategy_Return"] = stock["Daily_Return"] * stock["Buy_Signal"].shift(1)
    stock["Cumulative_Return"] = (1 + stock["Strategy_Return"]).cumprod()

    # 计算绩效指标
    annual_gmrr = stock["Cumulative_Return"].iloc[-1] ** (252 / len(stock)) - 1
    annual_vol = stock["Strategy_Return"].std() * np.sqrt(252)
    sharpe_ratio = annual_gmrr / annual_vol if annual_vol > 0 else 0

    # 绘制交易信号图
    plt.figure(figsize=(12,6))
    plt.plot(stock["Close"], label="Close Price", color="black", alpha=0.7)
    plt.scatter(stock.index[stock["Buy_Signal"]], stock["Close"][stock["Buy_Signal"]], label="Buy Signal", marker="^", color="green", s=100)
    plt.scatter(stock.index[stock["Sell_Signal"]], stock["Close"][stock["Sell_Signal"]], label="Sell Signal", marker="v", color="red", s=100)
    plt.legend()
    plt.savefig("static/trading_signals.png")
    plt.close()

    return {
        "gmrr": f"{annual_gmrr:.2%}",
        "volatility": f"{annual_vol:.2%}",
        "sharpe_ratio": f"{sharpe_ratio:.2f}"
    }

@app.route("/strategy", methods=["POST"])
def strategy():
    data = request.json
    result = compute_strategy(data["symbol"], data["strategy"], int(data["short_window"]), int(data["long_window"]), float(data["stop_loss_pct"]))
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
