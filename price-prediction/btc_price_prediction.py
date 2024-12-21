# -*- coding: utf-8 -*-
"""BTC_price_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1apSproxKbOA_bOEUk_U7qXmcCYc0nXwj
"""

!pip install mwclient

"""#Sentiment"""

import mwclient
import time

site = mwclient.Site('en.wikipedia.org')
page = site.pages["Bitcoin"]

revs = list(page.revisions())

revs[0]

revs = sorted(revs, key=lambda rev: rev["timestamp"])

revs[0]

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def find_sentiment(text):
  sent = sentiment_pipeline([text[:250]])[0]
  score = sent["score"]
  if sent["label"] == "NEGATIVE":
    score *=-1
  return score

edits = {}

for rev in revs:
  date = time.strftime("%Y-%m-%d", rev["timestamp"])

  if date not in edits:
    edits[date] = dict(sentiments=list(), edit_count=0)

    edits[date]["edit_count"] += 1

    comment = rev["comment"]
    edits[date]["sentiments"].append(find_sentiment(comment))

from statistics import mean

for key in edits:
    if len(edits[key]["sentiments"]) > 0:
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0

    del edits[key]["sentiments"]

import pandas as pd

edits_df = pd.DataFrame.from_dict(edits, orient="index")

edits_df

edits_df.index = pd.to_datetime(edits_df.index)

from datetime import datetime

dates = pd.date_range(start="2009-03-08",end=datetime.today())

dates

edits_df = edits_df.reindex(dates, fill_value=0.0)

edits_df

rolling_edits = edits_df.rolling(30).mean()

rolling_edits

rolling_edits = rolling_edits.dropna()

rolling_edits

rolling_edits.to_csv("wikepedia_edits.csv")

"""#Prediction"""

import yfinance as yf
import os
import pandas as pd

btc_ticker = yf.Ticker("BTC-USD")

btc = btc_ticker.history(period="max")

btc

btc.index = pd.to_datetime(btc.index)

del btc["Dividends"]
del btc["Stock Splits"]

btc.columns = [c.lower() for c in btc.columns]

btc.plot.line(y="close", use_index=True)

wiki = pd.read_csv("wikepedia_edits.csv", index_col=0, parse_dates=True)

wiki

print("BTC index timezone:", btc.index.tz)
print("Wiki index timezone:", wiki.index.tz)

# If btc has tz-aware index and wiki is naive
btc.index = btc.index.tz_localize(None)

# Or if wiki has tz-aware index and btc is naive
wiki.index = wiki.index.tz_localize(None)

btc = btc.merge(wiki, left_index=True, right_index=True)

btc["tomorrow"] = btc["close"].shift(-1)

btc["target"] = (btc["tomorrow"] > btc["close"]).astype(int)

btc["target"].value_counts()

btc

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)

train = btc.iloc[:-200]
test = btc.iloc[-200:]

predictors = ["close", "volume", "low", "high", "edit_count", "sentiment", "neg_sentiment"]
model.fit(train[predictors], train["target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["target"], preds)

def predict(train, test, predictors, model):
  model.fit(train[predictors], train["target"])
  preds = model.predict(test[predictors])
  preds = pd.Series(preds, index=test.index, name="predictions")
  combined = pd.concat([test["target"], preds], axis=1)
  return combined

def backtest(data, model, predictors, start=1095, step=150):
  all_predictions = []

  for i in range(start, data.shape[0], step):
    train = data.iloc[0:i].copy()
    test = data.iloc[i:(i+step)].copy()
    predictions = predict(train, test, predictors, model)
    all_predictions.append(predictions)

  return pd.concat(all_predictions)

from xgboost import XGBClassifier

model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
predictions = backtest(btc, model, predictors)

predictions["predictions"].value_counts()

precision_score(predictions["target"],predictions["predictions"])

def compute_rolling(btc):
    horizons = [2,7,60,365]
    new_predictors = ["close", "sentiment", "neg_sentiment"]

    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods=1).mean()

        ratio_column = f"close_ratio_{horizon}"
        btc[ratio_column] = btc["close"] / rolling_averages["close"]

        edit_column = f"edit_{horizon}"
        btc[edit_column] = rolling_averages["edit_count"]

        rolling = btc.rolling(horizon, closed='left', min_periods=1).mean()
        trend_column = f"trend_{horizon}"
        btc[trend_column] = rolling["target"]

        new_predictors+= [ratio_column, trend_column, edit_column]
    return btc, new_predictors

btc, new_predictors = compute_rolling(btc.copy())

predictions = backtest(btc, model, new_predictors)

precision_score(predictions["target"], predictions["predictions"])

predictions
