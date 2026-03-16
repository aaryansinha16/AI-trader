# Learning pipeline (ML training)

# 1. The learning pipeline overview

Your raw tick data is useless directly for ML.

You convert it into **features + labels**.

Pipeline:

```
Tick Data
   ↓
Resampling (1s / 1m candles)
   ↓
Feature Engineering
   ↓
Label Generation
   ↓
Training Dataset
   ↓
ML Training
   ↓
Backtesting
```

This means the **same dataset powers both ML training and backtesting**.

---

# 2. Step 1: Prepare historical tick data

If you have tick data, it usually looks like:

| timestamp | price | volume | bid | ask |
| --- | --- | --- | --- | --- |
| 09:15:01 | 22100 | 12 | 22099 | 22101 |

But ML works better on **structured intervals**.

Convert ticks into candles.

Example:

```
1 minute OHLCV
```

Result:

| time | open | high | low | close | volume |
| --- | --- | --- | --- | --- | --- |
| 09:15 | 22100 | 22140 | 22090 | 22130 | 5300 |

Python example:

```
df_1m=tick_df.resample("1min").agg({
"price": ["first","max","min","last"],
"volume":"sum"
})
```

---

# 3. Step 2: Feature engineering

This step creates **signals the AI can understand**.

Example features per candle:

| feature | meaning |
| --- | --- |
| RSI | momentum |
| VWAP distance | institutional bias |
| EMA slope | trend |
| volume spike | liquidity |
| OI change | derivatives flow |

Example code:

```
df["rsi"]=ta.rsi(df["close"],length=14)
df["ema20"]=ta.ema(df["close"],length=20)
df["ema50"]=ta.ema(df["close"],length=50)

df["vwap_dist"]= (df["close"]-df["vwap"])/df["vwap"]
```

Now each row represents **the market state at that moment**.

---

# 4. Step 3: Label creation (the most important step)

You must define what **“successful trade” means**.

Example label:

```
Will price move +0.4% in the next 10 minutes?
```

Code example:

```
future_return=df["close"].shift(-10)/df["close"]-1
df["target"]= (future_return>0.004).astype(int)
```

Now each row becomes:

| features | target |
| --- | --- |
| RSI=62 | 1 |
| VWAP_dist=0.2 |  |
| volume_ratio=1.7 |  |

Meaning:

```
This setup led to a profitable trade
```

This is how AI learns.

---

# 5. Step 4: Dataset creation

Final dataset:

| rsi | ema20 | ema50 | vwap_dist | volume_ratio | target |
| --- | --- | --- | --- | --- | --- |
| 62 | 22000 | 21980 | 0.002 | 1.5 | 1 |

Your ML model learns the relationship:

```
features → probability of success
```

---

# 6. Step 5: Model training

Example using XGBoost:

```
importxgboostasxgb

model=xgb.XGBClassifier(
max_depth=6,
learning_rate=0.05,
n_estimators=300
)

model.fit(X_train,y_train)
```

Output:

```
Probability of profitable trade
```

Example prediction:

```
0.67 probability
```

---

# 7. Step 6: Walk-forward backtesting

This is how you avoid **overfitting**.

Instead of training on all data, you simulate real trading.

Example:

```
Train on Jan–Mar
Test on Apr
Train on Feb–Apr
Test on May
```

Pipeline:

```
Train → Test → Move window → Train → Test
```

This mimics real-world conditions.

---

# 8. Step 7: Backtest engine

The system replays the market.

Pseudo code:

```
forcandleinhistorical_data:

compute_features()

prob=model.predict_proba(features)

ifprob>0.6:
simulate_trade()
```

This produces statistics:

| metric | value |
| --- | --- |
| win rate | 63% |
| profit factor | 1.7 |
| max drawdown | 8% |

---

# 9. Step 8: Continuous learning

Once live trading begins:

Every trade gets stored.

| features | outcome |
| --- | --- |
| RSI=61 | win |
| RSI=55 | loss |

Dataset grows.

Weekly retraining:

```
new trades added
↓
retrain model
↓
improved predictions
```

Over months the model becomes **specialized to your strategies**.

---

# 10. Tick data vs candle data for ML

Even if you have tick data, you usually train ML on:

```
1 minute candles
```

Reasons:

1. reduces noise
2. easier feature engineering
3. faster training

Tick data is mostly used for:

```
microstructure analysis
order flow signals
```

---

# 11. Data size estimation

6 months of 1-minute candles:

```
≈ 75 trading days
≈ 375 minutes/day
≈ 28,000 samples
```

That is enough for ML training.

---

# 12. The most important ML rule in trading

Never let ML decide trades alone.

The correct structure:

```
strategy rules
      ↓
ML probability filter
      ↓
risk manager
      ↓
execution
```

This dramatically reduces overfitting.

---

# 13. Example final decision pipeline

Live trading pipeline:

```
new market candle
      ↓
compute indicators
      ↓
generate signal
      ↓
ML evaluates probability
      ↓
if probability > 0.6
      ↓
place trade
```

Decision time:

```
<20 milliseconds
```

---

# 14. What your ML model will eventually learn

After training on 6 months of data, it will learn patterns like:

```
VWAP breakout + volume spike + RSI>60
→ 68% success probability
```

or

```
RSI extreme + low volume
→ 42% probability
```

So it filters bad trades automatically.

---

# 15. One powerful improvement (most retail systems miss)

Instead of labeling **price movement**, label **strategy outcomes**.

Example:

```
Did VWAP breakout strategy hit target before stop?
```

Now ML learns:

```
when this strategy works
```

This dramatically improves real trading performance.

---

✅ So yes, the system **fully incorporates learning from your historical tick data**, and that dataset powers:

- ML training

• strategy validation

• backtesting simulation

• future model improvement

---