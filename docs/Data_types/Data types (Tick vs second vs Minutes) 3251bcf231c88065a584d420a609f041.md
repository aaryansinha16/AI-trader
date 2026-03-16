# Data types (Tick vs second vs Minutes)

# 1. Tick vs second vs minute data

Market data comes in three main granularities.

### True tick data

Every trade update from the exchange.

Example:

| time | price |
| --- | --- |
| 09:15:00.231 | 22100 |
| 09:15:00.448 | 22101 |
| 09:15:00.891 | 22100 |

This is **sub-second data**.

Pros

• full market microstructure

• best for order flow analysis

Cons

• extremely heavy data

• harder to process

---

### Second-level data

Aggregated every second.

Example:

| time | open | high | low | close |
| --- | --- | --- | --- | --- |
| 09:15:01 | 22100 | 22102 | 22099 | 22101 |

Pros

• good detail

• manageable size

Cons

• still heavy for long histories

---

### Minute-level data

Aggregated per minute.

Example:

| time | open | high | low | close |
| --- | --- | --- | --- | --- |
| 09:15 | 22100 | 22140 | 22090 | 22130 |

Pros

• very easy to train ML

• widely used

Cons

• loses microstructure information

---

# 2. What professional systems usually do

Most quant systems store:

```
tick data
↓
aggregate into multiple timeframes
```

Typical derived datasets:

```
1 second candles
5 second candles
1 minute candles
5 minute candles
```

The ML models usually train on:

```
1 minute data
```

But **signals can use tick or second level information**.

---

# 3. What YOU should do

Best setup for your project:

### Store

```
tick data
```

### Train ML on

```
1 minute candles
```

### Use second-level features for

```
volume spikes
order flow signals
VWAP precision
```

This gives the best balance.

---

# 4. Why minute data works best for ML

Financial markets are noisy.

Second-level data often causes:

```
overfitting
```

Example:

If you train on second data, the model might learn random noise patterns.

Minute data smooths noise and reveals **real market structure**.

---

# 5. Data size comparison

Let’s estimate 6 months of data.

### Minute data

```
~375 candles/day
~75 trading days
≈ 28k rows
```

Very small.

---

### Second data

```
~22500 seconds/day
~75 days
≈ 1.6 million rows
```

Still manageable.

---

### Tick data

```
millions of ticks/day
```

6 months:

```
10–50 million rows
```

Requires good storage.

---

# 6. What TrueData provides

TrueData provides **real-time tick streaming**.

That means:

```
trade ticks
bid/ask
market depth
```

But there is an important limitation.

### Historical tick data

TrueData usually provides:

```
historical minute data
```

Tick history is limited and often expensive.

Most users record tick data themselves once subscribed.

Typical workflow:

```
subscribe to tick feed
↓
store ticks locally
↓
build historical dataset
```

---

# 7. Ideal data architecture for your system

```
Tick Feed (WebSocket)
        ↓
Tick Database
        ↓
Aggregation Engine
        ↓
1s candles
1m candles
5m candles
```

This way you can generate **any timeframe later**.

---

# 8. Example aggregation pipeline

Python example:

```
tick_df=load_ticks()

second_df=tick_df.resample("1S").agg({
"price":["first","max","min","last"],
"volume":"sum"
})

minute_df=tick_df.resample("1T").agg({
"price":["first","max","min","last"],
"volume":"sum"
})
```

Now you have **multi-resolution datasets**.

---

# 9. Best dataset combination for ML trading

Recommended training dataset:

```
1 minute OHLCV
+
second-level volume signals
+
options OI changes
+
VWAP metrics
```

Typical feature count:

```
50–120 features
```

That’s what quant models usually use.

---

# 10. Storage recommendation

For your laptop:

Best database:

```
TimescaleDB
```

or

```
ClickHouse
```

These handle millions of rows easily.

---

# 11. One extremely important detail

If you collect tick data yourself, **start collecting immediately**.

Even before your system is ready.

Because:

```
data history = trading edge
```

The earlier you start collecting, the better your ML model becomes.

---

# 12. Realistic data strategy for your system

Ideal setup:

```
tick data → stored
↓
1s candles → signals
↓
1m candles → ML training
↓
5m candles → regime detection
```

This multi-resolution view is very powerful.

---

# 13. One more thing that improves models dramatically

Most retail ML trading systems fail because they ignore **cross-asset signals**.

For Nifty trading you should also include:

```
BankNifty
India VIX
USDINR
SGX Nifty
```

These provide **context for market movement**.

[Elaborated (Challenges and solutions)](https://www.notion.so/Elaborated-Challenges-and-solutions-3251bcf231c880df9e54f07b125e94f0?pvs=21)