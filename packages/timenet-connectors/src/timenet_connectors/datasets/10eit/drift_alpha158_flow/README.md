# DRIFT Alpha158 stock-flow records

This restricted connector reads a user-supplied OHLCV and task bundle. The code can be distributed independently of the research price data. With access to the DRIFT research source, run the four-seed history export, the v2 task builder, and `data/prompt_data/export_drift_v2_timef.py` in the DRIFT workspace. Set `DRIFT_ALPHA158_BUNDLE` to the resulting `drift-alpha158-flow-input` directory.

The connector verifies the two source hashes, computes all 157 raw factors using pandas and NumPy, and stores 20 trading sessions per signal. A target and its comparison stock have complete factor panels. The five upstream names have daily returns and five price/volume factors. Use the record's `factor_order` and `trading_dates` annotations to reconstruct a panel. Missing factor values retain their original NaN semantics.

```bash
timenet-build build 10eit/drift-alpha158-flow --no-isolation --local-registry .timenet-registry
```

The release covers 30 core stocks in 11 sectors over 2018–2025. Each of the 240 observations has four AnswerTasks. There are 600 training, 120 validation and 240 test tasks. Future outcomes occur only in task supervision.

A team-hosted HTTPS directory containing the same two pinned files can be used through `DRIFT_ALPHA158_BUNDLE_URL`. The original local-directory setting has priority. For immediate anonymous access, `10eit/drift-alpha158-flow-public` builds the matched public-price subset directly from Kaggle's version 613 snapshot.
