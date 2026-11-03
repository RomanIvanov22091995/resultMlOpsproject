# Data drift report

- Scenario: **no drift (holdout sample)**, samples: 1000, mode: api
- Score PSI: **0.0075 (stable)**
- Features with significant drift: none
- Features with moderate drift: none
- ROC-AUC on new data: 0.7567 (reference test ROC-AUC: 0.7793)
- Mean PD: reference 0.2219 -> current 0.2129
- API latency: p50 67.1 ms, p95 74.0 ms

PSI thresholds: < 0.10 stable, 0.10-0.25 moderate, >= 0.25 significant.

| feature | type | PSI | status | reference mean | current mean |
|---|---|---|---|---|---|
| BILL_AMT1 | numeric | 0.0178 | stable | 51270.569 | 50261.016 |
| PAY_0 | categorical | 0.0161 | stable | -0.012 | -0.066 |
| PAY_4 | categorical | 0.0148 | stable | -0.219 | -0.246 |
| PAY_3 | categorical | 0.0134 | stable | -0.163 | -0.194 |
| BILL_AMT2 | numeric | 0.0117 | stable | 49292.17 | 48922.95 |
| PAY_2 | categorical | 0.0113 | stable | -0.128 | -0.193 |
| EDUCATION | categorical | 0.0107 | stable | 1.851 | 1.886 |
| LIMIT_BAL | numeric | 0.0096 | stable | 167272.638 | 167730.0 |
| PAY_5 | categorical | 0.0093 | stable | -0.266 | -0.307 |
| BILL_AMT3 | numeric | 0.0092 | stable | 47095.933 | 46878.814 |
| PAY_6 | categorical | 0.0092 | stable | -0.293 | -0.317 |
| PAY_AMT4 | numeric | 0.0087 | stable | 4807.29 | 3995.022 |
| BILL_AMT4 | numeric | 0.0086 | stable | 43378.61 | 42961.343 |
| BILL_AMT6 | numeric | 0.0084 | stable | 38899.466 | 38383.992 |
| BILL_AMT5 | numeric | 0.0078 | stable | 40381.773 | 39212.833 |
| PAY_AMT3 | numeric | 0.0076 | stable | 5224.316 | 4596.009 |
| PAY_AMT2 | numeric | 0.0062 | stable | 5933.725 | 5366.954 |
| AGE | numeric | 0.0059 | stable | 35.462 | 35.673 |
| PAY_AMT1 | numeric | 0.0050 | stable | 5652.104 | 5626.282 |
| PAY_AMT5 | numeric | 0.0047 | stable | 4791.897 | 5218.23 |
| PAY_AMT6 | numeric | 0.0036 | stable | 5264.869 | 5492.062 |
| MARRIAGE | categorical | 0.0016 | stable | 1.553 | 1.561 |
| SEX | categorical | 0.0011 | stable | 1.602 | 1.618 |
