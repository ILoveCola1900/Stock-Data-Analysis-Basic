# Information Leakage in CEEMDAN Decomposition for Volatility Forecasting

> **Key Finding**: Full-sample CEEMDAN decomposition shows **7.5%–37.2% lower reconstruction error** 
> compared to leakage-free decomposition, with the most severe leakage near extreme market events.

---

## TL;DR

Most papers apply CEEMDAN on the **full sample** before splitting into train/test — 
this leaks future information into the decomposition. 

This project quantifies that leakage using a **model-free reconstruction error metric**, 
avoiding the model-degradation trap that hides the effect in noisy volatility data.

---

## Key Results

Reconstruction error (RMSE) on the 50-day test set:

| k | Full-sample (leaky) | Rolling (strict) | Leakage |
|---|:---:|:---:|:---:|
| 1 | 5.71e-5 | 6.14e-5 | **+7.5%** |
| 3 | 5.28e-5 | 7.03e-5 | **+33.2%** |
| 5 | 8.11e-5 | 1.11e-4 | **+37.2%** |

*Leakage = how much "better" the leaky decomposition appears. This is a false advantage.*

![Leakage Comparison](results/figures/reconstruction_comparison_k3.png)

**Interpretation**: Leakage grows with the number of low-frequency IMFs used, 
and concentrates near extreme events (e.g., the September 2024 market spike).

---

## Boundary Effects Discovery

Before the leakage experiment, we observed **severe boundary effects** in CEEMDAN:

| IMF | Boundary Energy (initial) | After Mirror Extension | After Truncation |
|:---:|:---:|:---:|:---:|
| IMF 3 | 93.1% | 95.1% | **14.9%** |
| IMF 4 | 84.8% | 90.3% | **6.7%** |

**Root cause**: The right endpoint value (9.7× mean) driven by the Sept 2024 spike.

**Insight**: Mirror extension does **not** fix endpoint-value-driven boundary effects. 
Truncation does.

---

## Repository Layout

```text
.
├── data/           Frozen 404-day GK volatility series (rv_series.csv)
├── notebooks/      Four reproducible notebooks (01–04)
├── src/            Modular Python: volatility, decomposition, evaluation
├── results/        Key figures (figures/) and tables (tables/)
└── report/         Technical report (REPORT.md)
```

---

## Method Summary

### Data

- **Asset**: CSI 300 ETF (510300.SS)
- **Period**: Jan 2023 – Aug 2024 (404 trading days)
- **Volatility**: Garman-Klass estimator (uses OHLC, more efficient than close-to-close)

### Experimental Design

| Group | Decomposition | Leakage |
|:---:|---|:---:|
| A | Full sample (404 days) | Yes |
| B | Train-only (279 days) + ARIMA extrapolation | No |
| C | Rolling CEEMDAN (50-day tested) | No |

### Why Reconstruction Error, Not Prediction Accuracy?

Initial experiments with Ridge and XGBoost **collapsed to constant predictors** — 
volatility is too noisy for one-step-ahead prediction. 

Reconstruction error isolates **decomposition quality** from **model limitations**, 
making the leakage measurable.

---

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/03_information_leakage.ipynb
```
## Status

- √ Day 1: Data preparation + boundary effects discovery
- √ Day 2: Information leakage quantification
- √ Day 3: Rolling decomposition (50-day validation)
- × Day 4: Full 125-day run (in progress)
  
## References
Mo, J., et al. (2024). Predictive analysis of the sale-and-purchase shipping market: A PIMSE approach. Transportation Research Part E.

Lin, Y., et al. (2021). Forecasting stock index price using the CEEMDAN-LSTM model.

Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities. Journal of Business.

## Background
This project started as a coursework assignment on financial data analysis
and evolved into a methodological investigation of information leakage
in multi-scale decomposition. All experiments, code, and analysis are reproducible
from the notebooks above.

## Citation
If you find this analysis useful, please cite:
@misc{liuye2024ceemdan,
  title={Information Leakage in CEEMDAN Decomposition},
  author={Liuye},
  year={2024},
  url={https://github.com/ILoveCola1900/Stock-Data-Analysis-Basic}
}

## Acknowledgments
Thanks to Prof. Mo Jixian for the PIMSE paper that motivated this investigation 
into validation set construction for time series. 

Author: [Liuye]
Contact: [liuye1900@ruc.edu.cn]
