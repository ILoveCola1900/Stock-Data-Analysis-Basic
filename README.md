# Information Leakage in CEEMDAN Decomposition for Volatility Forecasting

> **Key Finding**: Full-sample CEEMDAN decomposition shows **12.9%–53.7% lower reconstruction error** 
> compared to leakage-free decomposition, with the most severe leakage near extreme market events.

---

## TL;DR

Most papers apply CEEMDAN on the **full sample** before splitting into train/test — 
this leaks future information into the decomposition. 

This project quantifies that leakage using a **model-free reconstruction error metric**, 
avoiding the model-degradation trap that hides the effect in noisy volatility data.

---

## Key Results

Reconstruction error (RMSE) on the 125-day test set, using the first *k* IMFs:

| k (IMFs used) | Full-sample (leaky) | Train-only (clean) | Leakage |
|:---:|:---:|:---:|:---:|
| 1 | 2.49e-4 | 2.86e-4 | **+12.9%** |
| 3 | 1.82e-4 | 3.03e-4 | **+39.8%** |
| 5 | 1.26e-4 | 2.73e-4 | **+53.7%** |

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

## Project Structure
.
├── README.md
├── requirements.txt
├── data/
│ └── rv_series.csv # GK volatility series (404 days)
├── notebooks/
│ ├── 01_data_preparation.ipynb # Download + GK volatility
│ ├── 02_ceemdan_decomposition.ipynb # Decomposition + boundary effects
│ ├── 03_information_leakage.ipynb # Core leakage experiment
│ └── 04_rolling_decomposition.ipynb # Rolling decomposition (WIP)
├── src/
│ ├── volatility.py # Garman-Klass estimator
│ ├── decomposition.py # CEEMDAN wrapper
│ └── evaluation.py # Reconstruction error metrics
├── results/
│ ├── figures/ # Key plots
│ └── tables/ # Comparison tables
└── report/
└── REPORT.md # Technical report

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
| C | Rolling (WIP) | No |

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
@misc{[yourname]2024ceemdan,
  title={Information Leakage in CEEMDAN Decomposition},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[yourname]/[repo]}
}

## Acknowledgments
Thanks to Prof. Mo Jixian for the PIMSE paper that motivated this investigation 
into validation set construction for time series. 

Author: [Liuye]
Contact: [Your Email]
