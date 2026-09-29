# Rolling CEEMDAN Leakage Audit

## Data

CSI 300 ETF 510300 Garman-Klass variance.

- Sample: 2023-01-03 to 2024-08-30
- Observations: 404
- Training: 279 observations through 2024-02-29
- Test: 125 observations from 2024-03-01

## Methods

A: full-sample CEEMDAN. The decomposition uses the complete series and therefore includes future information.

B: fragment decomposition plus ARIMA. CEEMDAN is fitted only to the 279-day training segment; IMFs are extrapolated with centered and scaled ARIMA(1,1,1), except the first two high-frequency IMFs, which are filled by their training mean.

C: rolling CEEMDAN. For test index t, CEEMDAN is rerun on rv_series[:t] and one-step IMF forecasts are formed with the same scaled ARIMA convention.

All schemes use the same CSV, the same first-k alignment helper, and the same RMSE function.

## Stage 1

Ten rolling decompositions were run.

- IMF count distribution: {7: 10}
- Average decomposition time: 5.72 seconds
- Aligned cube: (10, 7, 288)
- Failures: 0

## Stage 2

The first 50 test days were evaluated for k=1,3,5.

| k | A full-sample | B fragment+ARIMA | C rolling |
|---|---|---|---|
| 1 | 5.713793e-05 | 6.177020e-05 | 6.142699e-05 |
| 3 | 5.275226e-05 | 9.335073e-05 | 7.029100e-05 |
| 5 | 8.105321e-05 | 9.226900e-05 | 1.112079e-04 |

Rolling metadata:

- IMF count distribution: {6: 1, 7: 49}
- Average decomposition time: 5.08 seconds
- Total rolling time: 4.24 minutes

## Diagnostic note

An initial rolling run produced an impossible RMSE near 0.389 for k=3 and k=5. The anomaly was traced to 2024-03-22, where fitting ARIMA directly to very small-scale IMF values caused a numerical blow-up. It was not caused by the variable-IMF alignment or the n_imfs=6 observation on 2024-03-20.

The corrected implementation centers and scales each IMF before fitting ARIMA and restores the physical scale afterward. The saved 50-day run was recomputed without rerunning CEEMDAN.

## Interpretation

At k=1 rolling is very close to fragment+ARIMA and only 7.5% worse than the leaky full-sample decomposition. At k=3 and k=5 rolling is clearly worse than A by 33.2% and 37.2%. Its relation to B is less uniform: 24.7% lower at k=3 and 20.5% higher at k=5.

## Not included

- Stage 3, the full 125-day rolling run, has not been executed.
- Raw OHLC bars are not stored in this repository.
