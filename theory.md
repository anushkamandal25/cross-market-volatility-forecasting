# Finance Theory & Formulas

Detailed reference for all concepts and formulas used in this project.

---

## 1. Log Returns

**Formula used in notebook:**
```python
log_return = np.log(close / close.shift(1)) * 100
```

Mathematically:
```
r_t = ln(P_t / P_{t-1}) × 100
```

Preferred over simple percentage returns because:
- Time-additive: weekly return = sum of daily log returns
- Approximately normally distributed
- Symmetric: a 50% loss followed by a 100% gain nets to zero

> Note: the API pipeline (`model.py` → `wrangle_data`) uses simple percentage returns
> (`pct_change * 100`) for the GarchModel class. The notebook analysis uses log returns.

---

## 2. Annualised Return & Volatility

**Formula used:**
```python
TRADING_DAYS = 252

ann_return = r.mean() * TRADING_DAYS
ann_vol    = r.std()  * np.sqrt(TRADING_DAYS)
```

252 = number of trading days in a year (exchanges are closed on weekends and holidays).
Variance scales linearly with time, so standard deviation scales with √time — not time itself.

**Results:**
- INFY: annualised return 5.15%, annualised volatility 28.22%
- TSLA: annualised return 27.23%, annualised volatility 56.40%

---

## 3. Rolling Volatility

Used for visualisation only — not for forecasting.

**Formula used:**
```python
ROLLING_WINDOW = 30

rolling_vol = log_return.rolling(window=30).std() * np.sqrt(252)
```

A backward-looking measure over a fixed 30-day window, annualised.
Unlike GARCH, it does not model the process or produce forecasts.

---

## 4. Volatility Clustering

Large price moves tend to be followed by large moves, and small moves by small moves —
regardless of direction.

Visible in the ACF/PACF plots of **squared returns** (`log_return ** 2`).
Significant autocorrelation in squared returns confirms volatility is not random
but clustered over time. This is the core motivation for using a GARCH model.

---

## 5. ARCH Effect

AutoRegressive Conditional Heteroskedasticity — today's variance depends on
past squared return shocks.

Tested visually using ACF/PACF of squared log returns with 40 lags at 95% confidence (`alpha=0.05`).
Both INFY and TSLA show significant autocorrelation, confirming the ARCH effect is present.

---

## 6. GARCH(p,q) Model

Generalised ARCH — extends ARCH by including past variances as well as past shocks.

**General variance equation:**
```
σ²_t = ω + α₁ε²_{t-1} + ... + αₚε²_{t-p} + β₁σ²_{t-1} + ... + βqσ²_{t-q}
```

**GARCH(1,1) — final model used for both INFY and TSLA:**
```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

**Fitted using:**
```python
arch_model(returns, p=1, q=1, rescale=False).fit(disp=0)
```

Parameters from final fitted models:

| Parameter | INFY | TSLA |
|---|---|---|
| μ (mu) | 0.0492 | 0.0789 |
| ω (omega) | 0.5925 | 0.1001 |
| α (alpha[1]) | 0.1448 | 0.0330 |
| β (beta[1]) | 0.2517 | 0.9593 |

---

## 7. GARCH Order Selection

Tested all combinations of p, q ∈ {1, 2, 3} using AIC.

**AIC selected:**
- INFY: GARCH(1,3) — overridden to GARCH(1,1) because beta[2] ≈ 0 (p-value = 1.0),
  beta[1] and beta[3] also insignificant. AIC difference was only ~23 points.
  Parsimony principle applied.
- TSLA: GARCH(1,1) selected naturally, all coefficients significant.

---

## 8. Volatility Persistence

```
Persistence = α + β
```

| | INFY | TSLA |
|---|---|---|
| α + β | 0.3965 | 0.9923 |
| Interpretation | Low — shocks dissipate quickly | Near-integrated — shocks linger very long |

- TSLA at 0.9923 is close to IGARCH (integrated GARCH = 1.0), meaning volatility shocks
  almost never fully decay — visible in the COVID crash spike lasting months
- INFY at 0.3965 means shocks fade quickly, consistent with sharp event-driven spikes
  (earnings surprises) that resolve within days

---

## 9. Conditional Volatility

The GARCH model's time-varying estimate of volatility at each point in history:

```
σ_t = √σ²_t
```

Accessed via:
```python
model.conditional_volatility
```

Unlike rolling volatility (backward-looking average), conditional volatility is
updated dynamically at each step based on the fitted model parameters.

---

## 10. Multi-Step Volatility Forecast

**Formula used:**
```python
forecast = model.forecast(horizon=5, reindex=False).variance ** 0.5
```

For GARCH(1,1), the h-step ahead forecast reverts toward the long-run variance:
```
σ²_{t+h} = ω/(1-α-β) + (α+β)^h × (σ²_t - ω/(1-α-β))
```

- High persistence (α+β ≈ 1, TSLA) → forecast stays elevated, trends slowly upward
- Low persistence (α+β = 0.40, INFY) → forecast reverts to long-run mean quickly,
  producing the fluctuating pattern visible in the 5-day forecast plot

**5-day forecast results:**

| Date | INFY vol % | TSLA vol % |
|---|---|---|
| 2026-06-08 | 2.1438 | 2.9471 |
| 2026-06-09 | 2.3210 | 2.9527 |
| 2026-06-10 | 2.2850 | 2.9583 |
| 2026-06-11 | 2.1375 | 2.9638 |
| 2026-06-12 | 2.1537 | 2.9693 |

---

## 11. AIC & BIC — Model Selection

Used to compare GARCH(p,q) specifications. Both penalise complexity.

**Akaike Information Criterion:**
```
AIC = 2k - 2ln(L)
```

**Bayesian Information Criterion:**
```
BIC = k × ln(n) - 2ln(L)
```

Where k = number of parameters, L = model likelihood, n = observations.
Lower = better. BIC penalises extra parameters more heavily than AIC.

**Results for final GARCH(1,1) models:**
- INFY: AIC = 11707.49, BIC = 11743.53
- TSLA: AIC = 15710.63, BIC = 15734.65

---

## 12. Value at Risk (VaR) — 95%

Answers: *"What is the maximum loss I can expect on 95% of trading days?"*

**Formula used:**
```python
Z_95 = 1.645

infy_mean = infy_returns.mean()
VaR = -(infy_mean - Z_95 * sigma_forecast)
```

Where:
- `infy_mean` = mean daily log return over full history
- `1.645` = z-score for 95% confidence, one-tailed normal distribution
- `sigma_forecast` = GARCH volatility forecast for that day

**Results:**

| Date | INFY VaR | TSLA VaR |
|---|---|---|
| 2026-06-08 | 3.5061% | 4.7399% |
| 2026-06-09 | 3.7976% | 4.7491% |
| 2026-06-10 | 3.7383% | 4.7583% |
| 2026-06-11 | 3.4957% | 4.7674% |
| 2026-06-12 | 3.5224% | 4.7764% |

- INFY avg: on 95% of days, daily loss will not exceed **3.61%**
- TSLA avg: on 95% of days, daily loss will not exceed **4.76%**