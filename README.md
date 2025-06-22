# Vector-Autoregression-VAR-Analysis
Vector Autoregression (VAR) Analysis of U.S. Monetary Policy

## Vector Autoregression (VAR) Analysis of U.S. Monetary Policy
*Advanced Quantitative Methods and Machine Learning in Finance*

## **Business Question**
How can Federal Reserve policymakers and economic analysts leverage Vector Autoregression (VAR) modeling to understand the complex interactions between unemployment, inflation, and monetary policy decisions, and optimize policy interventions to achieve dual mandate objectives of price stability and full employment?

## **Business Case**
In the current macroeconomic environment characterized by complex interactions between monetary policy, labor markets, and price dynamics, traditional single-equation models fail to capture the simultaneous, bidirectional relationships among key economic variables. The Federal Reserve requires sophisticated analytical frameworks to understand policy transmission mechanisms, forecast economic outcomes, and formulate data-driven monetary policy decisions. VAR modeling provides a comprehensive approach to analyze these dynamic interdependencies, enabling policymakers to assess how changes in interest rates propagate through the economy, anticipate policy consequences, and optimize intervention timing. This analysis is crucial for maintaining macroeconomic stability, managing inflation expectations, and supporting sustainable economic growth while avoiding unintended policy consequences that could destabilize financial markets or employment conditions.

## **Analytics Question**
How can the systematic application of Vector Autoregression methodology, combined with comprehensive stationarity testing, optimal lag selection, and Granger causality analysis, help economic analysts develop robust empirical models that capture the temporal dynamics between unemployment, inflation, and federal funds rate while providing reliable forecasts and policy guidance for Federal Reserve decision-making?

## **Outcome Variables of Interest**
The analysis focuses on three primary macroeconomic indicators central to Federal Reserve policy:
1. **Unemployment Rate (UNRATE)**: Quarterly average representing labor market conditions
2. **Inflation Rate**: Year-over-year percentage change in Consumer Price Index (CPI) 
3. **Federal Funds Rate (FEDFUNDS)**: Quarterly average of the federal funds target rate

## **Key Predictors**
The VAR framework employs endogenous predictors where each variable serves as both dependent and independent:
- **Lagged Unemployment**: Past unemployment rates affecting current economic conditions
- **Lagged Inflation Changes**: Historical inflation dynamics influencing current price trends
- **Lagged Federal Funds Rate Changes**: Previous monetary policy decisions impacting current economic variables
- **Cross-Variable Interactions**: Bidirectional relationships among all three variables across multiple time lags
- **Temporal Dependencies**: Complex lag structures capturing policy transmission mechanisms

## **Dataset Description**
The analysis utilizes quarterly U.S. macroeconomic data sourced from the Federal Reserve Economic Data (FRED) database, spanning five decades of economic cycles to capture diverse policy regimes and economic conditions.

**Dataset Specifications:**
- **Temporal Coverage**: 1971 Q1 to 2019 Q4 (195 quarterly observations)
- **Frequency**: Quarterly data to align with business cycle analysis
- **Variable Sources**: FRED database with official government statistics
- **Data Period Rationale**: Excludes COVID-19 period to avoid pandemic-related distortions
- **Economic Coverage**: Multiple business cycles, financial crises, and policy regime changes

**Primary Data Sources:**
- **Unemployment Rate**: [UNRATE - FRED](https://fred.stlouisfed.org/series/UNRATE)
- **Consumer Price Index**: [CPIAUCSL - FRED](https://fred.stlouisfed.org/series/CPIAUCSL) 
- **Federal Funds Rate**: [FEDFUNDS - FRED](https://fred.stlouisfed.org/series/FEDFUNDS)

**Data Transformations:**
- **Unemployment**: Quarterly averages of monthly data
- **Inflation**: Year-over-year percentage change calculated from CPI levels
- **Federal Funds Rate**: Quarterly averages of monthly target rates

## **Stationarity Analysis & Data Preprocessing**

### **Augmented Dickey-Fuller Test Results**

![image](https://github.com/user-attachments/assets/ed9a8893-d414-4cea-b6db-8c3002b6ef35)


**Stationarity Assessment:**
| **Variable** | **ADF Statistic** | **p-value** | **Critical Value (5%)** | **Stationarity Status** |
|-------------|------------------|-------------|------------------------|------------------------|
| **Unemployment** | -2.8857 | 0.0470 | -2.8772 | **Stationary** |
| **Inflation** | -1.6939 | 0.4343 | -2.8775 | **Non-stationary** |
| **Federal Funds Rate** | -1.6217 | 0.4718 | -2.8770 | **Non-stationary** |

**Required Transformations:**
- **Unemployment Rate**: No transformation required (I(0) process)
- **Inflation Rate**: First differencing applied (I(1) → I(0))
- **Federal Funds Rate**: First differencing applied (I(1) → I(0))

**Post-Transformation Stationarity Verification:**
| **Transformed Variable** | **ADF Statistic** | **p-value** | **Result** |
|-------------------------|------------------|-------------|------------|
| **Δ Inflation** | -5.8750 | < 0.0001 | **Stationary** |
| **Δ Federal Funds Rate** | -5.9316 | < 0.0001 | **Stationary** |

### **Economic Interpretation of Transformations**
The stationarity analysis reveals important characteristics of these macroeconomic series. Unemployment demonstrates stationarity in levels, suggesting mean-reverting behavior consistent with natural rate theories. In contrast, inflation and federal funds rates require differencing, indicating these variables follow random walk processes with drift, reflecting persistent changes in price levels and monetary policy regimes over time.

## **VAR Model Specification & Selection**

### **Optimal Lag Selection**

**Information Criteria Comparison:**
| **Lags** | **AIC** | **BIC** | **FPE** | **HQIC** | **Selection** |
|----------|---------|---------|---------|----------|---------------|
| **1** | -3.394 | -3.183 | 0.03358 | -3.309 | |
| **2** | -4.017 | **-3.649*** | 0.01801 | -3.868 | **BIC Optimal** |
| **5** | -4.312 | -3.470 | 0.01342 | **-3.971*** | **HQIC Optimal** |
| **8** | **-4.467*** | -3.152 | **0.01154*** | -3.934 | **AIC/FPE Optimal** |

**Selected Model: VAR(8)**
- **Selection Criterion**: AIC minimization for comprehensive dynamics capture
- **Rationale**: Economic theory suggests monetary policy transmission occurs with substantial lags
- **Trade-off**: Higher complexity for improved dynamic representation

### **VAR(8) Model Estimation Results**

![image](https://github.com/user-attachments/assets/73181df5-9e8d-4317-a71a-12151ba2efe3)


![image](https://github.com/user-attachments/assets/1f50759d-aba8-4ac5-8066-0eee3edbf979)


![image](https://github.com/user-attachments/assets/f1aed375-b84b-4973-b5ff-284088016796)


![image](https://github.com/user-attachments/assets/14388506-ec70-4a25-bffd-50ddaf226b57)


**Model Performance Statistics:**
- **Number of Observations**: 187 quarters
- **Log Likelihood**: -308.447
- **AIC**: -4.41259 (optimal among tested specifications)
- **BIC**: -3.11669
- **Determinant of Residual Covariance**: 0.00836084

## **VAR Model Interpretation & Economic Insights**

### **Unemployment Equation Dynamics**
The unemployment equation reveals sophisticated temporal patterns with strong autoregressive characteristics. The first-order coefficient (1.567) indicates substantial persistence, while the second-order coefficient (-0.464) suggests eventual mean reversion. **Monetary policy transmission** manifests with significant delays at lags 6 and 8 (coefficients 0.065 and 0.058), confirming that contractionary policy impacts labor markets approximately 1.5-2 years after implementation.

**Key Findings:**
- **Policy Transmission Lag**: 6-8 quarters for full unemployment impact
- **Persistence**: High autoregressive behavior with eventual mean reversion
- **Inflation Independence**: No significant direct effects from inflation changes

### **Inflation Dynamics Analysis**
Inflation changes exhibit complex cyclical relationships with unemployment, characterized by alternating significant coefficients: negative first-order (-0.702), positive second-order (1.395), and negative fourth-order (-0.816) effects. **Federal funds rate changes** demonstrate consistent positive effects across early lags (0.120, 0.122, 0.174), potentially capturing the endogenous policy response rather than exogenous effects.

**Key Findings:**
- **Phillips Curve Complexity**: Non-linear unemployment-inflation relationships
- **Monetary Policy Response**: Immediate positive correlation with rate changes
- **Autoregressive Structure**: Complex lag patterns indicating inflation persistence

### **Federal Funds Rate Policy Reaction Function**
The monetary policy equation indicates substantial responsiveness to unemployment fluctuations with alternating patterns: negative effects at lags 1 (-1.043) and 6 (-1.207), positive effects at lags 2 (1.326) and 4 (1.521). This suggests **nuanced policy reactions** to labor market developments over different time horizons.

**Key Findings:**
- **Employment Sensitivity**: Strong reaction to unemployment changes across multiple lags
- **Policy Inertia**: Complex autoregressive behavior in rate adjustments
- **Limited Inflation Response**: Contrary to Taylor Rule expectations

### **Cross-Variable Relationships**
**Residual Correlation Matrix:**
- **Unemployment-Federal Funds**: -0.388 (moderate negative correlation)
- **Inflation-Federal Funds**: +0.271 (positive correlation)
- **Unemployment-Inflation**: -0.152 (weak negative correlation)

## **Model Validation & Diagnostic Testing**

### **Residual Analysis**

![image](https://github.com/user-attachments/assets/1dc7b271-50fd-4337-9cdb-573b19db4105)


![image](https://github.com/user-attachments/assets/3b602d6c-0381-4f11-b634-c20d576044f8)


![image](https://github.com/user-attachments/assets/21456125-9db0-410d-aefc-ad113a2806a3)


![image](https://github.com/user-attachments/assets/f64b2b88-b8dc-4ee0-9123-cba095b92167)



**Time Series Properties:**
- **Zero Mean**: All residual series oscillate around zero ✓
- **Constant Variance**: Generally satisfied with some heteroscedasticity in pre-1985 period
- **Serial Independence**: PACF analysis shows minimal significant autocorrelation

**Structural Break Evidence:**
- **Federal Funds Rate**: Marked volatility reduction post-1985 (Volcker-Greenspan transition)
- **Economic Crises**: Volatility spikes during 1975, 1980, and 2008-2009 recessions
- **Model Adequacy**: Successfully captures systematic patterns despite occasional outliers

### **PACF Analysis Results**
- **Unemployment Residuals**: Minor exceedances at lags 8 and 16 (acceptable)
- **Federal Funds Residuals**: Slight significance around lag 19 (minimal concern)
- **Overall Assessment**: VAR specification adequately addresses serial correlation

## **Forecasting Performance & Model Evaluation**

### **Out-of-Sample Forecast Analysis**

**Training-Test Split:**
- **Training Period**: 1971 Q1 - 2017 Q4 (187 quarters)
- **Test Period**: 2018 Q1 - 2019 Q4 (8 quarters)
- **Forecast Horizon**: 8 quarters ahead

![image](https://github.com/user-attachments/assets/20ebbbe7-f233-4115-becf-f462e60b351f)



**Forecast Performance by Variable:**

#### **Unemployment Rate Forecasting**
- **Model Prediction**: Steady increase to ~5.0%
- **Actual Outcome**: Continued decline to ~3.7%
- **Interpretation**: Model anticipated cyclical turning point that didn't materialize

#### **Inflation Rate Forecasting**
- **Model Prediction**: Stable around 2.0%
- **Actual Outcome**: Maintained near 2.0% target
- **Interpretation**: Successful capture of price stability dynamics

#### **Federal Funds Rate Forecasting**
- **Model Prediction**: Decline to ~0.5%
- **Actual Outcome**: Initial increase to 2.5%, then gradual decline
- **Interpretation**: Model missed Federal Reserve's policy tightening cycle

### **Quantitative Forecast Accuracy**

**Performance Metrics:**
| **Variable** | **MAPE (%)** | **RMSE** | **Interpretation** |
|-------------|--------------|----------|-------------------|
| **Unemployment** | 24.61 | 1.03 | Poor predictive accuracy |
| **Inflation** | 17.38 | 0.40 | Good predictive performance |
| **Federal Funds Rate** | 52.48 | 1.22 | Significant forecast errors |

**Key Insights:**
- **Best Performance**: Inflation forecasting with acceptable precision
- **Moderate Performance**: Unemployment with substantial directional errors
- **Poor Performance**: Federal funds rate reflecting policy unpredictability

## **Granger Causality Analysis**

### **Temporal Ordering Identification**

![image](https://github.com/user-attachments/assets/759f1714-bd5a-4341-b4a6-24ce2dfe563c)


**Significant Causal Relationships (p < 0.05):**
| **Cause → Effect** | **p-value** | **Optimal Lag** | **Economic Interpretation** |
|-------------------|-------------|-----------------|---------------------------|
| **Fed Funds → Unemployment** | 0.0002 | 1 quarter | **Rapid monetary policy transmission** |
| **Unemployment → Inflation** | 0.0000 | 5 quarters | **Phillips Curve dynamics** |
| **Fed Funds → Inflation** | 0.0001 | 4 quarters | **Monetary policy effectiveness** |
| **Unemployment → Fed Funds** | 0.0000 | 5 quarters | **Policy reaction to labor markets** |
| **Inflation → Fed Funds** | 0.0293 | 2 quarters | **Taylor Rule behavior** |

**Non-Significant Relationship:**
- **Inflation → Unemployment**: p-value = 0.2024 (No direct causality)

### **Policy Transmission Mechanism**
The Granger causality results reveal a **complex cyclical system**:
1. **Monetary Policy** → **Unemployment** (1 quarter lag)
2. **Unemployment** → **Inflation** (5 quarter lag)  
3. **Inflation** → **Federal Funds Rate** (2 quarter lag)
4. **Federal Funds Rate** → **Inflation** (4 quarter lag)

This empirical evidence supports a sophisticated understanding of macroeconomic dynamics beyond simple unidirectional relationships.

## **Implementation Guide**

### **Technical Requirements**
```python
# Core packages for VAR analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from pandas_datareader import data
```

### **VAR Implementation Workflow**

**Step 1: Data Collection and Preprocessing**
```python
# Download data from FRED
unemployment = data.DataReader('UNRATE', 'fred', start='1970-01-01', end='2019-12-31')
cpi = data.DataReader('CPIAUCSL', 'fred', start='1970-01-01', end='2019-12-31')
fed_funds = data.DataReader('FEDFUNDS', 'fred', start='1970-01-01', end='2019-12-31')

# Calculate inflation and convert to quarterly frequency
cpi['inflation'] = cpi['CPIAUCSL'].pct_change(12) * 100
unemployment_q = unemployment.resample('QS').mean()
inflation_q = cpi.resample('QS').mean()
fed_funds_q = fed_funds.resample('QS').mean()
```

**Step 2: Stationarity Testing and Transformation**
```python
# Test stationarity
def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"{name}: ADF Statistic = {result[0]:.4f}, p-value = {result[1]:.4f}")
    return result[1] <= 0.05

# Apply necessary transformations
if not check_stationarity(df['inflation'], 'Inflation'):
    df['d_inflation'] = df['inflation'].diff()

if not check_stationarity(df['fed_funds'], 'Federal Funds'):
    df['d_fed_funds'] = df['fed_funds'].diff()
```

**Step 3: VAR Model Selection and Estimation**
```python
# Select optimal lag order
model = VAR(var_data)
order_selection = model.select_order(maxlags=12)
optimal_lag = order_selection.aic

# Estimate VAR model
var_model = model.fit(maxlags=optimal_lag)
print(var_model.summary())
```

**Step 4: Granger Causality Testing**
```python
# Test causality relationships
for cause in variables:
    for effect in variables:
        if cause != effect:
            test_df = pd.DataFrame({effect: var_data[effect], cause: var_data[cause]})
            result = grangercausalitytests(test_df, maxlag=optimal_lag, verbose=False)
```

**Step 5: Forecasting and Evaluation**
```python
# Split data and generate forecasts
train_data = var_data.iloc[:-8]
var_model_train = VAR(train_data).fit(maxlags=optimal_lag)
forecasts = var_model_train.forecast(train_data.iloc[-optimal_lag:].values, 8)

# Calculate forecast accuracy
def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100
```

## **Policy Recommendations & Economic Implications**

### **Federal Reserve Policy Strategy**

**Short-Term Recommendations:**
1. **Graduated Rate Adjustments**: Implement measured interest rate changes recognizing 1-quarter unemployment transmission lag
2. **Inflation Targeting**: Leverage strong predictive accuracy for inflation (17.38% MAPE) in policy formulation
3. **Forward Guidance**: Provide clear policy communication given 2-quarter inflation-to-policy feedback loop

**Medium-Term Strategy:**
1. **Labor Market Monitoring**: Close surveillance of unemployment given 5-quarter lag to inflation impact
2. **Dual Mandate Balance**: Weight inflation indicators more heavily given superior forecast reliability
3. **Adaptive Framework**: Adjust policy based on evolving unemployment-inflation relationships

**Long-Term Considerations:**
1. **Structural Analysis**: Address high forecast errors suggesting structural economic changes
2. **Policy Communication**: Enhance forward guidance to smooth policy transmission
3. **Complementary Policies**: Coordinate with fiscal authorities for structural unemployment issues

### **Current Economic Environment Application**

**For High Inflation and Unemployment:**

**Phase 1 - Immediate Response (0-2 quarters):**
- Moderate interest rate increases targeting inflation control
- Clear communication about anti-inflation commitment
- Monitor unemployment response given rapid transmission

**Phase 2 - Adjustment Period (2-6 quarters):**
- Data-dependent policy adjustments based on unemployment trends
- Prepare for inflation feedback effects on policy expectations
- Maintain flexibility for labor market developments

**Phase 3 - Long-term Normalization (6+ quarters):**
- Gradual policy normalization considering unemployment-inflation lags
- Enhanced structural policy coordination
- Continuous model updating for changing economic relationships

### **Model Limitations & Future Enhancements**

**Current Limitations:**
- **High forecast uncertainty** for unemployment and federal funds rate
- **Structural break sensitivity** during crisis periods
- **Linear specification** may miss regime-switching behavior
- **Parameter instability** over long sample periods

**Recommended Extensions:**
- **Threshold VAR models** for regime-dependent relationships
- **Bayesian VAR** for improved forecasting with uncertainty bands
- **Structural VAR** for contemporaneous relationship identification
- **Time-varying parameter models** for evolving policy transmission

---

*This comprehensive VAR analysis provides Federal Reserve policymakers with empirical evidence for understanding monetary policy transmission mechanisms, optimizing policy timing, and formulating data-driven decisions to achieve dual mandate objectives while maintaining macroeconomic stability.*
