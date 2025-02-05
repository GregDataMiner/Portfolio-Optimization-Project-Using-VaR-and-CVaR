
# 📊 **Portfolio Optimization Using VaR and CVaR**

This project demonstrates how to optimize a financial portfolio by minimizing the Conditional Value at Risk (CVaR). While the report presents results based on a predefined set of parameters, the Python program is **interactive**, allowing users to:
- Select assets based on either **correlation** or **Sharpe ratio**.
- Specify the **number of assets** to include in the portfolio.

[The report display and analyse specific parameters available here](report_FR.pdf)

---
![cover](Image/cover.png)

---

## 📋 **Main Functions Overview**

### **1. Download and Process Asset Data**
```python
def get_Rdt(Code):
    Prix = [py.array(yf.Ticker(name).history(start="2023-10-01", end="2024-10-01").Close) for name in Code]
    R = [(Prix[i][1:] - Prix[i][:-1]) / Prix[i][:-1] for i in range(len(Code))]
    return py.array(R)
```
- **Purpose:** This function retrieves daily prices for each asset and calculates daily returns.
- **Input:** A list of asset tickers.
- **Output:** An array of daily returns for each asset.

---

### **2. Asset Selection by Correlation**
```python
def NotCorr(Code, t):
    Rdts = get_Rdt(Code).T
    Rdts = pd.DataFrame(Rdts, columns=Code)
    Mat_Corr = Rdts.corr()
    Mean_Corr = Mat_Corr.apply(lambda x: (x.sum() - 1) / (len(x) - 1), axis=1)
    return Mean_Corr.nsmallest(t).index.tolist()
```
- **Purpose:** Selects the top `t` assets with the lowest average correlation.
- **Input:** A list of asset tickers and the number of assets `t` to select.
- **Output:** A list of the least correlated assets.

---

### **3. Asset Selection by Sharpe Ratio**
```python
def ratio_sharpe(Code, t, risk_free_rate=0):
    Rdts = get_Rdt(Code)
    Rdt_Mean = [py.mean(Rdts[i]) for i in range(len(Rdts))]
    Rdt_Std = [py.std(Rdts[i]) for i in range(len(Rdts))]
    Sharpe_Ratio = [(Rdt_Mean[i] - risk_free_rate) / Rdt_Std[i] if Rdt_Std[i] != 0 else 0 for i in range(len(Rdt_Mean))]
    data = {'Ticker': Code, 'Rdt_Mean': Rdt_Mean, 'Volatility': Rdt_Std, 'Sharpe_Ratio': Sharpe_Ratio}
    df = pd.DataFrame(data)
    return df.sort_values(by='Sharpe_Ratio', ascending=False).head(t)['Ticker'].tolist()
```
- **Purpose:** Selects the top `t` assets with the highest Sharpe ratio.
- **Input:** A list of asset tickers, number of assets `t`, and an optional risk-free rate.
- **Output:** A list of assets with the best Sharpe ratios.

---

### **4. Optimization of VaR and CVaR**
```python
def Optimisation_VaR_CVaR(Rdt_Mean, Mat_Cov, t, epsilon, N, C, alpha=0.95):
    r = simulation(Rdt_Mean, Mat_Cov, N, t)
    gamma = pulp.LpVariable('gamma')  # VaR
    cvar = pulp.LpVariable('cvar')    # CVaR
    y = [pulp.LpVariable(f'y{i}', cat='Binary') for i in range(N)]
    w = [pulp.LpVariable(f'w{i}', lowBound=0) for i in range(t)]

    prob = pulp.LpProblem("Minimisation_CVaR", pulp.LpMinimize)
    prob += cvar

    for j in range(N):
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= gamma + C * y[j]
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= cvar

    prob += pulp.lpSum(y) <= int(epsilon * N)
    prob += pulp.lpSum(w) == 1
    prob += cvar >= gamma + (1 / (1 - alpha)) * (pulp.lpSum(y[j] for j in range(N)) / N)

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return [pulp.value(w[i]) for i in range(t)], pulp.value(cvar)
```
- **Purpose:** Minimizes the CVaR of the portfolio using linear programming.
- **Input:** Mean returns, covariance matrix, number of assets, scenario constraints, and CVaR confidence level.
- **Output:** Optimized portfolio weights and the minimized CVaR.

---

### **5. Display Results**
```python
def Affichage_CVaR_VaRSR(Pond, cvar, VaRSR, Small_Corr_Code, Rdt_Mean, Mat_Cov):
    print(f"\n{'Ticker':<20} {'Mean Return':<20} {'Daily Volatility':<20} {'Weight':<10}")
    for i in range(len(Small_Corr_Code)):
        volatility = np.sqrt(Mat_Cov[i][i])
        print(f"{Small_Corr_Code[i]:<20} {Rdt_Mean[i]:<20.6f} {volatility:<20.6f} {Pond[i]:<10.2f}")
    print(f"\nCVaR : {cvar}")
    print(f"Expected Portfolio Return : {sum(Rdt_Mean[i] * Pond[i] for i in range(len(Small_Corr_Code))):.6f}")
    print(f"Portfolio Volatility : {np.sqrt(np.dot(Pond.T, np.dot(Mat_Cov, Pond))):.6f}")
    print(f"VaRSR : {VaRSR:.6f}")
```
- **Purpose:** Displays the optimized portfolio's details, including returns, volatility, weights, CVaR, and VaRSR.
- **Input:** Portfolio weights, CVaR, VaRSR, asset names, mean returns, and covariance matrix.

---

## 📂 **Project Structure**
- **Code:** Python script implementing the optimization.
- **Dependencies:** `pandas`, `pulp`, `yfinance`, `numpy`.
- **Usage:** Run the script and choose the method to select assets (correlation or Sharpe ratio).

---

## 📈 **Sample Output**
The script will print the optimized portfolio weights, expected return, volatility, and risk metrics. This helps in making informed investment decisions based on portfolio risk management.
---
```
