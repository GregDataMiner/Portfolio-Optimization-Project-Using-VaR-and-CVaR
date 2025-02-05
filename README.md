# **📊 Portfolio Optimization Project Using VaR and CVaR**

This project focuses on optimizing a financial portfolio based on **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)**, using **Python**. The approach explores two main strategies for portfolio selection:
1. **Minimizing correlation** between asset returns.
2. **Maximizing the Sharpe ratio** for selected assets.

---

## **🛠️ Key Features of the Project**

1. **Asset Selection**
   - Assets are chosen from the **CAC 40** index based on their correlation or Sharpe ratio.
   - Example assets: Orange, Danone, Sanofi, Atos.

2. **Optimization Methods**
   - **VaR**: Measures the maximum potential loss at a certain confidence level (e.g., 95%).
   - **CVaR**: Captures the average loss in extreme scenarios where the loss exceeds the VaR threshold.
   - **VaRSR**: An advanced measure combining VaR with the Sharpe ratio for better risk-adjusted returns.

3. **Portfolio Simulation**
   - Generates future scenarios of returns to test risk models.
   - Optimizes allocation weights to minimize CVaR.

4. **Results Comparison**
   - Multiple portfolios are generated and evaluated to find the best combination of assets for risk and return.

---

## **🔍 Code Overview**

### **Libraries Used**
- `yfinance`: To fetch stock price data.
- `pandas`, `numpy`: For data manipulation.
- `pylab`: Numerical calculations and simulations.
- `pulp`: Optimization library for linear programming.

### **Key Functions**
1. **Asset Data Handling**
   - `get_Rdt()`: Calculates daily returns from historical price data.
   - `NotCorr()`: Selects assets with the lowest average correlation.
   - `ratio_sharpe()`: Selects assets with the highest Sharpe ratio.

2. **Portfolio Optimization**
   - `Optimisation_VaR_CVaR()`: Defines and solves the optimization problem to minimize CVaR.
   - `calculate_VaRSR()`: Computes the VaRSR to evaluate the risk-return balance.

3. **Result Visualization**
   - Output includes portfolio weights, expected returns, volatility, and risk measures (VaR, CVaR, VaRSR).

---

## **📈 Example Results**

### **Asset Violations of VaR**
![VaR Violations AXA](./images/var_violations_axa.png)
- This graph shows how often AXA returns exceeded the VaR threshold over time. Red points indicate violations.

### **Portfolio Risk and Return**
![Portfolio Optimization](./images/portfolio_optimization.png)
- The optimized portfolio demonstrates improved risk-adjusted returns by reducing CVaR while maximizing the Sharpe ratio.

### **Performance Table**
| Method       | VaR       | ES        |
|--------------|-----------|-----------|
| Historical   | -48639.58 | -68486.19 |
| Gaussian     | -40776.76 | -46725.64 |
| Modified     | -99970.03 | -99970.03 |

---

## **📝 Report Structure**

1. **Introduction**
   - Overview of portfolio risk management using VaR and CVaR.
   - Importance of asset diversification and risk reduction.

2. **Asset Selection**
   - Explanation of chosen assets from the CAC 40 index.
   - Criteria for selecting assets with low correlation or high Sharpe ratio.

3. **Optimization Process**
   - Steps to define the optimization problem and solve it using simulations.
   - Comparison between portfolios optimized by correlation and Sharpe ratio.

4. **Results and Analysis**
   - Detailed performance metrics for each portfolio.
   - Discussion on how market trends, like bullish phases, affect portfolio selection.

5. **Conclusion**
   - Summary of findings on the effectiveness of different risk models.
   - Recommendations for improving portfolio risk management strategies.

---

## **🚀 How to Run the Code**

1. Install the required libraries:
   ```bash
   pip install yfinance pandas numpy pulp matplotlib
   ```

2. Run the Python script to start the portfolio optimization process:
   ```python
   python portfolio_optimization.py
   ```

3. Choose the criteria for asset selection:
   - Option 1: Minimize correlation.
   - Option 2: Maximize Sharpe ratio.

4. View the output results, including portfolio allocations, risk measures, and visualizations.

---

## **🔗 Additional Resources**

- **Report Document**: [Optimization Report (PDF)](report.pdf)
- **Complete Python Code**: [Portfolio Optimization Script](code.py)

Feel free to clone the repository and experiment with the code!
