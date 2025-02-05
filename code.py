# 📊 Portfolio Optimization Using VaR and CVaR with Python

import yfinance as yf
import pylab as py
import pulp  # Linear programming optimization library
import pandas as pd
import numpy as np  # Import numpy for matrix operations

# 📥 Function to Download and Read Historical Price Data
def Lecture(name):
    return py.array(yf.Ticker(name).history(start="2023-10-01", end="2024-10-01").Close)

# 🏷️ Helper Function to Get Prices of Multiple Assets
def get_Prix(Code):
    Prix = [Lecture(Code[i]) for i in range(len(Code))]
    return Prix

# 🏷️ Function to Calculate Daily Returns from Price Data
def get_Rdt(Code):
    Prix = get_Prix(Code)
    R = [(Prix[i][1:] - Prix[i][:-1]) / Prix[i][:-1] for i in range(len(Code))]
    return py.array(R)

# 🧮 Select Assets with the Lowest Correlation
def NotCorr(Code, t):
    Rdts = get_Rdt(Code).T  # Transpose to structure returns as a time-series DataFrame
    Rdts = pd.DataFrame(Rdts, columns=Code)
    Mat_Corr = Rdts.corr()  # Calculate correlation matrix

    # Average correlation for each asset
    Mean_Corr = Mat_Corr.apply(lambda x: (x.sum() - 1) / (len(x) - 1), axis=1)
    # Select assets with the lowest average correlation
    Least_Corr = Mean_Corr.nsmallest(t).index.tolist()

    return Least_Corr

# 📈 Select Assets with the Highest Sharpe Ratio
def ratio_sharpe(Code, t, risk_free_rate=0):
    Rdts = get_Rdt(Code)

    # Calculate mean returns and volatility (standard deviation)
    Rdt_Mean = [py.mean(Rdts[i]) for i in range(len(Rdts))]
    Rdt_Std = [py.std(Rdts[i]) for i in range(len(Rdts))]

    # Calculate Sharpe Ratio for each asset
    Sharpe_Ratio = [(Rdt_Mean[i] - risk_free_rate) / Rdt_Std[i] if Rdt_Std[i] != 0 else 0 for i in range(len(Rdt_Mean))]

    # Create a DataFrame with return and risk information
    data = {'Ticker': Code, 'Rdt_Mean': Rdt_Mean, 'Volatility': Rdt_Std, 'Sharpe_Ratio': Sharpe_Ratio}
    df = pd.DataFrame(data)

    # Select the top t assets based on the Sharpe Ratio
    top_t = df.sort_values(by='Sharpe_Ratio', ascending=False).head(t)

    return top_t['Ticker'].tolist()

# 🏦 Function to Compute Mean Returns and Covariance Matrix
def Rdt_Cov(Code):
    Rdts = get_Rdt(Code)
    Rdt_Mean = [py.mean(Rdts[i]) for i in range(len(Rdts))]
    Mat_Cov = py.cov(Rdts)
    return Rdt_Mean, Mat_Cov

# 🔄 Simulate Asset Returns Based on Mean and Covariance Matrix
def simulation(R, V, N, t):
    L = np.linalg.cholesky(V).T  # Perform Cholesky decomposition for sampling
    Vect_Actifs = [R + np.dot(np.random.normal(0, 1, t), L) for i in range(N)]
    return Vect_Actifs

# ⚖️ Optimization of VaR and CVaR
def Optimisation_VaR_CVaR(Rdt_Mean, Mat_Cov, t, epsilon, N, C, alpha=0.95):
    r = simulation(Rdt_Mean, Mat_Cov, N, t)

    # Define decision variables
    gamma = pulp.LpVariable('gamma')  # VaR
    cvar = pulp.LpVariable('cvar')    # CVaR
    y = [pulp.LpVariable(f'y{i}', cat='Binary') for i in range(N)]
    w = [pulp.LpVariable(f'w{i}', lowBound=0) for i in range(t)]  # Portfolio weights (non-negative)

    # Define optimization problem: minimize CVaR
    prob = pulp.LpProblem("Minimisation_CVaR", pulp.LpMinimize)
    prob += cvar

    # Constraints for VaR and CVaR
    for j in range(N):
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= gamma + C * y[j]  # VaR constraint
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= cvar  # CVaR constraint

    prob += pulp.lpSum(y) <= int(epsilon * N)  # Number of losses exceeding VaR
    prob += pulp.lpSum(w) == 1  # Weights must sum to 1
    prob += cvar >= gamma + (1 / (1 - alpha)) * (pulp.lpSum(y[j] for j in range(N)) / N)

    # Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(pulp.LpStatus[prob.status])

    # Return optimized weights and CVaR
    Pond = [pulp.value(w[i]) for i in range(t)]
    Sol_cvar = pulp.value(cvar)

    return np.array(Pond), Sol_cvar

# 🧮 Calculate VaRSR (Sharpe Ratio adjusted for VaR)
def calculate_VaRSR(Pond, Rdt_Mean, Mat_Cov, gamma, risk_free_rate=0):
    portfolio_return = sum(Pond[i] * Rdt_Mean[i] for i in range(len(Pond)))
    portfolio_volatility = np.sqrt(np.dot(Pond.T, np.dot(Mat_Cov, Pond)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility - gamma

# 📋 Display Portfolio Results
def Affichage_CVaR_VaRSR(Pond, cvar, VaRSR, Small_Corr_Code, Rdt_Mean, Mat_Cov):
    print(f"\n{'Ticker':<20} {'Mean Return':<20} {'Daily Volatility':<20} {'Weight':<10}")
    for i in range(len(Small_Corr_Code)):
        volatility = np.sqrt(Mat_Cov[i][i])
        print(f"{Small_Corr_Code[i]:<20} {Rdt_Mean[i]:<20.6f} {volatility:<20.6f} {Pond[i]:<10.2f}")

    print(f"\nCVaR : {cvar}")
    print(f"Expected Portfolio Return : {sum(Rdt_Mean[i] * Pond[i] for i in range(len(Small_Corr_Code))):.6f}")
    print(f"Portfolio Volatility : {np.sqrt(np.dot(Pond.T, np.dot(Mat_Cov, Pond))):.6f}")
    print(f"VaRSR : {VaRSR:.6f}")

# 📈 List of CAC 40 Assets
Code = [
    'SW.PA', 'AI.PA', 'AIR.PA', 'BN.PA', 'ORA.PA', 'BNP.PA', 'SGO.PA', 'DG.PA', 
    'OR.PA', 'MC.PA', 'ACA.PA', 'AC.PA', 'GLE.PA', 'SAN.PA', 'ML.PA', 'VIV.PA', 
    'EN.PA', 'RI.PA', 'KER.PA', 'CA.PA', 'ATO.PA', 'VIE.PA', 'HO.PA', 'ENGI.PA', 
    'LR.PA', 'SU.PA', 'CAP.PA', 'WLN.PA'
]

# 🗂️ User Input for Portfolio Selection
choix = input("Select assets by (1) correlation or (2) Sharpe ratio: ")
t = int(input("How many assets do you want to select? "))

# Choose Assets Based on Input
if choix == '1':
    Small_Corr_Code = NotCorr(Code, t)
elif choix == '2':
    Small_Corr_Code = ratio_sharpe(Code, t)
else:
    print("Invalid choice")
    exit()

# Perform Optimization and Display Results
Rdt_Mean, Mat_Cov = Rdt_Cov(Small_Corr_Code)
Pond, cvar = Optimisation_VaR_CVaR(Rdt_Mean, Mat_Cov, len(Small_Corr_Code), 0.05, 1000, 1000)
VaRSR = calculate_VaRSR(Pond, Rdt_Mean, Mat_Cov, cvar)
Affichage_CVaR_VaRSR(Pond, cvar, VaRSR, Small_Corr_Code, Rdt_Mean, Mat_Cov)
