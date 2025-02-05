'code pour minimiser la cvar en se basant sur les actifs au meilleur ratio de Sharpe ou les moins corrélés'

import yfinance as yf
import pylab as py
import pulp
import pandas as pd
import numpy as np  # Importation de numpy

def Lecture(name):
    return py.array(yf.Ticker(name).history(start="2023-10-01", end="2024-10-01").Close)

def get_Prix(Code):
    Prix = [Lecture(Code[i]) for i in range(len(Code))]
    return Prix

def get_Rdt(Code):
    Prix = get_Prix(Code)
    R = [(Prix[i][1:] - Prix[i][:-1]) / Prix[i][:-1] for i in range(len(Code))]
    return py.array(R)

# Fonction pour sélectionner les t actifs les moins corrélés
def NotCorr(Code, t):
    Rdts = get_Rdt(Code)
    Rdts = Rdts.T
    Rdts = pd.DataFrame(Rdts, columns=Code)
    Mat_Corr = Rdts.corr()

    # Calculer la moyenne des corrélations pour chaque actif (sans lui-même)
    Mean_Corr = Mat_Corr.apply(lambda x: (x.sum() - 1) / (len(x) - 1), axis=1)
    # Trouver les actifs avec les plus faibles moyennes de corrélation
    Least_Corr = Mean_Corr.nsmallest(t).index.tolist()

    return Least_Corr

# Fonction pour sélectionner les t actifs ayant le meilleur ratio de Sharpe
def ratio_sharpe(Code, t, risk_free_rate=0):
    Rdts = get_Rdt(Code)

    # Calculer les rendements moyens et la volatilité
    Rdt_Mean = [py.mean(Rdts[i]) for i in range(len(Rdts))]
    Rdt_Std = [py.std(Rdts[i]) for i in range(len(Rdts))]  # Volatilité (écart-type des rendements)

    # Calculer le ratio de Sharpe pour chaque actif
    Sharpe_Ratio = [(Rdt_Mean[i] - risk_free_rate) / Rdt_Std[i] if Rdt_Std[i] != 0 else 0 for i in range(len(Rdt_Mean))]

    # Créer un DataFrame pour stocker les informations
    data = {'Ticker': Code, 'Rdt_Mean': Rdt_Mean, 'Volatility': Rdt_Std, 'Sharpe_Ratio': Sharpe_Ratio}
    df = pd.DataFrame(data)

    # Trier par ratio de Sharpe décroissant et sélectionner les t meilleurs
    top_t = df.sort_values(by='Sharpe_Ratio', ascending=False).head(t)

    return top_t['Ticker'].tolist()

# Fonction pour calculer les rendements moyens et la matrice de covariance
def Rdt_Cov(Code):
    Rdts = get_Rdt(Code)
    Rdt_Mean = [py.mean(Rdts[i]) for i in range(len(Rdts))]
    Mat_Cov = py.cov(Rdts)
    return Rdt_Mean, Mat_Cov

# Fonction de simulation de rendements
def simulation(R, V, N, t):
    L = np.linalg.cholesky(V).T
    Vect_Actifs = [R + np.dot(np.random.normal(0, 1, t), L) for i in range(N)]  # Correction ici
    return Vect_Actifs

# Fonction d'optimisation VaR et CVaR
def Optimisation_VaR_CVaR(Rdt_Mean, Mat_Cov, t, epsilon, N, C, alpha=0.95):
    r = simulation(Rdt_Mean, Mat_Cov, N, t)

    # Variables de décision
    gamma = pulp.LpVariable('gamma')  # VaR
    cvar = pulp.LpVariable('cvar')    # CVaR
    y = [pulp.LpVariable(f'y{i}', cat='Binary') for i in range(N)]
    w = [pulp.LpVariable(f'w{i}', lowBound=0) for i in range(t)]  # Pondérations non négatives

    # Définition du problème de minimisation de la CVaR
    prob = pulp.LpProblem("Minimisation_CVaR", pulp.LpMinimize)

    # Fonction à minimiser : CVaR
    prob += cvar

    # Contraintes pour la VaR et la CVaR
    for j in range(N):
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= gamma + C * y[j]  # VaR constraint
        prob += -pulp.lpSum(r[j][i] * w[i] for i in range(t)) <= cvar  # CVaR constraint

    prob += pulp.lpSum(y) <= int(epsilon * N)  # Nombre de scénarios où les pertes dépassent la VaR
    prob += pulp.lpSum(w) == 1  # La somme des poids doit être égale à 1
    prob += cvar >= gamma + (1 / (1 - alpha)) * (pulp.lpSum(y[j] for j in range(N)) / N)  # Condition de la CVaR

    # Résolution
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    print(pulp.LpStatus[prob.status])

    Pond = [pulp.value(w[i]) for i in range(t)]
    Sol_gamma = pulp.value(gamma)
    Sol_cvar = pulp.value(cvar)

    return np.array(Pond), Sol_cvar  # Ne plus retourner gamma

# Calcul du VaRSR
def calculate_VaRSR(Pond, Rdt_Mean, Mat_Cov, gamma, risk_free_rate=0):
    portfolio_return = sum(Pond[i] * Rdt_Mean[i] for i in range(len(Pond)))
    portfolio_volatility = np.sqrt(np.dot(Pond.T, np.dot(Mat_Cov, Pond)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility - gamma

# Fonction d'affichage des résultats incluant la volatilité journalière de chaque actif, la CVaR et le VaRSR
def Affichage_CVaR_VaRSR(Pond, cvar, VaRSR, Small_Corr_Code, Rdt_Mean, Mat_Cov):
    print(f"\n{'Symbole boursier':<20} {'Rendements Moyen':<20} {'Volatilité Journalière':<20} {'Pondérations':<10}")
    for i in range(len(Small_Corr_Code)):
        volatilité = np.sqrt(Mat_Cov[i][i])  # La volatilité journalière est l'écart-type de la diagonale de la matrice de covariance
        print(f"{Small_Corr_Code[i]:<20} {Rdt_Mean[i]:<20.6f} {volatilité:<20.6f} {Pond[i]:<10.2f}")

    print(f"\nCVaR : {cvar}")
    print(f"Rendement quotidien espéré du portefeuille : {sum(Rdt_Mean[i] * Pond[i] for i in range(len(Small_Corr_Code))):.6f}")
    print(f"Volatilité journalière du portefeuille : {np.sqrt(np.dot(Pond.T, np.dot(Mat_Cov, Pond))):.6f}")
    print(f"VaRSR : {VaRSR:.6f}")

# Code des actifs
Code = [
    'SW.PA', 'AI.PA', 'AIR.PA', 'BN.PA', 'ORA.PA', 'BNP.PA','SGO.PA', 'DG.PA', 'OR.PA', 'MC.PA', 'ACA.PA', 'AC.PA', 'GLE.PA',
    'SAN.PA', 'ML.PA', 'VIV.PA', 'EN.PA', 'RI.PA', 'KER.PA', 'CA.PA','ATO.PA',
    'VIE.PA', 'HO.PA', 'ENGI.PA', 'LR.PA', 'SU.PA', 'CAP.PA', 'WLN.PA'
]

# Permettre à l'utilisateur de choisir entre les actifs les moins corrélés ou ceux avec le meilleur ratio de Sharpe
choix = input("Sélectionnez les actifs basés sur (1) la corrélation ou (2) le ratio de Sharpe : ")
t = int(input("Combien d'actifs voulez-vous sélectionner ? "))

if choix == '1':
    Small_Corr_Code = NotCorr(Code, t)  # Sélectionne les t actifs les moins corrélés
elif choix == '2':
    Small_Corr_Code = ratio_sharpe(Code, t)  # Sélectionne les t actifs avec le meilleur ratio de Sharpe
else:
    print("Choix invalide")
    exit()

# Calcul des rendements moyens et de la matrice de covariance
Rdt_Mean, Mat_Cov = Rdt_Cov(Small_Corr_Code)

# Optimisation uniquement pour la CVaR
Pond, cvar = Optimisation_VaR_CVaR(Rdt_Mean, Mat_Cov, len(Small_Corr_Code), 0.05, 1000, 1000)

# Calcul du VaRSR
VaRSR = calculate_VaRSR(Pond, Rdt_Mean, Mat_Cov, cvar)

# Affichage des résultats incluant noms des actifs, pondérations, CVaR, VaRSR et volatilité journalière
Affichage_CVaR_VaRSR(Pond, cvar, VaRSR, Small_Corr_Code, Rdt_Mean, Mat_Cov)

