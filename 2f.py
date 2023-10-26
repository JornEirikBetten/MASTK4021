import pandas as pd 
import os 
import numpy as np 
import matplotlib.pyplot as plt 

""" 
Simple tandem iteration scheme for finding the optimal alpha and beta (empirical Bayes) in the 
pi(w) = N(w; 0, alpha^{-1}I), pi(y|w) = N(y; phiw, betaI), and running it on the bjornholt.csv 
dataset for skiing days per year. 
"""

path = os.getcwd()
fig_path = path + "/MA/Figures/"

data_path = path + "/MA/Data/bjornholt.csv"
df = pd.read_csv(data_path)
print(df)

years = df["year"]
skiing_days = df["days"]

# Design matrix
X = np.c_[np.ones(len(years)), years-1900]


def compute_precision_mean(alpha, beta, X, y): 
    """
    Compute the precision matrix and and mean vector of the posterior 
    distribution. 
    """
    Lambda = alpha * np.eye(X.shape[-1]) + beta * X.T @ X
    print("Y shape: ", y.shape)
    print("X shape: ", X.shape)
    mean_vector = beta* np.linalg.inv(Lambda) @ X.T @ y 
    return Lambda, mean_vector


def compute_gamma(X, beta, alpha): 
    """
    Computes gamma as defined in Equation (5.21)
    """
    eigvals = np.linalg.eigvalsh(beta * X.T @ X)
    gamma = 0 
    for eigval in eigvals: 
        gamma += eigval/(alpha + eigval)
    return gamma 


def update_alpha(alpha, beta, X, y):
    """
    \alpha = \frac{\gamma}{\mu^T\mu}
    """ 
    L, mu = compute_precision_mean(alpha, beta, X, y)
    gamma = compute_gamma(X, beta, alpha)
    new_alpha = gamma/(mu.T @ mu)
    return new_alpha 

def update_beta(alpha, beta, X, y): 
    """
    \frac{1}{\beta} = \frac{1}{n + \gamma}\sum_{i=1}^n[y_i - \mu^T\phi(x_i)]
    """
    n = len(y)
    L, mu = compute_precision_mean(alpha, beta, X, y)
    gamma = compute_gamma(X, beta, alpha)
    
    vector = y- X @ mu 
    beta_inv = 1/(n + gamma)*np.dot(vector, vector)
    new_beta = 1/beta_inv 
    return new_beta


def tandem_update(alpha, beta, X, y, tol = 1e-11, max_iter = 10000): 
    """
    Updates alpha and beta untill convergence. 
    """
    new_alpha = update_alpha(alpha, beta, X, y)
    new_beta = update_beta(new_alpha, beta, X, y)
    counter = 1 
    while abs(alpha-new_alpha) > tol and abs(beta - new_beta) >  tol: 
        alpha = new_alpha 
        beta = new_beta 
        new_alpha = update_alpha(alpha, beta, X, y)
        new_beta = update_beta(new_alpha, beta, X, y)
        counter += 1 
        if counter > max_iter: 
            break 
        
    return new_alpha, new_beta, counter 

init_alpha = 1.0; init_beta = 1.0 
alpha, beta, counter = tandem_update(init_alpha, init_beta, X, skiing_days, tol = 1e-15)


def compute_log_marginal_likelihood(alpha, beta, X, y):
    """
    log\pi(y) = log\alpha + n/2log\beta/2pi - 1/2logdet(L) - beta/2 y^Ty + 1/2(mu^TLmu)
    """
    L, mu = compute_precision_mean(alpha, beta, X, y)
    detL = np.linalg.det(L)
    n = len(y)
    return np.log(alpha) + n/2*np.log(beta/(2*np.pi)) - 1/2*np.log(detL) - beta/2 *y.T @ y + 1/2 * mu.T @ L @ mu 
     
print(f"Initial alpha = {init_alpha:.2f}, initial beta = {init_beta:.2f}")
print(f"Converged: alpha = {alpha}, beta = {beta}. Converged after {counter} iterations.")
print(f"Log marginal likelihood: {compute_log_marginal_likelihood(alpha, beta, X, skiing_days)}")
print(f"Initial log marginal likelihood: {compute_log_marginal_likelihood(init_alpha, init_beta, X, skiing_days)}")

mll_m1 = compute_log_marginal_likelihood(alpha, beta, X, skiing_days)
X = np.c_[np.ones(len(years))] 

alpha, beta, counter = tandem_update(init_alpha, init_beta, X, skiing_days, tol = 1e-15)
print(f"Initial alpha = {init_alpha:.2f}, initial beta = {init_beta:.2f}")
print(f"Converged: alpha = {alpha}, beta = {beta}. Converged after {counter} iterations.")
print(f"Log marginal likelihood: {compute_log_marginal_likelihood(alpha, beta, X, skiing_days)}")
print(f"Initial log marginal likelihood: {compute_log_marginal_likelihood(init_alpha, init_beta, X, skiing_days)}")
mll_m0 = compute_log_marginal_likelihood(alpha, beta, X, skiing_days)

mll_combined = np.log(0.5*np.exp(mll_m0) + 0.5*np.exp(mll_m1))
print(f"MLL combined model: {mll_combined}")


pi_m0 = np.exp(mll_m0); pi_m1 = np.exp(mll_m1)
p0 = 0.5*pi_m0/(0.5*(pi_m0 + pi_m1))
p1 = 0.5*pi_m1/(0.5*(pi_m0 + pi_m1))
print("Post prob m=0: ", p0)
print("Post prob m=1: ", p1)