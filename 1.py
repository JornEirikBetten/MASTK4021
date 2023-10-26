import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd  
import math 
import os  


path = os.getcwd()
fig_path = path + "/MA/Figures/"


def gamma(a, b, theta): 
    return b**a/math.gamma(a)*theta*(a-1)*np.exp(-b*theta)

def likelihood(theta, data): 
    output = 1 
    for datapoint in data: 
        output *= theta/(2*np.sqrt(datapoint))*np.exp(-theta*np.sqrt(datapoint))
    return output 

data = [0.771, 0.140, 0.135, 0.007, 0.088, 0.008,
0.268, 0.022, 0.131, 0.142, 0.421, 0.125]
thetas = np.linspace(0.001, 10, 1001)
prior = np.array([gamma(4.4, 2.2, theta) for theta in thetas])
prior = prior/np.sum(prior)
likelihood_ = np.array([likelihood(theta, data) for theta in thetas])
posterior = prior*likelihood_
posterior = posterior/np.sum(posterior) 




fig = plt.figure()
plt.plot(thetas, prior, label = "Prior", color = "black", linestyle = "dotted")
plt.plot(thetas, posterior, label = "Posterior", color = "black", linestyle = "solid")
plt.xlabel(r"$\theta$")
plt.ylabel("probability density")
plt.yticks([], [])
plt.legend()
plt.savefig(fig_path + "1g.pdf", format = "pdf", bbox_inches = "tight")


thetas = np.linspace(0, 1000, 100001)
prior_015 = np.array([gamma(4.4, 2.2, theta) for theta in thetas if theta<1.5])
prior_153 = np.array([gamma(4.4, 2.2, theta) for theta in thetas if theta>1.5 and theta<3])
full_prior = np.array([gamma(4.4, 2.2, theta) for theta in thetas])
probability_015 = np.sum(prior_015)/np.sum(full_prior)
probability_153 = np.sum(prior_153)/np.sum(full_prior)
probability_3inf = 1 - probability_015 - probability_153
likelihood_015 = np.array([likelihood(theta, data) for theta in thetas if theta < 1.5])
likelihood_153 = np.array([likelihood(theta, data) for theta in thetas if theta >1.5 and theta < 3])
full_likelihood = np.array([likelihood(theta, data) for theta in thetas])
posterior_015 = prior_015*likelihood_015 
posterior_153 = prior_153*likelihood_153
full_posterior = full_prior*full_likelihood

print("Prior:")
print(f"Inside (0, 1.5): {probability_015}")
print(f"Inside (1.5, 3): {probability_153}")
print(f"Inside (3, infinity): {probability_3inf}")

print("Posterior:")
print(f"Inside (0, 1.5): {np.sum(posterior_015)/np.sum(full_posterior)}")
print(f"Inside (1.5, 3): {np.sum(posterior_153)/np.sum(full_posterior)}")
print(f"Inside (3, infinity): {1-np.sum(posterior_015)/np.sum(full_posterior)-np.sum(posterior_153)/np.sum(full_posterior)}")