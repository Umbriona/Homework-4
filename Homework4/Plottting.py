import numpy as np
import matplotlib.pyplot as plt

def pFixAnalytical(S,N):

    return (1-np.exp(-S))/(1-np.exp(-2*S*N))

N = np.array([100,200,500,1000,5000,10000,100000])
s = np.ones(7)*10**-3

SimData = np.array([0.0053,0.002999,0.0018,0.0011,0.001,0.001,0.001])

fig, ax = plt.subplots()
ax.plot(N,SimData,label='Simulation')
ax.plot(N,pFixAnalytical(s[0],N),label='Analytical Kimura')
ax.plot(N,s,label='Analytical Haldane')
ax.set_xscale('log')
ax.set_xlabel('s')
ax.set_ylabel('Probability of fixation of advantageous')
ax.set_title('Likelihood of fixation')
ax.legend()

plt.show()
