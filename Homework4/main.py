import numpy as np

s = np.array([10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2,0.1])
N = 1000

genotypeFitness = {'AA': lambda s: 1,'Aa': lambda s: 1-s/2,'aa': lambda s: 1-s}
genotypeFrequencey = {'AA': 0, 'Aa': 1/N, 'aa': (N-1)/N}

for i in range(N):
    
