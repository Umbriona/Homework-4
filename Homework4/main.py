import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

def Mating(index,genotypeFitness,s):
    if(index[0] == 0 and index[1] == 0):
        return 'AA'
    elif(index[0] == 2 and index[1] == 2):
        return 'aa'
    elif(index[0] != index[1] and (index[0] == 1 or index[1] == 1) and index[0] != 2 and index[1] != 2):
        choseAllele = np.random.rand()
        if(choseAllele<1/2):
            return 'AA'
        else:
            return 'Aa'
    elif(index[0] != index[1] and (index[0] == 1 or index[1] == 1) and index[0] != 0 and index[1] != 0):
        choseAllele = np.random.rand()
        if (choseAllele < 1 / 2):
            return 'aa'
        else:
            return 'Aa'
    else:
        choseAllele = np.random.rand()
        if(choseAllele<1/4):
            return 'AA'
        elif(choseAllele<3/4):
            return 'Aa'
        else:
            return 'aa'


def Condition(genotypeFrequencey):
    if(genotypeFrequencey['AA'] == 0 and genotypeFrequencey['Aa'] == 0):
        return False, 'NoFix'
    elif(genotypeFrequencey['Aa'] == 0 and genotypeFrequencey['aa'] == 0):
        return False, 'Fix'
    else:
        return True,[]
def sim(s,N,genotypeFitness):
    genotypeFrequencey = {'AA': 0, 'Aa': 1 / N, 'aa': (N - 1) / N}
    listFixation = []
    X = True
    while X:
        tempList = []
        omega = genotypeFitness['AA'](s[i])*genotypeFrequencey['AA']+genotypeFitness['Aa'](s[i])*genotypeFrequencey['Aa']+genotypeFitness['aa'](s[i])*genotypeFrequencey['aa']
        sizeBoxAA = genotypeFitness['AA'](s[i]) * genotypeFrequencey['AA'] / omega
        sizeBoxAa = genotypeFitness['Aa'](s[i]) * genotypeFrequencey['Aa'] / omega
        sizeBoxaa = genotypeFitness['aa'](s[i]) * genotypeFrequencey['aa'] / omega
        sizeBoxList = np.array([sizeBoxAA,sizeBoxAa,sizeBoxaa])
        for n in range(N): # generate new generation
            choseParents = [np.random.rand(),np.random.rand()]
            box = np.array([sizeBoxList[0],sizeBoxList[0]])
            index = np.array([0,0])
            for k in range(len(choseParents)):
                while(box[k] < choseParents[k]):
                    index[k] += 1
                    box[k] += sizeBoxList[index[k]]
            tempList.append(Mating(index,genotypeFitness,s[i])) #New individual in the new generation
            # Generate new frequencis
            numAA = tempList.count('AA')
            numAa = tempList.count('Aa')
            numaa = tempList.count('aa')
            genotypeFrequencey['AA'] = numAA / N
            genotypeFrequencey['Aa'] = numAa / N
            genotypeFrequencey['aa'] = numaa / N
        X, temp = Condition(genotypeFrequencey)
        if(temp != []):
            listFixation.append(temp)               # count number of fixations
    pFix = listFixation.count('Fix') / numberOfRepeats
    return pFix
s = np.array([10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2,0.1])
N = 1000
numberOfRepeats = 10000
allels = {'AA','Aa','aa'}
genotypeFitness = {'AA': lambda s: 1,'Aa': lambda s: 1-s/2,'aa': lambda s: 1-s}


listPfix = []
for i in range(len(s)):
    num_cores = multiprocessing.cpu_count()
    pFix = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for repeat in range(numberOfRepeats))
    listPfix.append(pFix)

fig, ax = plt.subplots()
plt.plot(s,listPfix)
ax.set_xscale('log')
plt.show()



dt = 0.01
t = np.arange(dt, 20.0, dt)

