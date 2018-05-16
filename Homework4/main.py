import numpy as np

def Matin(index,genotypeFitness):
    if(index[0] == )
def Condition(genotypeFrequencey):
    if(genotypeFrequencey['AA'] == 0 and genotypeFrequencey['Aa'] == 0):
        return False
    elif(genotypeFrequencey['Aa'] == 0 and genotypeFrequencey['aa'] == 0):
        return False
    else:
        return True

s = np.array([10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2,0.1])
N = 1000
numberOfRepeats = 100
allels = {'AA','Aa','aa'}
genotypeFitness = {'AA': lambda s: 1,'Aa': lambda s: 1-s/2,'aa': lambda s: 1-s}
genotypeFrequencey = {'AA': 0, 'Aa': 1/N, 'aa': (N-1)/N}
for j in s:
    for repeat in range(numberOfRepeats):
        while Condition(genotypeFrequencey):
            omega = genotypeFitness['AA'](s[i])*genotypeFrequencey['AA']+genotypeFitness['Aa'](s[i])*genotypeFrequencey['Aa']+genotypeFitness['aa'](s[i])*genotypeFrequencey['aa']

            sizeBoxAA = genotypeFitness['AA'](s[i])*genotypeFrequencey['AA']/omega
            sizeBoxAa = genotypeFitness['Aa'](s[i])*genotypeFrequencey['Aa']/omega
            sizeBoxaa = genotypeFitness['aa'](s[i]) * genotypeFrequencey['aa'] / omega
            sizeBoxList = {sizeBoxAA,sizeBoxAa,sizeBoxaa}
            for i in range(N): # generate new generation
                choseParents = np.array([np.random.rand(),np.random.rand()])
                box = np.array([sizeBoxList[0],sizeBoxList[0]])
                index = np.array([0,0])
                for k in len(choseParents):
                    while(box < choseParents[k]):
                        index[k] += 1
                        box[k] += sizeBoxList[index]
                temp = Mating(index,genotypeFitness)
