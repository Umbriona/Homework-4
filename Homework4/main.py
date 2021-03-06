import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

def SaveData(fileName = 'new file', data = [0,0,0]):
    fw = open(fileName, 'w')
    fw.write(str(data))
    fw.close()
    return True

def ReadData(fileName = 'new file'):
    fr = open(fileName,'r')
    data = fr.read()
    flag = True
    if(data == []):
        flag = False
    fr.close()
    return data, flag

def Mating(index,s):
    if(index[0] == 0 and index[1] == 0):
        return 'AA'
    elif(index[0] == 2 and index[1] == 2):
        return 'aa'
    elif(index[0] != index[1] and (index[0] == 1 or index[1] == 1) and index[0] != 2 and index[1] != 2):
        choseAllele = np.random.rand()
        if(choseAllele<=1/2):
            return 'AA'
        else:
            return 'Aa'
    elif(index[0] != index[1] and (index[0] == 1 or index[1] == 1) and index[0] != 0 and index[1] != 0):
        choseAllele = np.random.rand()
        if (choseAllele <= 1 / 2):
            return 'aa'
        else:
            return 'Aa'
    else:
        choseAllele = np.random.rand()
        if(choseAllele<=1/4):
            return 'AA'
        elif(choseAllele<=3/4):
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
def Sim(s,N,numberOfRepeats,j):
    i=0
    #j=0
    genotypeFrequencey = {'AA': 0, 'Aa': 1 / N[j], 'aa': (N[j] - 1) / N[j]}
    listFixation = []
    X = True
    while X:
        tempList = []
        omega = 1*genotypeFrequencey['AA']+(1-s/2)*genotypeFrequencey['Aa']+(1-s)*genotypeFrequencey['aa']
        sizeBoxAA = 1 * genotypeFrequencey['AA'] / omega
        sizeBoxAa = (1-s/2) * genotypeFrequencey['Aa'] / omega
        sizeBoxaa = (1-s) * genotypeFrequencey['aa'] / omega
        sizeBoxList = np.array([sizeBoxAA,sizeBoxAa,sizeBoxaa])
        for n in range(N[j]): # generate new generation
            choseParents = [np.random.rand(),np.random.rand()]
            box = np.array([sizeBoxList[0],sizeBoxList[0]])
            index = np.array([0,0])
            for k in range(len(choseParents)):
                while(box[k] < choseParents[k]):
                    index[k] += 1
                    box[k] += sizeBoxList[index[k]]
            tempList.append(Mating(index,s)) #New individual in the new generation
            # Generate new frequencis
            numAA = tempList.count('AA')
            numAa = tempList.count('Aa')
            numaa = tempList.count('aa')
            genotypeFrequencey['AA'] = numAA / N[j]
            genotypeFrequencey['Aa'] = numAa / N[j]
            genotypeFrequencey['aa'] = numaa / N[j]
        X, temp = Condition(genotypeFrequencey)
        if(temp != []):
            listFixation.append(temp)               # count number of fixations
    pFix = listFixation.count('Fix') / numberOfRepeats
    return pFix
# Theoretical plot
def pFixAnalytical(S,N):

    return (1-np.exp(-S))/(1-np.exp(-2*S*N))

def main():
    #s = np.array([10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2,0.1])
    s = 10**-3
    N = np.array([100,1000,5000,10000,100000])
    numberOfRepeats = 10000
    allels = {'AA','Aa','aa'}
    genotypeFitness = {'AA': lambda s: 1,'Aa': lambda s: 1-s/2,'aa': lambda s: 1-s}
    listSPfix = []
    RangeRun = len(N)
    for i in range(RangeRun):
        num_cores = multiprocessing.cpu_count()
        print(num_cores)
        listPfix = Parallel(n_jobs=num_cores)(delayed(Sim)(s,N,numberOfRepeats,i) for repeat in range(numberOfRepeats))
        listSPfix.append(sum(listPfix))
        print(listSPfix)
    fig, ax = plt.subplots()
    ax.plot(s,listSPfix,label='Simulation')
    ax.plot(s,pFixAnalytical(s,N),label='Analytical Kimura')
    ax.plot(s,s,label='Analytical Haldane')
    ax.set_xscale('log')
    ax.set_xlabel('s')
    ax.set_ylabel('Probability of fixation of advantageous')
    ax.set_title('Likelihood of fixation')
    ax.legend()

    fileName = 'Data from Run with multiple N'
    data = str(s)+'\n'+ str(pFixAnalytical(s,N))+'\n'+str(listSPfix)
    SaveData(fileName= fileName, data= data)

    plt.show()



if __name__ == '__main__':
    main()