import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
import Ploting
import FourierAnalys as FA

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

def Mating(Parent_1,Parent_2,N):
    offspring = np.zeros([int(N/2),1])
    randomeVector = np.random.rand(int(N/2))
    logicVectorParent11 = Parent_1 == 1
    logicVectorParent10 = Parent_1 == 0
    logicVectorParent1M1 = Parent_1 == -1
    logicVectorParent21 = Parent_2 == 1
    logicVectorParent20 = Parent_2 == 0
    logicVectorParent2M1 = Parent_2 == -1

    #Both vector elements either 1 or -1
    temp = np.logical_and(logicVectorParent11,logicVectorParent21)
    offspring[np.logical_and(logicVectorParent11,logicVectorParent21)] = 1
    offspring[np.logical_and(logicVectorParent1M1, logicVectorParent2M1)] = -1

    # Vector element is either -1 or 0 not both zero or -1
    tempXor1 = np.logical_xor(logicVectorParent1M1, logicVectorParent2M1)
    tempXor2 = np.logical_xor(logicVectorParent10, logicVectorParent20)
    tempAnd = np.logical_and(tempXor1, tempXor2)
    offspring[np.logical_and(tempAnd,randomeVector<0.5)] = -1
    offspring[np.logical_and(tempAnd, randomeVector >= 0.5)] = 0

    # Vector element is either 1 or 0 not both zero or 1
    tempXor1 = np.logical_xor(logicVectorParent11, logicVectorParent21)
    tempXor2 = np.logical_xor(logicVectorParent10, logicVectorParent20)
    tempAnd = np.logical_and(tempXor1, tempXor2)
    offspring[np.logical_and(tempAnd, randomeVector < 0.5)] = 1
    offspring[np.logical_and(tempAnd, randomeVector >= 0.5)] = 0

    # Vector elements are both 0
    tempAnd = np.logical_and(logicVectorParent10, logicVectorParent20)
    offspring[np.logical_and(tempAnd, randomeVector < 0.25)] = 1
    offspring[np.logical_and(tempAnd, randomeVector >= 0.25)] = 0
    offspring[np.logical_and(tempAnd, randomeVector >= 0.75)] = 0

    return offspring

def Migration(Population_1,Population_2,m,N):
    minVal = 0
    maxVal = int(N / 2 - 1)
    numImigrants = int(np.trunc(N / 2 * m))
    indexesOfImigrantsPopulation_1 = np.random.randint(minVal, maxVal, size=numImigrants, dtype=np.int16)
    indexesOfImigrantsPopulation_2 = np.random.randint(minVal, maxVal, size=numImigrants, dtype=np.int16)

    tempImigrantsPopulation_1 = Population_1[indexesOfImigrantsPopulation_1]
    Population_1[indexesOfImigrantsPopulation_1] =(-1)*Population_2[indexesOfImigrantsPopulation_2]
    Population_2[indexesOfImigrantsPopulation_2] =(-1)*tempImigrantsPopulation_1
    return Population_1,Population_2

def Selection(Population,s,N):
    populationSize = int(N)
    individualPopulationSize = int(N / 2)
    Population=np.append(Population,Population)
    num1 = np.sum(Population==1)
    num2 = np.sum(Population==0)
    num3 = np.sum(Population==-1)
    omega = 1 * num1 + (1 - s / 2) * num2 + (1 - s) * num3

    interval1 = num1 / omega
    interval2 = interval1 + (1 - s / 2) * num2 / omega

    selectionVector = np.random.rand(populationSize)
    Population[selectionVector <= interval1]=1
    Population[selectionVector>interval1] = 0
    Population[selectionVector >= interval2] = -1
    return np.append([Population[:individualPopulationSize]],[Population[individualPopulationSize:]],axis=0)


def Sim(s,m,N):
    individualPopulationSize = int(N/2)
    Population_1 = np.ones([individualPopulationSize,1], dtype=np.float16)
    Population_2 = (-1)*np.ones([individualPopulationSize,1], dtype=np.float16)

    # Initialize allele A
    minVal = 0
    maxVal = int(N/2-1)
    initialAlleleA = int(np.trunc(N/2*0.0025))
    if(initialAlleleA <1):
        initialAlleleA =1
    indexesOfAlleleA = np.random.randint(minVal,maxVal,size = initialAlleleA,dtype = np.int16)
    Population_2[indexesOfAlleleA] = 0
    previousAverage = 0
    average = 100000
    time = 0

    #Storing Dynamics of Allels
    sumaaInPopulation_1 = []
    sumAAInPopulation_1 = []
    sumAaInPopulation_1 = []
    sumaaInPopulation_2 = []
    sumAAInPopulation_2 = []
    sumAaInPopulation_2 = []

    while(abs(previousAverage-average)>1):
        previousAverage = average
        for t in range(100):

            Population_1,Population_2 = Migration(Population_1, Population_2, m, N)
            virgensPopulation_1 = Selection(Population_1,s,N)
            Population_1 = Mating(virgensPopulation_1[0,:],virgensPopulation_1[1,:],N)
            virgensPopulation_2 = Selection(Population_2, s, N)
            Population_2 = Mating(virgensPopulation_2[0, :], virgensPopulation_2[1, :], N)
            sumaaInPopulation_1.append(np.sum(Population_1[Population_1==1]))
            sumAAInPopulation_1.append(abs(np.sum(Population_1[Population_1==-1])))
            sumAaInPopulation_1.append(individualPopulationSize-sumAAInPopulation_1[time]-sumaaInPopulation_1[time])
            sumaaInPopulation_2.append(np.sum(Population_2[Population_2 == 1]))
            sumAAInPopulation_2.append(abs(np.sum(Population_2[Population_2 == -1])))
            sumAaInPopulation_2.append(individualPopulationSize - sumAAInPopulation_2[time] - sumaaInPopulation_2[time])
            time += 1
        average = np.sum(Population_1)  + np.sum(Population_2)
        #check for inf loop
        if(time > 100):
            break
    rad1 = np.asarray(sumaaInPopulation_1)
    rad2 = np.asarray(sumAAInPopulation_1)
    rad3 = np.asarray(sumAaInPopulation_1)
    rad4 = np.asarray(sumaaInPopulation_2)
    rad5 = np.asarray(sumAAInPopulation_2)
    rad6 = np.asarray(sumAaInPopulation_2)
    radA1 = rad2*2+rad3
    radA2 = rad5*2+rad6
    returnData = np.array([[rad1],[rad2],[rad3],[rad4],[rad5],[rad6],[radA1],[radA2]])
    return returnData

# Theoretical plot
def pFixAnalytical(S,N):

    return (1-np.exp(-S))/(1-np.exp(-2*S*N))

def main():
    s = np.array([5*10**-2,0.1],dtype=np.float16)
    m = np.array([0.01,0.1],dtype=np.float16)
    N = 400
    numberOfRepeats = 5000
    num = 1
    for mValue in m:
        for sValue in s:
            num_cores = multiprocessing.cpu_count()
            print(num_cores)
            returnData = Parallel(n_jobs=num_cores)(delayed(Sim)(sValue,mValue,N) for repeat in range(numberOfRepeats))

            for p in range(numberOfRepeats):
                Ploting.Plot(returnData[p],uppgift = '2b',figure=num,m=mValue,s=sValue)
            num += 1
            for p in range(numberOfRepeats):
                filterdSignals = FA.FilterdSignals(returnData[p],lB=0,hB=0.1)
                Ploting.Plot(filterdSignals,uppgift = '2b',figure=num,m=mValue,s=sValue)
            num += 1
            returnDataAvrage=np.zeros(returnData[0].shape)
            count = 0
            for p in range(numberOfRepeats):
                temp = returnData[p]
                frequenceAPop2 = temp[7,:]
                condition = frequenceAPop2[0,70]>300 and frequenceAPop2[0,199]<300
                if(condition):
                    returnDataAvrage += returnData[p]
                    count += 1
            returnDataAvrage /=count
            Ploting.Plot(returnDataAvrage,uppgift = '2b',figure=num,m=mValue,s=sValue)
            num += 1

    plt.show()



if __name__ == '__main__':
    main()