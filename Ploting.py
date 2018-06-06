import matplotlib.pyplot as plt
import numpy as np
def Plot (data, uppgift = '1a',figure=1,m=0.1,s=0.1):

    if(uppgift == '2a'):
        sumaaInPopulation_1 = data[0,:]
        sumAAInPopulation_1 = data[1,:]
        sumAaInPopulation_1 = data[2,:]
        sumaaInPopulation_2 = data[3,:]
        sumAAInPopulation_2 = data[4,:]
        sumAaInPopulation_2 = data[5,:]
        lengthData = sumaaInPopulation_1.shape[1]
        X = np.linspace(0,lengthData-1,lengthData)
        #print('X',X,'\naa',sumaaInPopulation_1[0,:])
        plt.figure(figure)
        strFigureTitle = 'Migration value = '+str(m)+' s = '+str(s)
        plt.suptitle(strFigureTitle, fontsize=16)

        plt.subplot(2, 3, 1)
        plt.plot(X, sumaaInPopulation_1[0,:], '-')
        plt.title('Genotype aa Population 1')


        plt.subplot(2, 3, 2)
        plt.plot(X, sumAAInPopulation_1[0,:], '-')
        plt.title('Genotype AA Population 1')


        plt.subplot(2, 3, 3)
        plt.plot(X, sumAaInPopulation_1[0,:], '-')
        plt.title('Genotype Aa Population 1')


        plt.subplot(2, 3, 4)
        plt.plot(X, sumaaInPopulation_2[0,:], '-')
        plt.title('Genotype AA Population 2')


        plt.subplot(2, 3, 5)
        plt.plot(X, sumAAInPopulation_2[0,:], '-')
        plt.title('Genotype aa Population 2')


        plt.subplot(2, 3, 6)
        plt.plot(X, sumAaInPopulation_2[0,:], '-')
        plt.title('Genotype Aa Population 2')

    if (uppgift == '2b'):
        frequencyAPopulation1 = data[6, :]
        frequencyAPopulation2 = data[7, :]

        lengthData = data[6, :].shape[1]
        X = np.linspace(0, lengthData - 1, lengthData)
        # print('X',X,'\naa',sumaaInPopulation_1[0,:])
        plt.figure(figure)
        strFigureTitle = 'Migration value = ' + str(m) + ' s = ' + str(s)
        plt.suptitle(strFigureTitle, fontsize=16)

        plt.subplot(2, 1, 1)
        plt.plot(X, frequencyAPopulation1[0, :], '-')
        plt.title('A frequency Population 1')

        plt.subplot(2, 1, 2)
        plt.plot(X, frequencyAPopulation2[0, :], '-')
        plt.title('A frequency Population 2')
