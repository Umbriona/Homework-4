
import numpy as np
from scipy.fftpack import rfft, irfft, fftfreq

def BandFilter(signal,time=1,lowerBound=0,higherBound=100):
    #time  = np.linspace(0,10,2000)
    #signal = np.cos(5*np.pi*time) + np.cos(7*np.pi*time)

    W = fftfreq(signal.size, d=time)
    f_signal = rfft(signal)

    # If our original signal time was in seconds, this is now in Hz
    cut_f_signal = f_signal.copy()
    #print('W',W,'\ncutSignal',cut_f_signal)
    cut_f_signal[(W < lowerBound)] = 0
    cut_f_signal[(W > higherBound)] = 0

    cut_signal = irfft(cut_f_signal)
    return cut_signal

def FilterdSignals(rawSignals,lB=0,hB=100):

    aaSignalPopulation_1 = rawSignals[0, :]
    AASignalPopulation_1 = rawSignals[1, :]
    AaSignalPopulation_1 = rawSignals[2, :]
    aaSignalPopulation_2 = rawSignals[3, :]
    AASignalPopulation_2 = rawSignals[4, :]
    AaSignalPopulation_2 = rawSignals[5, :]
    ASignal1 = rawSignals[6, :]
    ASignal2 = rawSignals[7, :]
    filterSignalPopulation1aa = BandFilter(aaSignalPopulation_1[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterSignalPopulation1AA = BandFilter(AASignalPopulation_1[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterSignalPopulation1Aa = BandFilter(AaSignalPopulation_1[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterSignalPopulation2aa = BandFilter(aaSignalPopulation_2[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterSignalPopulation2AA = BandFilter(AASignalPopulation_2[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterSignalPopulation2Aa = BandFilter(AaSignalPopulation_2[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterASignal1 = BandFilter(ASignal1[0, :], time=1, lowerBound=lB, higherBound=hB)
    filterASignal2 = BandFilter(ASignal2[0, :], time=1, lowerBound=lB, higherBound=hB)
    rad1 = np.asarray(filterSignalPopulation1aa)
    rad2 = np.asarray(filterSignalPopulation1AA)
    rad3 = np.asarray(filterSignalPopulation1Aa)
    rad4 = np.asarray(filterSignalPopulation2aa)
    rad5 = np.asarray(filterSignalPopulation2AA)
    rad6 = np.asarray(filterSignalPopulation2Aa)
    rad7 = np.asarray(filterASignal1)
    rad8 = np.asarray(filterASignal2)
    returnData = np.array([[rad1], [rad2], [rad3], [rad4], [rad5], [rad6],[rad7],[rad8]])
    return returnData