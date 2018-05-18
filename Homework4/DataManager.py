import numpy as np

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