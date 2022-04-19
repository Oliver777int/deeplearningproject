import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
#os.environ['PROJ_LIB']= 'C:\Anaconda4\envs\snap\Library\share'
import pandas as pd
import shutil

#Folder where input data is stored
path=os.chdir('D:\CNN_storage\s1_output')
lista = os.listdir('D:\CNN_storage\s1_output')






waveheightarray = []
wavearray = []
aziarray = []
meanarray = []
vararray = []
count=0
for i in lista:
    q = i.split('.tif')[0]
    p = q.split('_')
    try:
        waveheight = int(10000000 * float(p[8]))

        wave=float(p[8])
        azimuth=float(p[7])
        mean=float(p[5])
        var=float(p[6])
        wavearray.append(wave)
        aziarray.append(azimuth)
        meanarray.append(mean)
        vararray.append(var)
        waveheightarray.append(waveheight)
    except Exception as e:
        print(str(e))
        count+=1



print('antal bilder som inte fungerade = ' + str(count))
#wavearray = wavearray[0:20000]
#vararray = vararray[0:20000]
#aziarray = aziarray[0:20000]
#meanarray = meanarray[0:20000]
plt.scatter(wavearray, vararray)
plt.title("Variance to waveheight relation")
plt.xlabel("Significant waveheight [m]")
plt.ylabel("Variance")
plt.show()
plt.scatter(wavearray, aziarray)
plt.title("Azimuth cutoff to waveheight relation")
plt.xlabel("Significant waveheight [m]")
plt.ylabel("Azimuth cutoff wavelenght [m]")
plt.show()
plt.scatter(wavearray, meanarray)
plt.title("Mean backscatter to waveheight relation")
plt.xlabel("Significant waveheight [m]")
plt.ylabel("Mean backscatter")
plt.show()

bins = np.linspace(0, 60000000, num=60)

Q = np.histogram(waveheightarray, bins=bins)
print(Q[0])
#print('0-100')
#print(np.sum(Q[0][20]))

a = [1, 2, 5, 4, 3, 6, 7, 8, 9]
a = np.array(a)
for i in range(len(a)):
    print(a[i])

print(np.shape(a))
#d = np.sort(a)
#np.asarray(a)
#b = np.asarray([2, 4])
#c = np.split(d, b)

#sortedwaveheightarray = np.sort(waveheightarray)
#subdividedwaveheightarray = np.split(sortedwaveheightarray, np.cumsum(Q[0]))
#finalwaveheightarray = []
#count1=0
#for j in range(0, 40):
#    current = subdividedwaveheightarray[j]
#    np.random.shuffle(current)
#    if len(current) > 100:
#        current = current[0:100]
#
#    for k in range(len(current)):
#        finalwaveheightarray.append(current[k])
#    print(np.size(finalwaveheightarray))
#    count1+=1
#    print(j)

#print(count1)
#print('The dataset will have ' + str(len(finalwaveheightarray)) + ' number of pixel images before subdivision')
#K = np.histogram(finalwaveheightarray, bins=bins)
#print(K[0])
#count2=0
#for i in lista:
#    q = i.split('.tif')[0]
#    p = q.split('_')
#    try:
#        waveheight = int(10000000 * float(p[8]))
#    except Exception as e:
#        print(str(e))
#
#    if waveheight in finalwaveheightarray:
#        filename = 'D:\CNN_storage\s1_output' + '\\' + str(i)
#        shutil.copy2(filename, 'D:\CNN_storage\Balanced_dataset_sep_2021_mini')
#        count2+=1
#        print('hej')

#print(count2)
print('klar')

plt.hist(waveheightarray, bins = bins)
plt.title("histogram")
plt.show()
#plt.hist(finalwaveheightarray, bins = bins)
#plt.title("histogram")
#plt.show()