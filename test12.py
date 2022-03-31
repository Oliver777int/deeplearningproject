import numpy as np
import matplotlib.pyplot as plt
import os
#os.environ['PROJ_LIB']= 'C:\Anaconda4\envs\snap\Library\share'
import pandas as pd
import shutil

#Folder where input data is stored
path=os.chdir('D:\Sar_download3')
lista = os.listdir('D:\Sardownload3')

waveheightarray = []
count=0
for i in lista:
    q = i.split('.tif')[0]
    p = q.split('')
    try:
        waveheight = int(10000000 * float(p[5]))
        waveheightarray.append(waveheight)
    except Exception as e:
        print(str(e))
        count+=1



print('antal bilder som inte fungerade = ' + str(count))

bins = np.linspace(0, 60000000, num=60)

Q = np.histogram(waveheightarray, bins=bins)
print(Q[0])
#print('0-100')
#print(np.sum(Q[0][20]))

#a = [1, 2, 5, 4, 3, 6, 7, 8, 9]
#d = np.sort(a)
#np.asarray(a)
#b = np.asarray([2, 4])
#c = np.split(d, b)

sortedwaveheightarray = np.sort(waveheightarray)
subdividedwaveheightarray = np.split(sortedwaveheightarray, np.cumsum(Q[0]))
finalwaveheightarray = []
count1=0

for j in range(3, 22):
    current = subdividedwaveheightarray[j]
    np.random.shuffle(current)
    current = current[0:173]

    for k in range(len(current)):
        finalwaveheightarray.append(current[k])
    #print(np.size(finalwaveheightarray)-176*(j-3))
    count1+=1
    #print(j)

print(count1)
print('The dataset will have ' + str(len(finalwaveheightarray)) + ' number of pixel images before subdivision')
count2=0
#for i in lista:
#    q = i.split('.tif')[0]
#    p = q.split('_')
#    waveheight = int(10000000 * float(p[5]))
#    if waveheight in finalwaveheightarray:
#        filename = 'D:\Sar_download' + '\' + str(i)
#        shutil.copy2(filename, 'D:\Sar_download3')
#        count2+=1

print(count2)
print('klar')

plt.hist(waveheightarray, bins = bins)
plt.title("histogram")
plt.show()