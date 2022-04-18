import os
import numpy as np
import csv
os.environ['PROJ_LIB'] = r'C:\Anaconda4\envs\snap\Library\share'

output_csv = r'D:\CNN_storage\CSV_output\params.csv'


#Folder where input data is stored
lista = os.listdir(r'D:\CNN_storage\Balanced_dataset_sep_2021_mini')


with open(r'D:\CNN_storage\CSV_output\params.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    for i in lista:
        q = i.split('.tif')[0]
        p = q.split('_')

        try:
            mean = float(p[5])
            var = float(p[6])
            azm = float(p[7])
            wave = float(p[8])
            wmdr = 0

            data = [mean, var, azm, wave, wmdr]
            writer.writerow(data)

        except Exception as e:
            print(str(e))



f.close()