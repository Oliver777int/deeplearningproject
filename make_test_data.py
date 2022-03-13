import csv
import numpy as np
import random

import torch
# TODO -> float32 and why
# TODO csv file to rows

def create_data(rows, cols):
    data = np.empty((rows,cols), dtype=float)
    for i in range(rows):
        for j in range(cols-1):
            data[i,j] = (i*random.random()+10)
        data[i,cols-1] = i+random.random()
   # data = np.transpose(data)
    #data = torch.from_numpy(data)
    return data
data = create_data(100, 4)

file_path = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\make_test_data_file1.csv'
with open(file_path, 'w', encoding='UTF8', newline="\n") as f:
    header = ['param1', 'param2', 'param3', 'labels']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
print(data)
