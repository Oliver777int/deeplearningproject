import csv
import numpy as np
import random

import torch
# TODO -> float32 and why
# TODO csv file to rows

def create_data(rows, cols):
    x = 100
    num = int(rows / x)
    data = np.empty((rows, cols), dtype=float)
    for k in range(num):
        for i in range(x):
            for j in range(cols-1):
                data[i+k*x,j] = (i + 0.5*random.random())
            data[i+k*x,cols-1] = i
       # data = np.transpose(data)
        #data = torch.from_numpy(data)ds
    return data
train_data = create_data(1000, 4)
validation_data = create_data(1000,4)

# file_path_train = r'C:\Users\pj2m1s\Simon\skola\deeplearningproject\train_data_file1.csv'
# file_path_validation = r'C:\Users\pj2m1s\Simon\skola\deeplearningproject\validation_data_file1.csv'

file_path_train = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\train_data_file1.csv'
file_path_validation = r'C:\Users\User\OneDrive\Skola\KEX\deeplearningproject\validation_data_file1.csv'

with open(file_path_train, 'w', encoding='UTF8', newline="\n") as f:
    header = ['param1', 'param2', 'param3', 'labels']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(train_data)

with open(file_path_validation, 'w', encoding='UTF8', newline="\n") as f:
    header = ['param1', 'param2', 'param3', 'labels']
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(validation_data)