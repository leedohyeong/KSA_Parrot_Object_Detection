from random import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd
import numpy as np

def search_csv(dirname):
    list = []
    df = pd.DataFrame()
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.csv':
                list.append(path+'\\'+filename)
    for i in list:
        data = pd.read_csv(i)
        df = pd.concat([df, data])
    return df


data = search_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject1\\dataset")
data=data.iloc[np.random.permutation(data.index)].reset_index(drop=True)
data = data.to_numpy()

print('____________________________________')
train_csv = pd.DataFrame(data[:-7500][:])
validation_csv = pd.DataFrame(data[-7500:-2500][:])
tset_csv = pd.DataFrame(data[23696:][:])
print(len(train_csv))
print(len(validation_csv))

# tset_csv.to_csv('tset_csv.csv', index=False, encoding='cp949')
# validation_csv.to_csv('validation_csv.csv', index=False, encoding='cp949')
# train_csv.to_csv('train_csv.csv', index=False, encoding='cp949')
