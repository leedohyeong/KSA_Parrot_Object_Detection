import shutil
from random import seed
from random import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

save_dir = './dataset/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def search(dirname):
    img_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                img_list.append(path+'/'+filename)
    return img_list

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


img_list1 = search("C:\\Users\\USER\\Desktop\\KSA\\module8\\work\\data")
csv_list1 = search_csv('C:\\Users\\USER\\Desktop\\KSA\\module8\\work\\data')
print(len(img_list1))
csv_list1 = csv_list1.sort_values(by='frame', ascending=True)
csv_list1 = csv_list1.reset_index(drop = True)

np_csv1 = csv_list1.to_numpy()

csv_list2 = csv_list1.copy()
csv_list3 = csv_list1.copy()
csv_list4 = csv_list1.copy()
csv_list5 = csv_list1.copy()
csv_list6 = csv_list1.copy()


csv_list2['frame'] = csv_list1['frame'].str.replace('.jpg','bright.jpg',regex=True)
csv_list3['frame'] = csv_list2['frame'].str.replace('bright.jpg','rotate.jpg',regex=True)
csv_list4['frame'] = csv_list3['frame'].str.replace('rotate.jpg','trans.jpg',regex=True)
csv_list5['frame'] = csv_list4['frame'].str.replace('trans.jpg','zoomin_trans.jpg',regex=True)
csv_list6['frame'] = csv_list5['frame'].str.replace('zoomin_trans.jpg','zoomin_rotate.jpg',regex=True)

num = csv_list1['frame'].to_numpy()
num2 = csv_list2['frame'].to_numpy()
num3 = csv_list3['frame'].to_numpy()
num4 = csv_list4['frame'].to_numpy()
num5 = csv_list5['frame'].to_numpy()
num6 = csv_list6['frame'].to_numpy()

csv_list1.to_csv('origin.csv', index=False, encoding='cp949')
csv_list2.to_csv('bright.csv', index=False, encoding='cp949')
csv_list3.to_csv('rotate.csv', index=False, encoding='cp949')
csv_list4.to_csv('trans.csv', index=False, encoding='cp949')
csv_list5.to_csv('zoomin_trans.csv', index=False, encoding='cp949')
csv_list6.to_csv('zoomin_rotate.csv', index=False, encoding='cp949')

def reset() :
    global xmin, xmax, ymin, ymax

    xmin = np_csv1[:, 1].copy()
    xmax = np_csv1[:, 2].copy()
    ymin = np_csv1[:, 3].copy()
    ymax = np_csv1[:, 4].copy()

    return xmin, xmax, ymin, ymax

reset()

img = []
img_rotate = []
img_bright = []
img_trans = []
img_zoomin_trans = []
img_zoomin_rotate = []
img_gray = []
csv_rotate = []
csv_trans = []
csv_zoomin_rotate = []
csv_zoomin_trans = []


for i in range(len(img_list1)):
    rotate_val = random.randrange(-20,20)
    bright_val = random.randrange(100)
    trans_val = random.randrange(-20,20)
    trans_val2 = random.randrange(-20,20)
    matrix_dot = np.array([[xmin[i],ymin[i],1],[xmax[i],ymax[i],1]])
    matrix_dot2 = np.array([[xmax[i], ymin[i],1], [xmin[i], ymax[i],1]])

    #orginal
    img.append(cv2.imread(img_list1[i], cv2.IMREAD_COLOR)) # img = [[리스트1_1그림],[리스트1_2그림],[리스트1_3그림],....]
    height, width, channel = img[i].shape
    cv2.rectangle(img[i], (xmin[i], ymin[i]), (xmax[i], ymax[i]), (0, 0, 0), 1)
    # cv2.imwrite(save_dir+num[i], img[i])

    #rotation
    matrix = cv2.getRotationMatrix2D((width/2, height/2), rotate_val, 1)
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_width = int(height*cos+width*sin)
    new_height = int(height*sin+width*cos)
    matrix[0, 2] += (new_width / 2) - width/2
    matrix[1, 2] += (new_height / 2) - height/2
    img_rotate.append(cv2.warpAffine(img[i], matrix, (new_width, new_height)))
    img_rotate[i]=(cv2.resize(img_rotate[i],dsize=(width, height)))
    scale = new_width/width

    matrix_dot_x = (matrix_dot[0,0] + matrix_dot[1,0])/2
    matrix_dot_y = (matrix_dot[1, 1] + matrix_dot[0, 1]) / 2
    matrix_dot_m = cv2.getRotationMatrix2D((width/2, height/2), rotate_val, 1)

    matrix_dot = np.dot(matrix_dot_m,matrix_dot.T)
    matrix_dot2 = np.dot(matrix_dot_m,matrix_dot2.T)
    matrix_dot[0, : ] += ((new_width / 2) - width/2)
    matrix_dot[1, :] += ((new_height / 2) - height/2)

    matrix_dot2[0, : ] += ((new_width / 2) - width/2)
    matrix_dot2[1, :] += ((new_height / 2) - height/2)
    matrix_dot /= scale
    matrix_dot2 /= scale
    matrix_dot= matrix_dot.astype('int32')
    matrix_dot2 = matrix_dot2.astype('int32')

    xmin[i] = int(min(matrix_dot[0,0],matrix_dot2[0,0],matrix_dot[0,1],matrix_dot2[0,1]))
    xmax[i] = int(max(matrix_dot[0,0],matrix_dot2[0,0],matrix_dot[0,1],matrix_dot2[0,1]))
    ymin[i] = int(min(matrix_dot[1,0],matrix_dot2[1,0],matrix_dot[1,1],matrix_dot2[1,1]))
    ymax[i] = int(max(matrix_dot[1,0],matrix_dot2[1,0],matrix_dot[1,1],matrix_dot2[1,1]))

    cv2.rectangle(img_rotate[i], (xmin[i], ymin[i]), (xmax[i], ymax[i]), (255, 0, 0), 1)
    csv_rotate.append([xmin[i],xmax[i],ymin[i],ymax[i]])
    csv_zoomin_rotate.append([xmin[i],xmax[i],ymin[i],ymax[i]])
    # plt.figure(figsize=(16, 16))
    # plt.imshow(img_rotate[i])
    # cv2.imwrite(save_dir+num3[i], img_rotate[i])
    reset()



    #translation
    M = np.float32([[1,0,trans_val],[0,1,trans_val2]])
    img_trans.append(cv2.warpAffine(img[i], M, (width+trans_val, height+trans_val2)))
    width_trans, height_trans, channel = img_trans[i].shape
    scale_x = round(width_trans / width, 3)
    scale_y = round(height_trans / height, 3)

    img_trans[i] = (cv2.resize(img_trans[i], dsize=(width, height)))

    xmin[i] = int((xmin[i] + trans_val)/scale_y)
    ymin[i] = int((ymin[i] + trans_val2)/scale_x)
    xmax[i] = int((xmax[i] + trans_val)/scale_y)
    ymax[i] = int((ymax[i] + trans_val2)/scale_x)

    cv2.rectangle(img_trans[i], (xmin[i], ymin[i]), (xmax[i], ymax[i]), (255, 0, 0), 1)

    csv_trans.append([xmin[i],xmax[i],ymin[i],ymax[i]])
    csv_zoomin_trans.append([xmin[i],xmax[i],ymin[i],ymax[i]])
    # plt.figure(figsize=(16, 16))
    # plt.imshow(img_trans[i])
    reset()
    # cv2.imwrite(save_dir+num4[i], img_trans[i])

    #bright
    img_bright.append(cv2.add(img[i], (bright_val,bright_val,bright_val,0)))
    # plt.figure(figsize=(16, 16))
    # plt.imshow(img_bright[i])
    # cv2.imwrite(save_dir+num2[i], img_bright[i])



for i in range(len(img_list1)):
    height, width, channel = img[i].shape
    zoomin_val = random.randrange(300,600)

    #zoomout translation1
    img_zoomin_trans.append(img_trans[i])
    scale_zoomin = (zoomin_val - height) / 2

    M = np.float32([[1, 0, scale_zoomin], [0, 1, scale_zoomin]])
    img_zoomin_trans[i] = cv2.warpAffine(img_zoomin_trans[i], M, (zoomin_val, zoomin_val))
    scale_zoomin2 = round(zoomin_val/height,2)
    img_zoomin_trans[i] = (cv2.resize(img_zoomin_trans[i],dsize=(width,height)))
    csv_zoomin_trans[i][0] = int((csv_zoomin_trans[i][0]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_trans[i][1] = int((csv_zoomin_trans[i][1]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_trans[i][2] = int((csv_zoomin_trans[i][2]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_trans[i][3] = int((csv_zoomin_trans[i][3]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_trans[i].append([csv_zoomin_trans[i][0],csv_zoomin_trans[i][2],csv_zoomin_trans[i][1],csv_zoomin_trans[i][3]])
    cv2.rectangle(img_zoomin_trans[i], (csv_zoomin_trans[i][0],csv_zoomin_trans[i][2]),(csv_zoomin_trans[i][1],csv_zoomin_trans[i][3]), (255, 255, 255), 1)
    # plt.figure(figsize=(16,16))
    # plt.imshow(img_zoomin_trans[i])
    # cv2.imwrite(save_dir+num5[i], img_zoomin_trans[i])

    #zoomout rotation
    img_zoomin_rotate.append(img_rotate[i])
    img_zoomin_rotate[i] = cv2.warpAffine(img_zoomin_rotate[i], M, (zoomin_val, zoomin_val))
    img_zoomin_rotate[i] = (cv2.resize(img_zoomin_rotate[i], dsize= (width,height)))
    csv_zoomin_rotate[i][0] = int((csv_zoomin_rotate[i][0]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_rotate[i][1] = int((csv_zoomin_rotate[i][1]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_rotate[i][2] = int((csv_zoomin_rotate[i][2]+scale_zoomin)/scale_zoomin2)
    csv_zoomin_rotate[i][3] = int((csv_zoomin_rotate[i][3]+scale_zoomin)/scale_zoomin2)
    cv2.rectangle(img_zoomin_rotate[i], (csv_zoomin_rotate[i][0],csv_zoomin_rotate[i][2]),(csv_zoomin_rotate[i][1],csv_zoomin_rotate[i][3]), (255, 255, 255), 1)
    csv_zoomin_rotate.append([csv_zoomin_rotate[i][0],csv_zoomin_rotate[i][2],csv_zoomin_rotate[i][1],csv_zoomin_rotate[i][3]])
    # cv2.imwrite(save_dir+num6[i], img_zoomin_rotate[i])
    # plt.figure(figsize=(16, 16))
    # plt.imshow(img_zoomin_rotate[i])

csv_trans = pd.DataFrame(csv_trans)
csv_rotate = pd.DataFrame(csv_rotate)
csv_zoomin_rotate = pd.DataFrame(csv_zoomin_rotate)
csv_zoomin_trans = pd.DataFrame(csv_zoomin_trans)
csv_trans.to_csv('csv_trans.csv', index=False, encoding='cp949')
csv_rotate.to_csv('csv_rotate.csv', index=False, encoding='cp949')
csv_zoomin_rotate.to_csv('csv_zoomin_rotate.csv', index=False, encoding='cp949')
csv_zoomin_trans.to_csv('csv_zoomin_trans.csv', index=False, encoding='cp949')
# plt.show()