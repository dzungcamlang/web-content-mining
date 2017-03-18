# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:23:27 2016

@author: Yeong-su Kim
"""
import os, csv
import numpy as np
import random



### Description <csv_data_load_from_multiple_folders> ###
# input format is 1D array, output format is 2D array
# csv data, which is a file not a folder, loading from multiple folders, each of which have various depth (folder가 어떤 depth를 가져도 상관없다.)
def csv_data_load_from_multiple_folders(path, list_for_storing, token): # input format is 1D list
    
    number_of_file = 0
    # 이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
    for dirName, subdirList, fileList in os.walk(path):
        #print('Found directory: %s' % dirName)
        for fname in fileList:
            
            # 파일의 절대 경로를 만들어 주기 위한 코드
            file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.csv
  
            temp_str = dirName.split('\\')
            publisher = temp_str[-1] # 현재 publisher 폴더이름

        
            # Trick for inner/outer test
            if token==0:
                if list(publisher)[-1]=='1': break
            elif token==1:
                if not list(publisher)[-1]=='1': break
                    
                    
        
        
            with open(file_path, 'rbU') as f: # 'rb'하면 에러.. 왜..?
                reader = csv.reader(f)
                list_for_storing += list(reader)
                number_of_file += 1
    
#    print "[ csv_data_load_from_multiple_folders complete ]"
#    print('#of_files:',number_of_file, '#of_leafs:',len(list_for_storing), 'from data_manipulation_library.py')           
    return list_for_storing # output: 2D list


    
    
    
def sample_csv_data_for_aug_publisher(path, list_for_storing, random_list, test_or_train, n_of_aug): # input format is 1D list
    
    print test_or_train
    list_for_storing = []
    number_of_file = 0
    parent_folder = -2
    
    # 이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
    for dirName, subdirList, fileList in os.walk(path):
        
        
        # parent folder를 제외시키기 위해
        parent_folder += 1
        if parent_folder==-1: continue     
        if random_list[parent_folder]==test_or_train: continue
        if random_list[parent_folder]==n_of_aug: continue
        print('Found directory: %s' % dirName)

                
        for fname in fileList:
    
            # 파일의 절대 경로를 만들어 주기 위한 코드
            file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.csv
  
            temp_str = dirName.split('\\')
            publisher = temp_str[-1] # 현재 publisher 폴더이름

        
            with open(file_path, 'rbU') as f: # 'rb'하면 에러.. 왜..?
                reader = csv.reader(f)
                list_for_storing += list(reader)
                number_of_file += 1
    
 #   print "[ csv_data_load_from_multiple_folders complete ]"
 #   print('#of_files:',number_of_file, '#of_leafs:',len(list_for_storing), 'from data_manipulation_library.py')           
    
    return list_for_storing # output: 2D list       
    
    
    
    
    
    
    
    
def sample_csv_data_load_from_multiple_folders(path, list_for_storing, random_list, test_or_train): # input format is 1D list
    
    print test_or_train
    list_for_storing = []
    number_of_file = 0
    parent_folder = -2
    
    # 이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
    for dirName, subdirList, fileList in os.walk(path):
        
        
        # parent folder를 제외시키기 위해
        parent_folder += 1
        if parent_folder==-1: continue     
        if random_list[parent_folder]==test_or_train: continue
        print('Found directory: %s' % dirName)

                
        for fname in fileList:
    
            # 파일의 절대 경로를 만들어 주기 위한 코드
            file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.csv
  
            temp_str = dirName.split('\\')
            publisher = temp_str[-1] # 현재 publisher 폴더이름

        
            with open(file_path, 'rbU') as f: # 'rb'하면 에러.. 왜..?
                reader = csv.reader(f)
                list_for_storing += list(reader)
                number_of_file += 1
    
 #   print "[ csv_data_load_from_multiple_folders complete ]"
 #   print('#of_files:',number_of_file, '#of_leafs:',len(list_for_storing), 'from data_manipulation_library.py')           
    
    return list_for_storing # output: 2D list    
    
        
    
def new_load_dataset(path, random_list, test_or_train): # input format is 1D list
    
    print test_or_train
    list_for_storing = []
    number_of_file = 0
    parent_folder = -2
    
    # 이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
    for dirName, subdirList, fileList in os.walk(path):
        
        
        # parent folder를 제외시키기 위해
        parent_folder += 1
        if parent_folder==-1: continue     
        if random_list[parent_folder]==test_or_train: continue
        print('Found directory: %s' % dirName)

                
        for fname in fileList:
    
            # 파일의 절대 경로를 만들어 주기 위한 코드
            file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.csv
  
            temp_str = dirName.split('\\')
            publisher = temp_str[-1] # 현재 publisher 폴더이름
        
            with open(file_path, 'rbU') as f: # 'rb'하면 에러.. 왜..?
                reader = csv.reader(f)
                list_for_storing += list(reader)
                number_of_file += 1
    
       
    
    return list_for_storing # output: 2D list  
    
    

    
def mean_normalization(x, mean, std):
    if std==0: return 0
    else: return (x-mean)/std    
    


    



### Description <feature_scaling> ###
# input list & output list are the same format (2D Array)
# input idx_start, idx_end are array index, ex) 2 means array[2], that is 3th position not 2nd position of array
def feature_scaling(list2D, idx_start, idx_end): 
    
    total = idx_end-idx_start+1 # for dynamic length allocation
    temp2D = np.full((total, len(list2D)), 0) # initialize temp 2D list
    mean, std = [], []

    a=0
    print len(list2D)
    # 일딴 각 dimension의 feature 값들을 vector단위로 각각 저장, 평균과 평균오차 등을 원활히 계산하기 위한 과정
    for i, row in enumerate(temp2D):
        for j, element in enumerate(row):
            temp2D[i][j] = float(list2D[j][i+idx_start]) # 혹시나 string이 있으면 int형으로 처리, int형을 int형으로 처리해도 상관없다.

    # 평균/평균오차 계산
    for row in temp2D:
        mean.append(np.mean(row)) # 각각의 feature 평균값을 mean vector에 저장
        std.append(np.std(row)) # 각각의 feature 평균오차값을 std vector에 저장

    # Normalized값으로 치환
    for i, row in enumerate(temp2D):
        for j, element in enumerate(row):
            list2D[j][i+idx_start] = mean_normalization(float(list2D[j][i+idx_start]), mean[i], std[i])
            
            
            
            
###### 원시적인 버젼
#     data1, data2, data3, data4, data5, data6, data7, data8, data9 = [], [], [], [], [], [], [], [], []
#     for row in list2D:
#         data1.append(int(row[1]))
#         data2.append(int(row[2]))
#         data3.append(int(row[3]))
#         data4.append(int(row[4]))
#         data5.append(int(row[5]))
#         data6.append(int(row[6])) 
#         data7.append(int(row[7]))
#         data8.append(int(row[8]))
#         #data9.append(int(row[9]))
        
#     data1_mean = np.mean(data1)
#     data2_mean = np.mean(data2)
#     data3_mean = np.mean(data3)
#     data4_mean = np.mean(data4)
#     data5_mean = np.mean(data5)
#     data6_mean = np.mean(data6)
#     data7_mean = np.mean(data7)
#     data8_mean = np.mean(data8)
#     #data9_mean = np.mean(data9)
    
#     data1_std = np.std(data1)
#     data2_std = np.std(data2)
#     data3_std = np.std(data3)
#     data4_std = np.std(data4)
#     data5_std = np.std(data5)
#     data6_std = np.std(data6)
#     data7_std = np.std(data7)
#     data8_std = np.std(data8)
#     #data9_std = np.std(data9)


#     for row in list2D:
#         row[1] = mean_normalization(float(row[1]), data1_mean, data1_std)
#         row[2] = mean_normalization(float(row[2]), data2_mean, data2_std)
#         row[3] = mean_normalization(float(row[3]), data3_mean, data3_std)
#         row[4] = mean_normalization(float(row[4]), data4_mean, data4_std)
#         row[5] = mean_normalization(float(row[5]), data5_mean, data5_std)
#         row[6] = mean_normalization(float(row[6]), data6_mean, data6_std)
#         row[7] = mean_normalization(float(row[7]), data7_mean, data7_std)
#         row[8] = mean_normalization(float(row[8]), data8_mean, data8_std)
#         #row[9] = mean_normalization(float(row[9]), data9_mean, data9_std) 

    print "[ feature_scaling complete ]"
    return list2D
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
