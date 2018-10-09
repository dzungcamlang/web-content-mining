# -*- coding: utf-8 -*-
# 데이터를 로드, 샘플링, 프린트 등을 하기 위한 함수들이 정의되어 있는 파일
# 데이터 처리를 위한 함수들의 집합소

import random

import os, csv
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

global labeled_dataset
labeled_dataset = []
number_of_file = 0


def load_data(dataset_path, num_of_test_sample):
    
    # Sampling process
    group_of_items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
    num_to_select = num_of_test_sample # test set을 구성할 언론사 폴더 개수                          
    list_of_random_items = random.sample(group_of_items, num_to_select)
    random_list = [0]*len(group_of_items)

    for j, ele in enumerate(list_of_random_items):
        for i, num in enumerate(group_of_items):
            if num==ele:
                random_list[i]=1
                
    # Load data from multiple folders while spliting train/test
    labeled_train_dataset = sample_csv_data_load_from_multiple_folders(dataset_path, labeled_dataset, random_list, 1) # train 0
    labeled_test_dataset = sample_csv_data_load_from_multiple_folders(dataset_path, labeled_dataset, random_list, 0) # train 1
    
    return labeled_train_dataset, labeled_test_dataset


def load_data_aug_publisher(dataset_path, num_of_test_sample, n_of_publisher):
    
    # Sampling process
    group_of_items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
    num_to_select = num_of_test_sample # test set을 구성할 언론사 폴더 개수                          
    list_of_random_items = random.sample(group_of_items, num_to_select)
    random_list = [0]*len(group_of_items)
    
    for j, ele in enumerate(list_of_random_items):
        for i, num in enumerate(group_of_items):
            if num==ele:
                random_list[i]=1
    
    # for assigning aug_publisher
    for i in list_of_random_items:
        group_of_items.remove(i)
    
    aug_n_of_publisher = n_of_publisher
    list_of_random_items = random.sample(group_of_items, aug_n_of_publisher)
    
    for j, ele in enumerate(list_of_random_items):
        for i, num in enumerate(group_of_items):
            if num==ele:
                random_list[num]=2
    
    # Load data from multiple folders while spliting train/test
    labeled_train_dataset = sample_csv_data_for_aug_publisher(dataset_path, labeled_dataset, random_list, 0, 1) # train 2
    labeled_test_dataset = sample_csv_data_for_aug_publisher(dataset_path, labeled_dataset, random_list, 0, 2) # test 1
    
    return labeled_train_dataset, labeled_test_dataset


def sampling_data_total(labeled_train_dataset, labeled_test_dataset, sample_percent, length_features, end_point_for_feature_scaling):
    # Data Load

    #### 왜 이런지 모르겠음
    # 혹시 필요할지도.. 20 length가 아니면 모두 filtering
    labeled_train_dataset = [row for row in labeled_train_dataset if len(row)==length_features]
    labeled_test_dataset = [row for row in labeled_test_dataset if len(row)==length_features]
    
    #####
    # Sampling
    labeled_train_dataset, class_1_test = train_test_split(labeled_train_dataset, train_size = sample_percent)
    labeled_test_dataset, class_2_test = train_test_split(labeled_test_dataset, train_size = sample_percent)
    #####
    
    # '[4' 이런식으로 찍히는거 '4'로 변환
    for row in labeled_train_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]
    for row in labeled_test_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]

#    print len(labeled_train_dataset)
#    print len(labeled_test_dataset)

    ### Filter dataset based on class
    train_class_1_dataset = [row for row in labeled_train_dataset if '1'==row[0] or '[1'== row[0]]
    train_class_2_dataset = [row for row in labeled_train_dataset if '2'==row[0] or '[2'== row[0]]
    train_class_3_dataset = [row for row in labeled_train_dataset if '3'==row[0] or '[3'== row[0]]
    train_class_4_dataset = [row for row in labeled_train_dataset if '4'==row[0] or '[4'== row[0]]

    test_class_1_dataset = [row for row in labeled_test_dataset if '1'==row[0] or '[1'== row[0]]
    test_class_2_dataset = [row for row in labeled_test_dataset if '2'==row[0] or '[2'== row[0]]
    test_class_3_dataset = [row for row in labeled_test_dataset if '3'==row[0] or '[3'== row[0]]
    test_class_4_dataset = [row for row in labeled_test_dataset if '4'==row[0] or '[4'== row[0]]

#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100)

    #########
    # Sampling

#    train_class_1_dataset, class_1_test = train_test_split(train_class_1_dataset, train_size = sample_percent)
#    train_class_2_dataset, class_2_test = train_test_split(train_class_2_dataset, train_size = sample_percent)
#    train_class_3_dataset, class_3_test = train_test_split(train_class_3_dataset, train_size = sample_percent)
#    train_class_4_dataset, class_4_test = train_test_split(train_class_4_dataset, train_size = sample_percent)

#    test_class_1_dataset, class_1_test = train_test_split(test_class_1_dataset, train_size = sample_percent)
#    test_class_2_dataset, class_2_test = train_test_split(test_class_2_dataset, train_size = sample_percent)
#    test_class_3_dataset, class_3_test = train_test_split(test_class_3_dataset, train_size = sample_percent)
#    test_class_4_dataset, class_4_test = train_test_split(test_class_4_dataset, train_size = sample_percent)

    #########

#    print "After Sampling"
#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100)

    ### Join dataset
    outer_train_dataset = train_class_1_dataset+train_class_2_dataset+train_class_3_dataset+train_class_4_dataset
    outer_test_dataset = test_class_1_dataset+test_class_2_dataset+test_class_3_dataset+test_class_4_dataset

    ### Shuffle dataset
    outer_shuf_train_dataset = shuffle(outer_train_dataset)
    outer_shuf_test_dataset = shuffle(outer_test_dataset)

    ### Split X and Y
    outer_X_train = []
    outer_Y_train = []
    outer_temp1 = []

    outer_X_test = []
    outer_Y_test = []
    outer_temp = []

    for idx, row in enumerate(outer_shuf_train_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp1.append(outer_shuf_train_dataset[idx][j])
        outer_X_train.append(outer_temp1)
        outer_temp1 = [] 

    for idx, row in enumerate(outer_shuf_train_dataset): outer_Y_train.append(outer_shuf_train_dataset[idx][0])


    for idx, row in enumerate(outer_shuf_test_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp.append(outer_shuf_test_dataset[idx][j])
        outer_X_test.append(outer_temp)
        outer_temp = [] 

    for idx, row in enumerate(outer_shuf_test_dataset): outer_Y_test.append(outer_shuf_test_dataset[idx][0])

    ### Convert to numpy array with data-type
    X_train = np.array(outer_X_train, dtype='float32')
    Y_train = np.array(outer_Y_train, dtype='int64')
    outer_X_test = np.array(outer_X_test, dtype='float32')
    outer_Y_test = np.array(outer_Y_test, dtype='int64')

    # for outer test
    test_X = outer_X_test
    test_y = outer_Y_test
    
    return X_train, Y_train, test_X, test_y


def sampling_data_class(labeled_train_dataset, labeled_test_dataset, sample_percent, length_features, end_point_for_feature_scaling):
    # Data Load

    #### 왜 이런지 모르겠음
    # 혹시 필요할지도.. 20 length가 아니면 모두 filtering
    labeled_train_dataset = [row for row in labeled_train_dataset if len(row)==length_features]
    labeled_test_dataset = [row for row in labeled_test_dataset if len(row)==length_features]
    
    # '[4' 이런식으로 찍히는거 '4'로 변환
    for row in labeled_train_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]
    for row in labeled_test_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]

#    print len(labeled_train_dataset)
#    print len(labeled_test_dataset)

    ### Filter dataset based on class
    train_class_1_dataset = [row for row in labeled_train_dataset if '1'==row[0] or '[1'== row[0]]
    train_class_2_dataset = [row for row in labeled_train_dataset if '2'==row[0] or '[2'== row[0]]
    train_class_3_dataset = [row for row in labeled_train_dataset if '3'==row[0] or '[3'== row[0]]
    train_class_4_dataset = [row for row in labeled_train_dataset if '4'==row[0] or '[4'== row[0]]

    test_class_1_dataset = [row for row in labeled_test_dataset if '1'==row[0] or '[1'== row[0]]
    test_class_2_dataset = [row for row in labeled_test_dataset if '2'==row[0] or '[2'== row[0]]
    test_class_3_dataset = [row for row in labeled_test_dataset if '3'==row[0] or '[3'== row[0]]
    test_class_4_dataset = [row for row in labeled_test_dataset if '4'==row[0] or '[4'== row[0]]

#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100)

    #########
    # Sampling
    train_class_1_dataset, class_1_test = train_test_split(train_class_1_dataset, train_size = sample_percent)
    train_class_2_dataset, class_2_test = train_test_split(train_class_2_dataset, train_size = sample_percent)
    train_class_3_dataset, class_3_test = train_test_split(train_class_3_dataset, train_size = sample_percent)
    train_class_4_dataset, class_4_test = train_test_split(train_class_4_dataset, train_size = sample_percent)

    test_class_1_dataset, class_1_test = train_test_split(test_class_1_dataset, train_size = sample_percent)
    test_class_2_dataset, class_2_test = train_test_split(test_class_2_dataset, train_size = sample_percent)
    test_class_3_dataset, class_3_test = train_test_split(test_class_3_dataset, train_size = sample_percent)
    test_class_4_dataset, class_4_test = train_test_split(test_class_4_dataset, train_size = sample_percent)

    #########
#    print "After Sampling"
#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100)
    
    ### Join dataset
    outer_train_dataset = train_class_1_dataset+train_class_2_dataset+train_class_3_dataset+train_class_4_dataset
    outer_test_dataset = test_class_1_dataset+test_class_2_dataset+test_class_3_dataset+test_class_4_dataset

    ### Shuffle dataset
    outer_shuf_train_dataset = shuffle(outer_train_dataset)
    outer_shuf_test_dataset = shuffle(outer_test_dataset)

    ### Split X and Y
    outer_X_train = []
    outer_Y_train = []
    outer_temp1 = []

    outer_X_test = []
    outer_Y_test = []
    outer_temp = []

    for idx, row in enumerate(outer_shuf_train_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp1.append(outer_shuf_train_dataset[idx][j])
        outer_X_train.append(outer_temp1)
        outer_temp1 = [] 

    for idx, row in enumerate(outer_shuf_train_dataset): outer_Y_train.append(outer_shuf_train_dataset[idx][0])

    for idx, row in enumerate(outer_shuf_test_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp.append(outer_shuf_test_dataset[idx][j])
        outer_X_test.append(outer_temp)
        outer_temp = [] 

    for idx, row in enumerate(outer_shuf_test_dataset): outer_Y_test.append(outer_shuf_test_dataset[idx][0])

    ### Convert to numpy array with data-type
    X_train = np.array(outer_X_train, dtype='float32')
    Y_train = np.array(outer_Y_train, dtype='int64')
    outer_X_test = np.array(outer_X_test, dtype='float32')
    outer_Y_test = np.array(outer_Y_test, dtype='int64')

    # for outer test
    test_X = outer_X_test
    test_y = outer_Y_test
    
    return X_train, Y_train, test_X, test_y


def no_sampling_data(labeled_train_dataset, labeled_test_dataset, length_features, end_point_for_feature_scaling):
    # Data Load

    #### 왜 이런지 모르겠음
    # 혹시 필요할지도.. 20 length가 아니면 모두 filtering
    labeled_train_dataset = [row for row in labeled_train_dataset if len(row)==length_features]
    labeled_test_dataset = [row for row in labeled_test_dataset if len(row)==length_features]
    # '[4' 이런식으로 찍히는거 '4'로 변환
    for row in labeled_train_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]
    for row in labeled_test_dataset:
        if list(row[0])[0]=='[': 
            row[0] = list(row[0])[1]

#    print len(labeled_train_dataset)
#    print len(labeled_test_dataset)

    ### Filter dataset based on class
    train_class_1_dataset = [row for row in labeled_train_dataset if '1'==row[0] or '[1'== row[0]]
    train_class_2_dataset = [row for row in labeled_train_dataset if '2'==row[0] or '[2'== row[0]]
    train_class_3_dataset = [row for row in labeled_train_dataset if '3'==row[0] or '[3'== row[0]]
    train_class_4_dataset = [row for row in labeled_train_dataset if '4'==row[0] or '[4'== row[0]]

    test_class_1_dataset = [row for row in labeled_test_dataset if '1'==row[0] or '[1'== row[0]]
    test_class_2_dataset = [row for row in labeled_test_dataset if '2'==row[0] or '[2'== row[0]]
    test_class_3_dataset = [row for row in labeled_test_dataset if '3'==row[0] or '[3'== row[0]]
    test_class_4_dataset = [row for row in labeled_test_dataset if '4'==row[0] or '[4'== row[0]]

#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100) 

#    print "After Sampling"
#    print '*** labeled_train_dataset ***'
#    print ('#class1:', len(train_class_1_dataset), '#class2:', len(train_class_2_dataset),
#            '#class3:', len(train_class_3_dataset), '#class4:', len(train_class_4_dataset))
#    nor = len(labeled_train_dataset)
#    print ('%class1:', len(train_class_1_dataset)/float(nor)*100, '%class2:', len(train_class_2_dataset)/float(nor)*100,
#            '%class3:', len(train_class_3_dataset)/float(nor)*100, '%class4:', len(train_class_4_dataset)/float(nor)*100)#
#
#    print '*** labeled_test_dataset ***'
#    print ('#class1:', len(test_class_1_dataset), '#class2:', len(test_class_2_dataset),
#            '#class3:', len(test_class_3_dataset), '#class4:', len(test_class_4_dataset))
#    nor = len(labeled_test_dataset)
#    print ('%class1:', len(test_class_1_dataset)/float(nor)*100, '%class2:', len(test_class_2_dataset)/float(nor)*100,
#            '%class3:', len(test_class_3_dataset)/float(nor)*100, '%class4:', len(test_class_4_dataset)/float(nor)*100)

    ### Join dataset
    outer_train_dataset = train_class_1_dataset+train_class_2_dataset+train_class_3_dataset+train_class_4_dataset
    outer_test_dataset = test_class_1_dataset+test_class_2_dataset+test_class_3_dataset+test_class_4_dataset

    ### Shuffle dataset
    outer_shuf_train_dataset = shuffle(outer_train_dataset)
    outer_shuf_test_dataset = shuffle(outer_test_dataset)

    ### Split X and Y
    outer_X_train = []
    outer_Y_train = []
    outer_temp1 = []

    outer_X_test = []
    outer_Y_test = []
    outer_temp = []

    for idx, row in enumerate(outer_shuf_train_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp1.append(outer_shuf_train_dataset[idx][j])
        outer_X_train.append(outer_temp1)
        outer_temp1 = [] 

    for idx, row in enumerate(outer_shuf_train_dataset): outer_Y_train.append(outer_shuf_train_dataset[idx][0])

    for idx, row in enumerate(outer_shuf_test_dataset):
        for j in range(1, end_point_for_feature_scaling+1):
            outer_temp.append(outer_shuf_test_dataset[idx][j])
        outer_X_test.append(outer_temp)
        outer_temp = [] 

    for idx, row in enumerate(outer_shuf_test_dataset): outer_Y_test.append(outer_shuf_test_dataset[idx][0])

    ### Convert to numpy array with data-type
    X_train = np.array(outer_X_train, dtype='float32')
    Y_train = np.array(outer_Y_train, dtype='int64')
    outer_X_test = np.array(outer_X_test, dtype='float32')
    outer_Y_test = np.array(outer_Y_test, dtype='int64')

    # for outer test
    test_X = outer_X_test
    test_y = outer_Y_test
    
    return X_train, Y_train, test_X, test_y


def SVM(X_train, Y_train, test_X, test_y):
    
    # Train the model 
    clf2 = SVC()
    clf2.fit(X_train, Y_train) 
    
    # Predict the model
    prediction_SVM = clf2.predict(test_X)
    
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print(classification_report(test_y, prediction_SVM, target_names=target_names))
       
    
def classification_result_report(a):
   
    c0p_list = []
    c0r_list = []
    c0f_list = []
    c0s_list = []

    c1p_list = []
    c1r_list = []
    c1f_list = []
    c1s_list = []

    c2p_list = []
    c2r_list = []
    c2f_list = []
    c2s_list = []

    c3p_list = []
    c3r_list = []
    c3f_list = []
    c3s_list = []

    c0p = a[72]+a[73]+a[74]+a[75]
    c0r = a[82]+a[83]+a[84]+a[85]
    c0f = a[92]+a[93]+a[94]+a[95]
    c0s = a[101]+a[102]+a[103]+a[104]+a[105]

    c1p = a[125]+a[126]+a[127]+a[128]
    c1r = a[135]+a[136]+a[137]+a[138]
    c1f = a[145]+a[146]+a[147]+a[148]
    c1s = a[154]+a[155]+a[156]+a[157]+a[158]

    c2p = a[178]+a[179]+a[180]+a[181]
    c2r = a[188]+a[189]+a[190]+a[191]
    c2f = a[198]+a[199]+a[200]+a[201]
    c2s = a[207]+a[208]+a[209]+a[210]+a[211]

    c3p = a[231]+a[232]+a[233]+a[234]
    c3r = a[241]+a[242]+a[243]+a[244]
    c3f = a[251]+a[252]+a[253]+a[254]
    c3s = a[260]+a[261]+a[262]+a[263]+a[264]
    
    return c0p_list, c0r_list, c0f_list, c0s_list, c1p_list, c1r_list, c1f_list, c1s_list, c2p_list, c2r_list, c2f_list, c2s_list, c3p_list, c3r_list, c3f_list, c3s_list
        

def string_to_float(li):
    for i, j in enumerate(li):
        li[i] = float(j)
    return li


def mean_std_print(l1, l2, l3, l4, k1, k2, k3, k4, m1, m2, m3, m4, n1, n2, n3, n4):
    
    l1 = string_to_float(l1)
    l2 = string_to_float(l2)
    l3 = string_to_float(l3)
    l4 = string_to_float(l4)
    
    k1 = string_to_float(k1)
    k2 = string_to_float(k2)
    k3 = string_to_float(k3)
    k4 = string_to_float(k4)
    
    m1 = string_to_float(m1)
    m2 = string_to_float(m2)
    m3 = string_to_float(m3)
    m4 = string_to_float(m4)
    
    n1 = string_to_float(n1)
    n2 = string_to_float(n2)
    n3 = string_to_float(n3)
    n4 = string_to_float(n4)
    
    print('\t\t', 'PRECISION avg.', '\t', 'RECALL avg.', '\t\t', 'F1-SCORE avg.')
    print('\n')
    print('TITLE:\t\t', "%0.2f" % (np.mean(l1)*100), "(%0.2f)" % (np.std(l1)*100), '\t\t', "%0.2f" % (np.mean(l2)*100), "(%0.2f)" % (np.std(l2)*100), '\t\t', "%0.2f" % (np.mean(l3)*100), "(%0.2f)" % (np.std(l3)*100))
    print('DATE:\t\t', "%0.2f" % (np.mean(k1)*100), "(%0.2f)" % (np.std(k1)*100), '\t\t', "%0.2f" % (np.mean(k2)*100), "(%0.2f)" % (np.std(k2)*100), '\t\t', "%0.2f" % (np.mean(k3)*100), "(%0.2f)" % (np.std(k3)*100))
    print('PARAGRAPH:\t', "%0.2f" % (np.mean(m1)*100), "(%0.2f)" % (np.std(m1)*100), '\t\t', "%0.2f" % (np.mean(m2)*100), "(%0.2f)" % (np.std(m2)*100), '\t\t', "%0.2f" % (np.mean(l3)*100), "(%0.2f)" % (np.std(m3)*100))
    print('NOISE:\t\t', "%0.2f" % (np.mean(n1)*100), "(%0.2f)" % (np.std(n1)*100), '\t\t', "%0.2f" % (np.mean(n2)*100), "(%0.2f)" % (np.std(n2)*100), '\t\t', "%0.2f" % (np.mean(l3)*100), "(%0.2f)" % (np.std(n3)*100))
   

#######################################################################################################    
    
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
                                
            #f = open(file_path, 'rt')
            #reader = csv.reader(f)
            #print(reader)
            
            #list_for_storing += list(reader)  
            #number_of_file += 1
            #print(number_of_file)
            #print(file_path)
            with open(file_path) as f: # 'rb'하면 에러.. 왜..?
                
                try:
                    reader = csv.reader(f)
                    list_for_storing += list(reader)
                    number_of_file += 1
                except(UnicodeDecodeError):
                    pass
    
#    print "[ csv_data_load_from_multiple_folders complete ]"
#    print('#of_files:',number_of_file, '#of_leafs:',len(list_for_storing), 'from data_manipulation_library.py')  

    return list_for_storing # output: 2D list


def sample_csv_data_for_aug_publisher(path, list_for_storing, random_list, test_or_train, n_of_aug): # input format is 1D list
    
    print(test_or_train)
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
    
    print(test_or_train)
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

            with open(file_path) as f: # 'rb'하면 에러.. 왜..?
                try:
                    reader = csv.reader(f)
                    list_for_storing += list(reader)
                    number_of_file += 1
                except(UnicodeDecodeError):
                    pass
    
 #   print "[ csv_data_load_from_multiple_folders complete ]"
 #   print('#of_files:',number_of_file, '#of_leafs:',len(list_for_storing), 'from data_manipulation_library.py')           
    
    return list_for_storing # output: 2D list    
    
        
def new_load_dataset(path, random_list, test_or_train): # input format is 1D list
    
    #print(test_or_train)
    list_for_storing = []
    number_of_file = 0
    parent_folder = -2
    
    # 이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
    for dirName, subdirList, fileList in os.walk(path):
        
        # parent folder를 제외시키기 위해
        parent_folder += 1
        if parent_folder==-1: continue     
        if random_list[parent_folder]==test_or_train: continue
        #print('Found directory: %s' % dirName)
 
        for fname in fileList:
    
            # 파일의 절대 경로를 만들어 주기 위한 코드
            file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.csv
  
            temp_str = dirName.split('\\')
            publisher = temp_str[-1] # 현재 publisher 폴더이름
        
            with open(file_path) as f:
                try:
                    reader = csv.reader(f)
                    list_for_storing += list(reader)
                    number_of_file += 1
                except(UnicodeDecodeError):
                    pass                
                
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
    print(len(list2D))
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

    print("[ feature_scaling complete ]")
    return list2D    
    
    
### outer test를 위해..
def bit_and_op(list1, list2):
    result_list = [0] * len(list1)
    for i, _ in enumerate(result_list):
        if list1[i] == 1:
            result_list[i] = 1
        if list2[i] == 1:
            result_list[i] = 1
    return result_list


def ran_test_list(n_publisher):
    
    # 램덜 생성
    ysize = n_publisher
    xsize = n_publisher
    a = np.zeros((ysize, xsize))
    a[np.arange(ysize), np.random.choice(np.arange(xsize), ysize, replace=False)] = 1

    # 5-fold로 임위로 만들어주기
    list1 = bit_and_op(bit_and_op(a[0], a[1]), a[2]) # 3개
    list2 = bit_and_op(a[3], a[4]) # 2개
    list3 = bit_and_op(bit_and_op(a[5], a[6]), a[7]) # 3개
    list4 = bit_and_op(a[8], a[9]) # 2개
    list5 = bit_and_op(bit_and_op(a[10], a[11]), a[12]) # 3개
    
    return [list1, list2, list3, list4, list5]    
