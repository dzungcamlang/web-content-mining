# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:59:06 2016

@author: Office
"""
import os, csv
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
from data_manipulation_library import *

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
    
    print 'class 0:', np.mean(l1)*100, np.std(l1)*100, '-', np.mean(l2)*100, np.std(l2)*100, '-', np.mean(l3)*100,np.std(l3)*100, '-', np.mean(l4)*100,np.std(l4)*100
    print 'class 1:', np.mean(k1)*100, np.std(k1)*100, '-', np.mean(k2)*100, np.std(k2)*100, '-', np.mean(k3)*100,np.std(k3)*100, '-', np.mean(k4)*100,np.std(k4)*100
    print 'class 2:', np.mean(m1)*100, np.std(m1)*100, '-', np.mean(m2)*100, np.std(m2)*100, '-', np.mean(l3)*100,np.std(m3)*100, '-', np.mean(m4)*100,np.std(m4)*100
    print 'class 3:', np.mean(n1)*100, np.std(n1)*100, '-', np.mean(n2)*100, np.std(n2)*100, '-', np.mean(l3)*100,np.std(n3)*100, '-', np.mean(n4)*100,np.std(n4)*100	
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
