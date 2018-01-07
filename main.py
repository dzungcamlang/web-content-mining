# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:59:06 2016
@author: Office
"""
import os, csv, sys
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
from data_handling import *

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

arg_str = ' '.join(sys.argv[1:])

global labeled_dataset
labeled_dataset = []
number_of_file = 0

### Parameter Setting
length_features = 95 # normal index
end_point_for_feature_scaling = 87 # array index 
num_of_test_sample = 3

sample_percent = 0.1
sample_token = 1

full_path = os.getcwd()
dataset_path = full_path+'\\dataset\\data_labeled'

c0p_list, c0r_list, c0f_list, c0s_list = [], [], [], []
c1p_list, c1r_list, c1f_list, c1s_list = [], [], [], []
c2p_list, c2r_list, c2f_list, c2s_list = [], [], [], []
c3p_list, c3r_list, c3f_list, c3s_list = [], [], [], []

p_micro_list, r_micro_list, f_micro_list = [], [], []
p_macro_list, r_macro_list, f_macro_list = [], [], []





if arg_str == 'inner':

    ### Load Dataset (feature vectors)
    labeled_dataset = csv_data_load_from_multiple_folders(dataset_path, labeled_dataset, 0)
    labeled_dataset = [row for row in labeled_dataset if len(row)==length_features]
    labeled_dataset = shuffle(labeled_dataset)

    ### Dataset is divided into 5 parts
    cross_val_0 = labeled_dataset[0:int(len(labeled_dataset)*0.2)]
    cross_val_1 = labeled_dataset[int(len(labeled_dataset)*0.2):int(len(labeled_dataset)*0.4)]
    cross_val_2 = labeled_dataset[int(len(labeled_dataset)*0.4):int(len(labeled_dataset)*0.6)]
    cross_val_3 = labeled_dataset[int(len(labeled_dataset)*0.6):int(len(labeled_dataset)*0.8)]
    cross_val_4 = labeled_dataset[int(len(labeled_dataset)*0.8):int(len(labeled_dataset)*1.0)]

    for i in range(0, 5):
        
        # Assign train and test dataset
        if i==0: 
            shuf_test_dataset = cross_val_0
            shuf_train_dataset = cross_val_1+cross_val_2+cross_val_3+cross_val_4
        elif i==1:
            shuf_test_dataset = cross_val_1
            shuf_train_dataset = cross_val_0+cross_val_2+cross_val_3+cross_val_4
        elif i==2:
            shuf_test_dataset = cross_val_2
            shuf_train_dataset = cross_val_0+cross_val_1+cross_val_3+cross_val_4
        elif i==3:
            shuf_test_dataset = cross_val_3
            shuf_train_dataset = cross_val_0+cross_val_1+cross_val_2+cross_val_4
        elif i==4:
            shuf_test_dataset = cross_val_4
            shuf_train_dataset = cross_val_0+cross_val_1+cross_val_2+cross_val_3

        ### Split X and Y
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        temp1, temp2 = [], []

        for idx, row in enumerate(shuf_train_dataset):
            for j in range(1, end_point_for_feature_scaling+1): # csv file에서 feature가 있는 index (1~13)
                temp1.append(shuf_train_dataset[idx][j])
            X_train.append(temp1)
            temp1 = [] # temp 0으로 초기화

        for idx, row in enumerate(shuf_test_dataset):
            for j in range(1, end_point_for_feature_scaling+1):
                temp2.append(shuf_test_dataset[idx][j])
            X_test.append(temp2)
            temp2 = [] 

        for idx, row in enumerate(shuf_train_dataset): Y_train.append(shuf_train_dataset[idx][0])
        for idx, row in enumerate(shuf_test_dataset): Y_test.append(shuf_test_dataset[idx][0])

        ### Convert to numpy array with data-type
        X_train = np.array(X_train, dtype='float32')
        X_test = np.array(X_test, dtype='float32')
        Y_train = np.array(Y_train, dtype='int64')
        Y_test = np.array(Y_test, dtype='int64')

        # for inner test
        test_X = X_test
        test_y = Y_test
        
        ### Train Model
        ##############################################
        clf2 = SVC()
        clf2.fit(X_train, Y_train) 
        prediction_SVM = clf2.predict(test_X)
        
        target_names = ['class 0', 'class 1', 'class 2', 'class 3']
        a = classification_report(test_y, prediction_SVM, target_names=target_names)
        
        #print(a)
        #print("\n")
        
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
        
        c0p_list.append(c0p)
        c0r_list.append(c0r)
        c0f_list.append(c0f)
        c0s_list.append(c0s)

        c1p_list.append(c1p)
        c1r_list.append(c1r)
        c1f_list.append(c1f)
        c1s_list.append(c1s)

        c2p_list.append(c2p)
        c2r_list.append(c2r)
        c2f_list.append(c2f)
        c2s_list.append(c2s)

        c3p_list.append(c3p)
        c3r_list.append(c3r)
        c3f_list.append(c3f)
        c3s_list.append(c3s)
        
        labeled_dataset = []

    # Print out Results
    print('\n')
    print('******************************* INNER TEST *******************************')  
    print('SVM (RBF kernel), 5-fold cross validation based on news website \n')  
    print('\n')
    print(mean_std_print(c0p_list,c0r_list,c0f_list,c0s_list, c1p_list,c1r_list,c1f_list,c1s_list, c2p_list,c2r_list,c2f_list,c2s_list, c3p_list,c3r_list,c3f_list,c3s_list))

elif arg_str == 'outer':
    
    ### Prepare Dataset for outer test
    ran_list = ran_test_list(13) # 13 publishers
    
    train1 = new_load_dataset(dataset_path, ran_list[0], 1)
    test1 = new_load_dataset(dataset_path, ran_list[0], 0)
    
    train2 = new_load_dataset(dataset_path, ran_list[1], 1)
    test2 = new_load_dataset(dataset_path, ran_list[1], 0)

    train3 = new_load_dataset(dataset_path, ran_list[2], 1)
    test3 = new_load_dataset(dataset_path, ran_list[2], 0)

    train4 = new_load_dataset(dataset_path, ran_list[3], 1)
    test4 = new_load_dataset(dataset_path, ran_list[3], 0)

    train5 = new_load_dataset(dataset_path, ran_list[4], 1)
    test5 = new_load_dataset(dataset_path, ran_list[4], 0)    
    
    ### Train Model
    for i in range(0, 5):
        if i==0: 
            shuf_test_dataset = test1
            shuf_train_dataset = train1
        elif i==1:
            shuf_test_dataset = test2
            shuf_train_dataset = train2
        elif i==2:
            shuf_test_dataset = test3
            shuf_train_dataset = train3
        elif i==3:
            shuf_test_dataset = test4
            shuf_train_dataset = train4
        elif i==4:
            shuf_test_dataset = test5
            shuf_train_dataset = train5 
    
        shuf_train_dataset = [row for row in shuf_train_dataset if len(row)==length_features]
        shuf_test_dataset = [row for row in shuf_test_dataset if len(row)==length_features]
        
        ### Split X and Y
        X_train, X_test = [], []
        Y_train, Y_test = [], []
        temp1, temp2 = [], []

        for idx, row in enumerate(shuf_train_dataset):
            for j in range(1, end_point_for_feature_scaling+1): # csv file에서 feature가 있는 index (1~13)
                temp1.append(shuf_train_dataset[idx][j])
            X_train.append(temp1)
            temp1 = [] # temp 0으로 초기화

        for idx, row in enumerate(shuf_test_dataset):
            for j in range(1, end_point_for_feature_scaling+1):
                temp2.append(shuf_test_dataset[idx][j])
            X_test.append(temp2)
            temp2 = [] 

        for idx, row in enumerate(shuf_train_dataset): Y_train.append(shuf_train_dataset[idx][0])
        for idx, row in enumerate(shuf_test_dataset): Y_test.append(shuf_test_dataset[idx][0])

        ### Convert to numpy array with data-type
        X_train = np.array(X_train, dtype='float32')
        X_test = np.array(X_test, dtype='float32')
        Y_train = np.array(Y_train, dtype='int64')
        Y_test = np.array(Y_test, dtype='int64')

        # for inner test
        test_X = X_test
        test_y = Y_test
                
        clf2 = SVC()
        clf2.fit(X_train, Y_train) 
        prediction_SVM = clf2.predict(test_X)
        
        target_names = ['class 0', 'class 1', 'class 2', 'class 3']
        a = classification_report(test_y, prediction_SVM, target_names=target_names)
        #print(a)
                
        p_micro_list.append(metrics.precision_score(test_y, prediction_SVM, average='micro'))
        r_micro_list.append(metrics.recall_score(test_y, prediction_SVM, average='micro'))
        f_micro_list.append(metrics.f1_score(test_y, prediction_SVM, average='micro')) 
         
        p_macro_list.append(metrics.precision_score(test_y, prediction_SVM, average='macro'))
        r_macro_list.append(metrics.recall_score(test_y, prediction_SVM, average='macro'))
        f_macro_list.append(metrics.f1_score(test_y, prediction_SVM, average='macro'))
                
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
        
        c0p_list.append(c0p)
        c0r_list.append(c0r)
        c0f_list.append(c0f)
        c0s_list.append(c0s)

        c1p_list.append(c1p)
        c1r_list.append(c1r)
        c1f_list.append(c1f)
        c1s_list.append(c1s)

        c2p_list.append(c2p)
        c2r_list.append(c2r)
        c2f_list.append(c2f)
        c2s_list.append(c2s)

        c3p_list.append(c3p)
        c3r_list.append(c3r)
        c3f_list.append(c3f)
        c3s_list.append(c3s)    
       
        
    # Print out Results
    print('\n')
    print('******************************* OUTER TEST *******************************')  
    print('SVM (RBF kernel), 5-fold cross validation based on news website \n')  
    print('\n')
    print(mean_std_print(c0p_list,c0r_list,c0f_list,c0s_list, c1p_list,c1r_list,c1f_list,c1s_list, c2p_list,c2r_list,c2f_list,c2s_list, c3p_list,c3r_list,c3f_list,c3s_list))    
    
    # micro, macro
    print('\n')
    print('MICRO avg.:\t', "%0.2f" % (np.mean(np.array(p_micro_list))*100), "(%0.2f)" % (np.std(np.array(p_micro_list))*100), '\t\t', "%0.2f" % (np.mean(np.array(r_micro_list))*100), "(%0.2f)" % (np.std(np.array(r_micro_list))*100), '\t\t', "%0.2f" % (np.mean(np.array(f_micro_list))*100), "(%0.2f)" % (np.std(np.array(f_micro_list))*100))
    print('MACRO avg.:\t', "%0.2f" % (np.mean(np.array(p_macro_list))*100), "(%0.2f)" % (np.std(np.array(p_micro_list))*100), '\t\t', "%0.2f" % (np.mean(np.array(r_macro_list))*100), "(%0.2f)" % (np.std(np.array(r_macro_list))*100), '\t\t', "%0.2f" % (np.mean(np.array(f_macro_list))*100), "(%0.2f)" % (np.std(np.array(f_macro_list))*100))

else:
    print('wrong or empty argument!')
    print('please type: python main.py inner or python main.py outer')

