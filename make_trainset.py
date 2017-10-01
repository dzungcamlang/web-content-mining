# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import numpy as np
#nltk.download()
import pandas as pd
import codecs
import os
import nltk
import csv
#import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8') # np.savetxt(train_dataset_path, final_vector, fmt='%s', delimiter=",")을 원활히 해주기 위해 그리고 이것 때문에 print 결과가 안나온다.
from data_handling import *
from html_preprocessing import *
from feature_design import *

global leafs_nseq_info, leafs_seq_info, leafs_nested_level, publisher, final_vector
global tagName_dic_1hot_list, tagAttr_dic_1hot_list, tagString_dic_1hot_list
leafs_nseq_info = []
leafs_seq_info = []
leafs_nested_level = []
final_vector = []
tagName_dic_1hot_list = []
tagAttr_dic_1hot_list = []
tagString_dic_1hot_list = []

global n_token_min, n_token_max, n_int_min, n_int_max
global n_sibling_min, n_sibling_max, n_parent_min, n_parent_max, n_element_min, n_element_max
n_token_min = 50
n_token_max = 0
n_int_min = 50
n_int_max = 0
n_sibling_min = 50
n_sibling_max = 0
n_parent_min = 50
n_parent_max = 0
n_element_min = 50
n_element_max = 0

global count
count = 1



# extracting number of the same tags
# soup때문에 main에 상주
def n_of_sametag(tag): 
    return len(list(soup.find_all(tag.name)))

def distance_to_h1(tag): # ex input: (nested tag list, 'body') 
    h1 = len(list(soup.h1.previous_elements))
    current_tag = len(list(tag.previous_elements))
    return abs(h1-current_tag)

def rescale_bounded(min_value, max_value, x):
    return ((2*x - max_value - min_value)/float(max_value - min_value)) if float(max_value - min_value)!=0 else 0
    
def find_min_max(tag):   
    global n_token_min, n_token_max, n_int_min, n_int_max
    global n_sibling_min, n_sibling_max, n_parent_min, n_parent_max, n_element_min, n_element_max
    if tag.string!=None and tag.string!=' ' and len(list(tag.children))==1 and len(list(tag.descendants))==1:
        
        n_token_temp =  n_of_tok(tag)
        if n_token_min > n_token_temp: 
            n_token_min = n_token_temp
        if n_token_max < n_token_temp: 
            n_token_max = n_token_temp
                
        n_int_temp = n_of_int(tag)
        if n_int_min > n_int_temp: 
            n_int_min = n_int_temp
        if n_int_max < n_int_temp: 
            n_int_max = n_int_temp
        
        n_sibling_temp = n_total_sib(tag)
        if n_sibling_min > n_sibling_temp: 
            n_sibling_min = n_sibling_temp
        if n_sibling_max < n_sibling_temp: 
            n_sibling_max = n_sibling_temp        

        n_parent_temp = n_of_parents(tag)
        if n_parent_min > n_parent_temp: 
            n_parent_min = n_parent_temp
        if n_parent_max < n_parent_temp: 
            n_parent_max = n_parent_temp            
            
        n_element_temp = (len(list(tag.previous_elements)))
        if n_element_min > n_element_temp: 
            n_element_min = n_element_temp
        if n_element_max < n_element_temp: 
            n_element_max = n_element_temp 

               
def find_leaf_node(tag):
    global leafs_nseq_info, leafs_seq_info, leafs_nested_level, publisher, final_vector, count
    if tag.string!=None and tag.string!=' ' and len(list(tag.children))==1 and len(list(tag.descendants))==1:
        tagli = extract_nested_level(tag)
        
        if filtering_tagname(tag, tagli)==1 and filtering_tagattrs(tag.attrs.values())==1: # if pass, continue 
            if filtering_tagattrs(tag.parent.attrs.values())==1 and filtering_tagattrs(tag.parent.parent.attrs.values())==1:
                if filtering_tagattrs(tag.parent.parent.parent.attrs.values())==1 and filtering_etc(tag)==1:
                    
                    #if filtering_full_hyperlink(tag)==1:
                        
                    #list = [ # of tokens , # of integers, diff, ratio, # of parents, # of siblings, # of sametag ]
                    nonseq_list = [
                        
                        # Label
                        (labeling_data(tag, publisher)), 
                        
                        
                        # Continuous features
                        rescale_bounded(n_token_min, n_token_max, n_of_tok(tag)), 
                        #rescale_bounded(n_int_min, n_int_max, n_of_int(tag)),
                        rescale_bounded(n_element_min, n_element_max, len(list(tag.previous_elements))),  #(len(list(tag.next_elements))),
                        rescale_bounded(n_parent_min, n_parent_max, n_of_parents(tag)), 
                        rescale_bounded(n_sibling_min, n_sibling_max, n_total_sib(tag)), #(n_next_sib(tag)), (n_pre_sib(tag)),    
                        
                        
                        # Binary features
                        (is_comma_in_string(tag)), 
                        (ratio(n_of_int(tag), n_of_tok(tag))),
                        (ratio(n_of_uppercase(tag), n_of_tok(tag))),
                        (is_date(tag)), #(distance_to_h1(tag)), # for date class
                        (is_hyper_bothends(tag)),
                        
                        
                        # Current_TagName-Dic-based binary features
                         (is_tag(tag, 'h1')), (is_tag(tag, 'p')), #(is_tag(tag, 'span')),      
                         (is_tag(tag, 'time')), (is_tag(tag, 'span')),

#                         (allli_is_tag(tagli, 'article')), (allli_is_tag(tagli, 'header')),
#                         (allli_is_tag(tagli, 'section')), (allli_is_tag(tagli, 'li')), #(allli_is_tag(tagli, 'ol')),
#                         (allli_is_tag(tagli, 'ui')), 
#                         (allli_is_tag(tagli, 'figure')), 
#                         (allli_is_tag(tagli, 'select')), (allli_is_tag(tagli, 'form')),
#                         #(allli_is_tag(tagli, 'h2')),(allli_is_tag(tagli, 'h3')),(allli_is_tag(tagli, 'h4')), 
#                         (allli_is_tag(tagli, 'p'))
                    ]
                    leafs_nseq_info.append(nonseq_list) 

                    
                    # All_Parrents_TagName-Dic-based binary features
                    tagName_dic_1hot_list.append(make_1hot_features(tag, 'tag_name'))
                    
                    # TagAttrs-Dic-based binary features
                    tagAttr_dic_1hot_list.append(make_1hot_features(tag, 'tag_attr'))
                    
                    # TagString-Dic-based binary features
                    tagString_dic_1hot_list.append(make_1hot_features(tag, 'tag_string'))
                
                
                
                    #seq_list = [ labeling_data(tag), extract_tag_name_attrs(tag) ]
                    temp_list = []
                    temp_list.append(labeling_data(tag, publisher))
                    seq_list = temp_list + extract_tag_name_attrs(tag)
                    leafs_seq_info.append(seq_list)

                    leafs_nested_level.append(extract_nested_level(tag).encode('utf-8')) # 'ascii' codec can't decode byte 0xc2 in position 23: ordinal not in range(128) 에러 발생 -> utf-8로 encoding
                    return tag
					

					

#==============================================================================
#                               Main Function
#==============================================================================
# 윈도우는 경로 '\\' 이지만, 리눅스는 '/'이다.
# 윈도우에서 파일 열고있는 상태에서 (해당파일에 접근하는) 프로그램 실행하면 오류난다.
# http://pythoncentral.io/how-to-traverse-a-directory-tree-in-python-guide-to-os-walk/
#dataset_path = './article_dataset'
#full_path = 'C:\\Users\\Office\\Desktop\\Python Workspace\\Tag Classification Project\\'
full_path = os.getcwd()
#dataset_path = '.\\Dataset\\test_ori'
dataset_path = full_path+'\\dataset\\data_unlabeled'
#dataset_path = full_path+'Dataset\\test'
#dataset_path = full_path+'Dataset\\testest'

def str2DArray(list_2D):
    for i, list_1D in enumerate(list_2D):
        for j, _ in enumerate(list_1D):
            if type(list_2D[i][j]) == type(b'str'):
                list_2D[i][j] = str(list_2D[i][j])

    return list_2D


#이중 폴더 속에 있는 파일들을 Load하기 위한 이중 for문
for dirName, subdirList, fileList in os.walk(dataset_path):
    #print('Found directory: %s' % dirName)
    for fname in fileList:

        # 파일의 절대 경로를 만들어 주기 위한 코드
        file_path = dirName+'\\'+fname # ex: ./test_dataset/Telegraph/Google-2010-Telegraph-20160706164841234.html
        temp_str = dirName.split('\\')
        publisher = temp_str[-1] # 현재 publisher 폴더이름
        
        
        ### a single html file load & preprocessing
        html_file = codecs.open(file_path, 'r')
        soup = BeautifulSoup(html_file, 'html.parser')
        soup = html_preprocessing(soup)
        
        
        ### Calculate min/max values for Rescale bounded continuous features
        soup.find_all(find_min_max)
 

        ### Make training data set
        leaf_nodes = soup.find_all(find_leaf_node)
        
        ## 모든 vector들의 dim의 통일해주기 위한 코드
        final_vector = leafs_nseq_info
        
        
        
        for idx, vector in enumerate(final_vector):
            vector += tagName_dic_1hot_list[idx][:]
            vector += tagAttr_dic_1hot_list[idx][:]
            vector += tagString_dic_1hot_list[idx][:]
            vector += leafs_seq_info[idx][1:] # list+list
            vector.append(str(leaf_nodes[idx]).replace(",", "(c)")) # list+string
            vector.append(leafs_nested_level[idx])
    
        
        #final_vector.insert(0, ["class","# token","# int","dklf"]) # column name

        
        ### Storing to CSV file 
        #train_dataset_path = './labeled_dataset'
        #train_dataset_path = './labeled_dataset'
        
        train_dataset_path = full_path+'\\dataset\\data_labeled'
        #train_dataset_path = full_path+'Dataset\\'
        
		
        final_vector = str2DArray(final_vector)
        #for i, list_1D in enumerate(final_vector):
        #    for j, _ in enumerate(list_1D):
        #        print(type(final_vector[i][j]))
                #if type(final_vector[i][j]) == type(b''):
                #    print('hi')
        
        final_vector = np.array(final_vector)
        
        file_name = fname.replace("html", "csv")
        train_dataset_path = train_dataset_path+'\\'+publisher+'\\'+file_name
        
        #np.savetxt(train_dataset_path, final_vector, fmt='%s', delimiter=",")
        df = pd.DataFrame(final_vector)
        df.to_csv(train_dataset_path, header=None,index=False)
        
        #with open(train_dataset_path, "wb") as f:
        #    writer = csv.writer(f)
        #    writer.writerows(final_vector)
        
        
        
        ### Initialize global variables
        leafs_nseq_info = []
        leafs_seq_info = []
        leafs_nested_level = []
        final_vector = []
        leaf_nodes = []
        final_vector = []
        tagName_dic_1hot_list = []
        tagAttr_dic_1hot_list = []
        tagString_dic_1hot_list = []
        
        n_token_min = 50
        n_token_max = 0
        n_int_min = 50
        n_int_max = 0
        n_sibling_min = 50
        n_sibling_max = 0
        n_parent_min = 50
        n_parent_max = 0
        n_element_min = 50
        n_element_max = 0
        
        
        
        #break

print('>>> Complete make_trainset.py !!!')