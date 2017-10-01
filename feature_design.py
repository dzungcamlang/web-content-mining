# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:50:57 2016

@author: Office
"""
import nltk
import numpy as np




####################################################################### Leaf_information.py #################
##### For non-sequential data
# extracting number of all tokens including strings and integers
# 일반적인 pyhton 문법인 split로 token화 하면, 단순히 공백으로 구분하기 때문에 정밀하지 못하다. tokenization tool을 사용해서 . ? , 등 정밀하게 나눌 수 있도록 하자 


def clean_data_for_tag_string(tag_string):
    nltk_token = nltk.word_tokenize(tag_string)
    new_list = ' '.join(nltk_token)
    new_list = new_list.replace('-', ' ')
    new_list = new_list.replace('_', ' ')
    new_list = new_list.replace('.', ' ')
    new_list = new_list.replace(',', ' ')
    new_list = new_list.replace('/', ' ')
    new_list = new_list.replace(':', ' ')
    new_nltk_token_list = new_list.split(' ')
    new_nltk_token_list = [w.lower() for w in new_nltk_token_list] # make uppercase to be lowercase
    return new_nltk_token_list


def clean_tag_string_with_no_lowercase(tag_string):
    nltk_token = nltk.word_tokenize(tag_string)
    new_list = ' '.join(nltk_token)
    new_list = new_list.replace('-', ' ')
    new_list = new_list.replace('_', ' ')
    new_list = new_list.replace('.', ' ')
    new_list = new_list.replace(',', ' ')
    new_list = new_list.replace('/', ' ')
    new_list = new_list.replace(':', ' ')
    new_nltk_token_list = new_list.split(' ')    
    return new_nltk_token_list


def n_of_tok(tag): # 단어/숫자 개수 
    return len(clean_data_for_tag_string(tag.string))

# extracting number of all integers
def n_of_int(tag):
    count = 0
    tag_string = clean_data_for_tag_string(tag.string)
    for i in tag_string:
        if(i.isdigit()): # 숫자인지 확인
            count = count+1
    return count


def n_of_uppercase(tag):
    new_nltk_token_list = clean_tag_string_with_no_lowercase(tag.string)
    count=0
    for word in new_nltk_token_list:
        for char in word:
            if char.isupper()==1:
                count += 1
    return count
    

def is_hyper_bothends(tag):
    tag_string_list = clean_data_for_tag_string(tag.string)
    if tag_string_list[0]=='starthyper' and tag_string_list[-1]=='endhyper':
        return 1
    else:
        return 0

    
def is_comma_in_string(tag):
    token=0
    for char in list(tag.string):
        if char=='.': token=1
    if token==1: return 1
    else: return 0

# calculating ratio
# 수정 필요... 그냥 0/1로 나타내는게 아니라 소숫점자리까지 계산할 필요가 있다.
def ratio(numer, denom):
    #smoothing_term = 0.000000001
    #return (numer+smoothing_term)/(denom+smoothing_term)
    if(numer==0 or denom==0):
        return 0
    else: 
        return float(numer/float(denom))
    

def uuper_ratio(numer, denom):
    #smoothing_term = 0.000000001
    #return (numer+smoothing_term)/(denom+smoothing_term)
    if(numer==0 or denom==0):
        return 0
    else: 
        result = (numer/float(denom))
        if result>=1.0: result = 1
        return result
    
    
# extracting number of tag's parents
def n_of_parents(tag):
    return (len(list(tag.parents)))


# extracting number of tag's siblings
def n_total_sib(tag): return (len(list(tag.previous_siblings)))+(len(list(tag.next_siblings)))
def n_pre_sib(tag): return (len(list(tag.previous_siblings)))
def n_next_sib(tag): return ((len(list(tag.next_siblings))))


def is_tag(tag, what):
    if tag.name==what: return 1
    else: return 0

    
def allli_is_tag(taglist, what): # ex input: (nested tag list, 'body') 
    token=False 
	
    for tag in taglist.split('.'):
        if tag==what: token=True 
    if token==True: return 1    
    else: return 0    
    
    
    
from dateutil.parser import parse
def is_date(tag):
    string = tag.string.split(' ')
    for word in string:
        if word=='Published': string.remove('Published')
        if word=='published': string.remove('published')
        if word=='Updated': string.remove('Updated')
        if word=='updated': string.remove('updated')
      
    tag_string = ' '.join(string)

    try: 
        parse(tag_string)
        return 1
    except ValueError:
        return 0    

    
##### For sequential data
# [['abc'], 'def'] or ['abc', ['def']] -> ['abc', 'def']
def convt_1d_arr(old_list): 
    
    # 문제점: class 속성값들이 항상 string이 아닌 list값을 가지고, 심지어 띄어쓰기가 있을 경우 각각 나눠서 element로 잡는다. 
    # class를 제외한 다른 태그속성들은 그렇지 않다.
    
    old_list = list(old_list)   
	
    ## Step1: [['ab','c'], 'def'] -> [['ab c'], ['def']] (띄어쓰기를 살려둔채 하나로 묶는다.)
    for idx, content in enumerate(old_list):
        if type(content)==type([]):
            if len(content)>=2: 
                temp = " ".join(str(x) for x in content) # " " 쉼표를 연결점으로 합체
                old_list[idx] = [] # 초기화 하고 (비워놓고)
                old_list[idx].append(temp) # [u'story', u'story-body__h2'] -> ['story body__h1'] 할당
    
    ## Step2: [['abc'], 'def'] or ['abc', ['def']] -> ['abc', 'def']
    new_list = []
    for i in old_list:
        if type(i)==type([]): 
            new_list.insert(0, i[0]) # Step1을 거쳤기 때문에 하나의 element만 가지는 array가 존재하므로 그냥 i[0] 할당, 그리고 class 속성이므로 맨 앞자리로 할당
        else: new_list.append(i)
            
    return new_list

# Among many attribute values, choosing one of them
# 현재 기준은 class 속성 우선순위(class속성은 list를 가지는 성질 이용), 콤파와 " 가 있는 속성제외
# 나중에 좀 더 정밀하게 나눌 필요가 있다.(되도록 숫자 또는 이상한 문자들이 있으면 안된다. 그리고 길이가 긴 속성도 있으면 안좋다.)
def select_attr_value(list): 
    # 기준: 다음과 같은 문자들이 나오면 제외한다.
    # 이는 arbitrary한 속성값을 사용한다는 가정에는 위배되지만, 일반적인 속성값이 아닌 특수한 속성값이므로 제외한다.
    # 여기서 말하는 일반적인 속성값은 영어char들로 이뤄진 속성값과 자주 사용되는 ' ', '-', '_'가 포함된 속성값을 말한다.
    for idx in list: # 어차피 for문으로 돌리기 때문에 가장 왼쪽일수록 우선순위를 가진다.
        if not ',' in idx: 
            if not '"' in idx: 
                if not '/' in idx:
                    if not '.' in idx:
                        if not ':' in idx:
                            if len(str(idx))<=30: # 일딴, 30 길이 filter 나중에 좀 더 정교한 방법이 필요 무작정 길이가 길다고 짜르는 것 보다는...
                                return idx # 거기에 , 와 " 를 filtering 추가하였다.
                            else: return ''
                        else: return ''
                    else: return ''
                else: return ''
            else: return ''
        else: return ''
                            
# for extracting tag name and tag attributes (not class) 
def extract_tag_name_attrs(tag):
    tag_name_attrs_list = [] # 모든 Sequential 정보가 들어가는 하나의 vector
 


    ### 5단계 부모 노드 정보
    #tag_name_attrs_list.append(tag.parent.parent.parent.name) # parnet tag name
    #tag_name_attrs_list.append(str(tag.parent.parent.parent.parent.parent.attrs).replace(",", "(c)"))


    ### 3단계 부모 노드 정보
    #tag_name_attrs_list.append(tag.parent.parent.parent.name) # parnet tag name
    tag_name_attrs_list.append(str(tag.parent.parent.parent.attrs).replace(",", "(c)"))


    ### 2단계 부모 노드 정보
    tag_name_attrs_list.append(tag.parent.parent.name) # parnet tag name
    tag_name_attrs_list.append(str(tag.parent.parent.attrs).replace(",", "(c)"))

    
    ### 1단계 부모 노드 정보
    tag_name_attrs_list.append(tag.parent.name) # parnet tag name
    tag_name_attrs_list.append(str(tag.parent.attrs).replace(",", "(c)"))
    


#    ### Attract parent (a single) tag attribute
#    parent_attrs_list = convt_1d_arr(tag.parent.attrs.values()) # my tag attributs
#        
#    if(len(parent_attrs_list)==0): # attrs_list에 아무 내용이 없을 때,
#        tag_name_attrs_list.append('') # 그냥 아무런 값이 없는 string을 넣자.
#    #elif(len(parent_attrs_list)==1): # attrs_list에 1개가 있으면
#    #    tag_name_attrs_list.append(parent_attrs_list[0]) # 그 속성값을 그대로 할당
#    else: # attrs_list에 여러개가 있으면
#        tag_name_attrs_list.append(select_attr_value(parent_attrs_list)) # select_attr_value 함수 기준으로 속성값 할당
#     
#     
#    ### Attract current (a single) tag name 
#    tag_name = '<'+tag.name+'>' # 태그이름과 태그속성을 구분해주기 위해 < > 를 사용.
#    if len(str(tag_name))<=10: # 길이 10으로 제한
#        tag_name_attrs_list.append(tag_name) # my tag name
#     
#     
#    ### Attract current (a single) tag attribute
#    # class 태그속성이름으로부터 얻어진 태그속성내용은 string이 아닌 list이다. -> convert_1D_array 함수로 전처리 
#    attrs_list = convt_1d_arr(tag.attrs.values()) # my tag attributs
#     
#    if(len(attrs_list)==0): # attrs_list에 아무 내용이 없을 때,
#        tag_name_attrs_list.append('') # 그냥 아무런 값이 없는 string을 넣자.
#    #elif(len(parent_attrs_list)==1): # attrs_list에 1개가 있으면
#    #    tag_name_attrs_list.append(attrs_list[0]) # 그 속성값을 그대로 할당   
#    else: # attrs_list에 여러개가 있으면
#        tag_name_attrs_list.append(select_attr_value(attrs_list)) # select_attr_value 함수 기준으로 속성값 할당

    



# 그냥 dictlist에 있는 모든 attributes들을 list에 append    
#     for index in attrs_list:
#         #if(len(index[0])==1):
#         #    tag_name_attrs_list.append(index)
#         a = True
#         for j in index:
#             if(len(j)!=1):
#                 tag_name_attrs_list.append(j)
#             if(len(j)==1 and a==True):
#                 a = False
#                 tag_name_attrs_list.append(index)
    
    return tag_name_attrs_list




##### For extracting nested level of a leaf node
def extract_nested_level(tag):
    return '.'.join(reversed([p.name for p in tag.parentGenerator() if p])) # utf-8 encoding이 필요하다. main함수에서 불러올 떄, utf-8 encoding을 하자.
########## End of Leaf_information.py ##########



def clean_data_for_tag_attr(tag_attr_values):
    # Prepare making values in a list
    a_list = convt_1d_arr(tag_attr_values)
   
    temp = []
    for i, word in enumerate(a_list):   
        for j, char in enumerate(word):
            if char.isupper():
                temp.append('-'+char.lower())
            else:
                temp.append(char)
        temp.append('-')    

    new_list =  ''.join(temp)
    new_list = new_list.replace('-', ' ')
    new_list = new_list.replace('_', ' ')
    
    final_a_list = new_list.split(' ') 
    
    return final_a_list



def make_1hot_features(tag, whatkind):
    
    if whatkind=='tag_name':
        # Extract tag names from data        
        tag_name_from_data = []
        tagli = extract_nested_level(tag)
        tag_name_from_data.append(tag.name)
        tag_name_from_data += tagli.split('.')
        
        # Load tag name dictionary
        tag_name_dic = [row.rstrip('\n') for row in open('assets/tag_name_dic.txt')]
        tag_name_dic_1hot_list = [0]*len(tag_name_dic)
        
        for i, word in enumerate(tag_name_dic):
            for data in tag_name_from_data:
                for split_word in word.split(' '):
                    if split_word==data: 
                        tag_name_dic_1hot_list[i] = 1
                
        return tag_name_dic_1hot_list
    
    
    if whatkind=='tag_attr':
        # Extract tag attributes from data
        tag_attr_from_data = []
        tag_attr_list = clean_data_for_tag_attr(tag.attrs.values())        
        #print tag_attr_list
        tag_1parent_attr_list = clean_data_for_tag_attr(tag.parent.attrs.values())
        tag_2parent_attr_list = clean_data_for_tag_attr(tag.parent.parent.attrs.values())
        tag_3parent_attr_list = clean_data_for_tag_attr(tag.parent.parent.parent.attrs.values())
        
        # merge
        tag_attr_from_data += tag_attr_list
        tag_attr_from_data += tag_1parent_attr_list
        tag_attr_from_data += tag_2parent_attr_list
        tag_attr_from_data += tag_3parent_attr_list

        
        # Load tag name dictionary
        tag_attr_dic = [row.rstrip('\n') for row in open('assets/tag_attr_dic.txt')]
        tag_attr_dic_1hot_list = [0]*len(tag_attr_dic)
        
        for i, word in enumerate(tag_attr_dic):
            for data in tag_attr_from_data:
                for split_word in word.split(' '):
                    if split_word==data: 
                        tag_attr_dic_1hot_list[i] = 1
                
        return tag_attr_dic_1hot_list

    
    if whatkind=='tag_string':
        # Extract tag attributes from data
        tag_string_from_data = []
        tag_string_list = clean_data_for_tag_string(tag.string)
        
        #tag_1parent_string_list = clean_data_for_tag_string(tag.parent.string)
        #tag_2parent_string_list = clean_data_for_tag_string(tag.parent.parent.string)
        #tag_3parent_string_list = clean_data_for_tag_string(tag.parent.parent.parent.string)
        
        
        # merge
        tag_string_from_data += tag_string_list
        #tag_string_from_data += tag_1parent_string_list
        #tag_string_from_data += tag_2parent_string_list
        #tag_string_from_data += tag_3parent_string_list

        
        
        # Load tag name dictionary
        noise_word_dic = [row.rstrip('\n') for row in open('assets/noise_word_dic.txt')]
        noise_word_dic_1hot_list = [0]*len(noise_word_dic)
        
        for i, word in enumerate(noise_word_dic):
            for data in tag_string_from_data:
                for split_word in word.split(' '):
                    if split_word==data: 
                        noise_word_dic_1hot_list[i] = 1
                
        return noise_word_dic_1hot_list




###############################################################################################
# 각 언론사 웹사이트별 제목, 날짜, 저자, 본문 규칙 리스트
# publisher 변수가 main에 상주해야 되기 때문에 어쩔 수 없이...
def labeling_data(tag, publisher):
    # class 태그속성이름 조심 - 태그속성이 배열로 저장되어 있고, get함수로 태그속성을 extract하면 띄어쓰기가 있는 경우 배열의 element로 분리된다.
    # and 조건이 복잡할수록, and로 표현하지말고 그냥 직렬로 표현하자. 많은 and로 if문을 만들면 안되는 경우가 많았다.
    TITLE_LABEL = 1
    DATE_LABEL = 2
    BODY_LABEL = 3
    NOISE_LABEL = 4   
    
    if(publisher=='ABCAU'):
        if(tag.name=='h1'):
            if tag.parent.get('class')==['article', 'section']: return TITLE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='span'):
            if(tag.get('class')==['print'] or tag.get('class')==['timestamp'] or tag.get('class')==['noprint']): return DATE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='p'):
            if tag.parent.name=='div' and tag.parent.get('class')==['article', 'section']: return BODY_LABEL
            else: return NOISE_LABEL # BODY_NOISE라고 볼수 있다.   
            # p가 본문의 paragraph일수도 있고, 광고, 추천섹션의 내용일수도 있다.
        #elif(tag.name=='h2'):
        #    if tag.parent.name=='div' and tag.parent.get('class')==['article', 'section']: return BODY_LABEL
        #    else: return NOISE_LABEL # BODY_NOISE라고 볼수 있다.   
            # h2가 본문의 소제목일수도 있고, 광고,추천섹션의 제목일수도 있다.                 
        else: return NOISE_LABEL
    
    elif(publisher=='BBC'):
        if(tag.name=='h1'):
            if tag.get('class')==['story-body__h1']: return TITLE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='div'): # 나중에 클래스를 여러개로 분리할 때, 이 부분은 수정이 필요. 하나의 태그가 기준이 되야 한다.
            if(tag.get('class')==['date', 'date--v2'] or tag.get('class')==['date', 'date--v1']):
                if(tag.parent.get('id')=='media-asset-page-text') or (tag.parent.get('class')==['mini-info-list__item']): return 2
                else: return 4
            else: return 4
        elif(tag.namep=='p'):
            if(tag.parent.get('div')=='story-body__inner') or (tag.parent.get('class')==['body-content']): return 3
            else: return 4
        elif(tag.parent.name=='div'):
            if tag.parent.get('class')==['story-body__inner']: return 3
            else: return NOISE_LABEL 
        else: return NOISE_LABEL    
    
    # CNN HTML 파일 자체가 잘못된게 많다...
    elif(publisher=='CNN'):
        if(tag.name=='h1'):
            if tag.get('class')==['pg-headline']: return TITLE_LABEL
            elif tag.parent.get('class')==['cnn_storyarea']: return 1
            else: return 4
        elif(tag.name=='p'):
            if tag.get('class')==['update-time']: return DATE_LABEL
            elif tag.parent.get('class')==['cnn_strycntntlft']: return 3
            else: return 4    
        elif(tag.name=='div' and tag.parent.get('class')==['l-container']):
            if tag.get('class')==['zn-body__paragraph']: return 3
            else: return 4
        elif(tag.name=='div' and tag.parent.get('class')==['zn-body__read-all']):
            if tag.get('class')==['zn-body__paragraph']: return 3
            else: return 4
        else: return 4
    
    elif(publisher=='FoxNews'):
        if(tag.name=='h1'):
            if tag.get('itemprop')=='headline': return TITLE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='time'): return DATE_LABEL       
        
        elif(tag.name=='p'):
            if tag.parent.get('class')==['article-text']: return BODY_LABEL
            else: return NOISE_LABEL
        else: return NOISE_LABEL
 
    elif(publisher=='Independent'):
        if tag.name=='p':
            if tag.parent.get('class')==['intro']: return TITLE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='time'):
            if tag.parent.name=='li': return DATE_LABEL
            else: return NOISE_LABEL
        elif(tag.name=='p'):
            if tag.parent.get('itemprop')=='articleBody': return BODY_LABEL
            else: return NOISE_LABEL
        else: return NOISE_LABEL
        
    elif(publisher=='Reuters'):
        if(tag.name=='h1'):
            if tag.get('class')==['article-headline']: return 1
            else: return 4
        elif(tag.name=='span'):
            if tag.get('class')==['timestamp']: return 2  
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('id')=='article-text' or tag.parent.parent.get('id')=='article-text': return 3
            elif tag.parent.get('id')=='articleText': return 3
            else: return 4
        else: return 4        

    elif(publisher=='SkyNews'):
        if(tag.name=='span' or tag.name=='p'):
            if tag.parent.get('class')==['story__header'] or tag.parent.name=='h1': return 1
            else: return 4
        elif(tag.name=='span'):
            if tag.parent.get('class')==['last-updated__text'] or tag.parent.name=='time': return 2
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['content-column'] or tag.parent.get('class')==['sky-component-story-article'] or tag.parent.get('class')==['sky-component-story-article__body']: return 3
            else: return 4
        else: return 4        

    elif(publisher=='LATimes'):
        if(tag.name=='h1'):
            if tag.get('class')==['trb_ar_hl_t']: return 1
            else: return 4
        elif(tag.name=='time'):
            if tag.get('itemprop')=='datePublished': return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['trb_ar_page']: return 3
            else: return 4          
        else: return 4    
    
    elif(publisher=='Telegraph'):
        if(tag.name=='h1' and tag.get('itemprop')=='headline name'): return 1
        #if(tag.name=='h2' and tag.get('itemprop')=='alternativeHeadline description'): return 1 # 소제목
        elif(tag.name=='time' and tag.get('class')==['article-date-published']): return 2
        elif(tag.name=='p' and tag.get('class')==['publishedDate']): return 2       
        elif(tag.name=='p' and tag.parent.parent.get('itemprop')=='articleBody'): return 3
        else: return 4
        
    elif(publisher=='Aljazeera'):
        if(tag.name=='h1'):
            if tag.get('class')==['heading-story']: return 1
            else: return 4
        elif(tag.name=='time'): return 2 
        elif(tag.name=='p'):
                if tag.parent.get('class')==['article-body'] or tag.parent.get('class')==['article-body-full'] or tag.parent.get('class')==['caption']: return 3
                else: return 4          
        elif(tag.parent.name=='p'):
            if tag.get('lang')=='EN-GB': return 3
            else: return 4        
        else: return 4
        
    elif(publisher=='USAToday'):
        if(tag.name=='h1'):
            if tag.get('class')==['asset-headline']: return 1
            else: return 4
        elif(tag.name=='span'):
            if tag.get('class')==['asset-metabar-time','asset-metabar-item','nobyline']: return 2  
            elif tag.parent.name=='p': return 3
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('role')=='main' or tag.parent.get('itemprop')==['articleBody']: return 3
            return 4
        else: return 4   
    
    elif(publisher=='TheTimesUK'):
        if(tag.name=='h1'):
            if tag.get('id')=='title' or tag.get('class')==['Article-headline', 'Headline', 'Headline--article']: return 1
            else: return 4    
        elif(tag.name=='div' or tag.name=='time'):
            if tag.get('class')==['f-regular-update'] or tag.get('class')==['Dateline']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['Article-content'] or tag.parent.get('class')==['contentpage', 'currentpage']: return 3
            else: return 4       
        else: return 4

    elif(publisher=='Boston'):
        if(tag.name=='h1'):
            if tag.parent.get('id')=='blogEntry' or tag.get('class')==['content-header__headline'] or tag.get('id')=='headTools': return 1
            else: return 4    
        elif(tag.name=='div'):
            if tag.get('class')==['content-byline__timestamp']: return 2 
            else: return 4
        elif(tag.name=='span'):
            if tag.get('id')=='dateline': return 2
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['content-text', 'content-text-article'] or tag.parent.get('class')==['articlePluckHidden']: return 3
            elif tag.parent.name=='div': return 3
            else: return 4       
        else: return 4

    elif(publisher=='MSNBC'):
        if(tag.name=='h1'):
            if tag.get('class')==['is-title-pane','panel-pane', 'pane-node-title']: return 1
            else: return 4    
        elif(tag.name=='time'):
            if tag.get('class')==['article-date-posted']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('itemprop')=='articleBody': return 3
            else: return 4       
        else: return 4

    elif(publisher=='WSJ'):
        if(tag.name=='h1'):
            if tag.get('itemprop')=='headline': return 1
            else: return 4    
        elif(tag.name=='time'):
            if tag.get('class')==['timestamp']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['wsj-snippet-body'] or tag.parent.get('itemprop')=='articleBody': return 3
            else: return 4       
        else: return 4
        
    elif(publisher=='EuroNews'):
        if(tag.name=='h1'):
            if tag.parent.get('id')=='title-wrap-new' or tag.get('style')=='padding:10px;': return 1
            else: return 4    
        elif(tag.parent.get('id')=='title-wrap-new'):
            if tag.get('class')=='wireCet' or tag.get('class')==['cet', 'leftFloat', 'clearFloat']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('id')=='articleTranscript' or tag.parent.get('id')=='article-text': return 3
            else: return 4       
        else: return 4
        
    elif(publisher=='CSmonitor'):
        if(tag.name=='h1'):
            if tag.get('id')=='headline': return 1
            else: return 4    
        elif(tag.name=='time'):
            if tag.get('id')=='date-published': return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('id')=='summary' or tag.parent.get('id')=='story-body': return 3
            elif tag.parent.name=='blockquote': return 3
            else: return 4       
        else: return 4

    elif(publisher=='TheGlobeandmail'):
        if(tag.name=='h1'):
            if tag.get('class')==['entry-title']: return 1
            else: return 4    
        elif(tag.name=='time'): return 2
            #if tag.get('id')=='date-published': return 2 
            #else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['column-2', 'gridcol']: return 3
            else: return 4       
        else: return 4
        
    elif(publisher=='NYTimes'):
        if(tag.name=='h1'):
            if tag.get('itemprop')=='headline': return 1
            else: return 4    
        elif(tag.name=='time'):  
            if tag.get('class')==['dateline']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['story-body', 'story-body-1']: return 3
            else: return 4       
        else: return 4

    elif(publisher=='ABCNews'):
        if(tag.name=='h1'):
            if tag.parent.get('class')==['article-header']: return 1
            else: return 4    
        elif(tag.name=='span'):  
            if tag.get('class')==['timestamp']: return 2 
            else: return 4
        elif(tag.name=='p'):
            if tag.parent.get('class')==['article-copy']: return 3
            else: return 4       
        else: return 4






