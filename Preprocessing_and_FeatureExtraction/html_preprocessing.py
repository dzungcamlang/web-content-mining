# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:49:37 2016

@author: Office
"""
from bs4 import BeautifulSoup
from leaf_information import allli_is_tag, convt_1d_arr, n_of_tok, n_of_int, n_total_sib


def find_leaf_node(tag):
    if tag.string!=None and tag.string!=' ' and len(list(tag.children))==1 and len(list(tag.descendants))==1:
        return tag

       
    
def html_preprocessing(soup):

    ##### Remove tag content, keeping tag itself
    ## Pruning away <script>, javascript code, <style>m <link> CSS code. they are redundant and they are many leaf tags
    # 태그 자체는 삭제하지 않고, 태그 내용을 삭제한다.(어차피 나중에 leaf node filtering할 때 거르게 된다.) 
    # 태그 자체를 삭제하게 되면 트리 구조가 바뀌기 때문이다. non-seq 정보가 바뀐다. 
    for tag in soup.find_all('script'): tag.clear()
    for tag in soup.find_all('style'): tag.clear()    
    for tag in soup.find_all('link'): tag.clear()
    
    ##### Remove tag attributess, keeping tag itself
    # 속성값을 제거하는 이유는 나중에 char2vec 성능을 높이기 위해서이다
    for tag in soup.findAll('meta'): tag.attrs = None # meta 태그 같은 경우 속성값에 encoded url 주소가 유독 많다. 특히 encoded url 주소는 매우 무의미한 정보이다.
    for tag in soup.findAll('stript'): tag.attrs = None # stript 태그는 javascript 소스 정보를 나타내므로, 역시 js 경로인 url 태그속성이 많다. (encode되었진 않았지만 그래도 태그속성의 정체성이 없다.)
    for tag in soup.findAll('link'): tag.attrs = None # link 태그는 대표적인 url 태그속성을 가지고 있다.
    
    ##### Removing tag itself, keeping its content
    ## a phrase tag such as <`em`>, <`strong`>, <`code`>, <`samp`>, <`kbd`>, <`var`>, a phrase renders as emphasized text, a phrase list is refered by w3schools.com
    phrase_tag_list = ['em', 'strong', 'code', 'samp', 'kbd', 'var']
    for tag in phrase_tag_list:
        for match in soup.find_all(tag):
            match.replaceWithChildren()
    ## text highlight tag such as <`em`>, <`strong`>, <`b`>, <`mark`> except for <`h1~h6`>, list is refered by w3schools.com
    texthighlight_tag_list = ['b', 'mark']
    for tag in texthighlight_tag_list:
        for match in soup.find_all(tag):
            match.replaceWithChildren()
    
    
    ##### Removing both tag and its content
    ## <head> tag : 그냥 통째로 삭제. 주로 태그들이 body에 있다. html구조 변형도 미미하므로 그냥 삭제한다.
    ## 보통 haed tag에는 css, style 정보들이 많다. 
    for head in soup("head"): soup.head.extract()
    
    ##### Replacing a text with another text
    ## Replace a text with another text, for example, when we want to replce 'boy' with 'girl'
    #import re
    #findtoure = soup.find_all(text = re.compile('boy'))
    #for comment in findtoure:
    #    fixed_text = unicode(comment).replace('boy', 'girl')
    #    comment.replace_with(fixed_text)
    
    
    ##### Replace a tag with a text
    ## In order to deal with tag type (not ordinary text), we translate soup as str(soup)
    # <br> 정보를 남겨둬야 한다. 하지만, 그대로 남겨두게 되면 lean node filter 기준에 부합되므로, 옳지 않은 결과를 초래하게 된다.
    # 따라서 <br> 대신에 (br)로 대체한다.
    soup = BeautifulSoup(str(soup).replace("<br>", "(br)"))
    
    
    ## hyperlink tag such as <`a href`>
    for tag in soup.findAll('a', href=True):
        tag.wrap(soup.new_tag("grit"))
        tag.replaceWithChildren()
    
    soup = BeautifulSoup(str(soup).replace("</grit>", " endhyper"))
    soup = BeautifulSoup(str(soup).replace("<grit>", "starthyper "))
    
    ## Marking to parent tag of #grit# (hyperlink tag)
#    for i, leaf_tag in enumerate(soup.find_all(find_leaf_node)):
#        split_list = leaf_tag.string.split('#') 
#        for i, word in enumerate(split_list):
#            split_list[i] = word.strip(' ')
#        for word in split_list:

#            if word=='': split_list.remove('')
#        if split_list[0]=='grit' and split_list[-1]=='grit':     
#            leaf_tag.parent.name = 'grit'
            
    return soup



def filtering_full_hyperlink(tag):
    if tag.parent.name=='grit': return 0
    else: return 1


    
def filtering_etc(tag):
    # when leaf node has only 1 token and 1 integer, prune it.
    if not n_of_int(tag)==1: 
        if not n_of_tok(tag)==1:
            return 1


    

def filtering_tagname(tag, tagli):
    # baed on current tag name
    if not tag.name=='li':
        if not tag.name=='option':
            if not tag.name=='button':
                if not tag.name=='label':
                    if not tag.name=='td':
                        if not tag.name=='footer':
                            if not tag.name=='figcaption':
                                # based on all parents tag name (even once)
                                if not (allli_is_tag(tagli, 'ol') and allli_is_tag(tagli, 'li')):
                                    if not (allli_is_tag(tagli, 'ul') and allli_is_tag(tagli, 'li')):
                                        if not (allli_is_tag(tagli, 'aside')):
                                            if not (allli_is_tag(tagli, 'footer')):
                                                if not (allli_is_tag(tagli, 'nav')):
                                                    if not (allli_is_tag(tagli, 'tr') and allli_is_tag(tagli, 'td')):
                                                        return 1

                                        
       
    
def filtering_tagattrs(tag_attr_values):

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
   

    #print final_a_list

    # Check the list 
    token = 0
    for word in final_a_list:
        if word=='footerinfo': token = 1
        elif word=='footer': token = 1
        elif word=='nav': token = 1
        elif word=='img': token = 1
        elif word=='image': token = 1
        elif word=='images': token = 1
        elif word=='video': token = 1
        elif word=='gallery': token = 1
        elif word=='photo': token = 1
        elif word=='btn': token = 1
        elif word=='btns': token = 1
        elif word=='button': token = 1
        elif word=='ad': token = 1
        elif word=='action': token = 1
        elif word=='plugin': token = 1
        elif word=='facebook': token = 1
        elif word=='twitter': token = 1
        elif word=='twite': token = 1
        elif word=='twt': token = 1
        elif word=='fbk': token = 1
        elif word=='sidebar': token = 1
        #elif word=='item': token = 1
        elif word=='dialog': token = 1
        elif word=='widget': token = 1
        elif word=='widgets': token = 1
        #elif word=='panel': token = 1
        elif word=='caption': token = 1
        elif word=='blurb': token = 1
        elif word=='promo': token = 1
       
    if token==1: return 0
    else: return 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


