# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:45:01 2016

@author: Office
"""

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












        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

















        