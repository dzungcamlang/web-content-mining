# Web Content Mining with News Article Dataset
We want to extract only useful information from news article because there are lots of useless (e.g. advertisement) in there. At the same time we want to show our machine learning based model is better than rule based model in terms of dealing with new types of website. The research has been publihsed at [KST International Conference](http://ieeexplore.ieee.org/document/7886134/) (*SVM-based Web Content Mining with Leaf Classification Unit from DOM-tree*). Also, there is the [presentation](https://1drv.ms/p/s!AllPqyV9kKUrgieYsNFYWKqCvjKo) for the conference. 


## Prerequisites
* python 3.5.2
* nltk 3.2.2
* scikit-learn 0.19.0
* bs4 4.5.1 (BeatifulSoup)
* pandas 0.20.3

## Dataset
We have collected about 2500 news articles published by 13 different news websites. You can download it in here. Note that the type of data is just a html file, which is not annotated. 

## Usage
```
python make_trainset.py
```
* Annotate data based on rule list (where is title, date, paragraph in each news website)
* Extract feature vectors out excel files
```
python make_main.py inner
```
* Train and test with the same kinds of news website (inner test) 
```
python make_main.py outer
```
* Train and test with the different kinds of news website (outer test)

## Example



## Contribution
* 

## Summary


## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) and University of Science and Technology (UST), Korea (2016.7 ~ 2016.10)

