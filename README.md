# Web Content Mining with News Article Dataset
We want to extract only useful information from news article because there are lots of useless (e.g. advertisement) in there. At the same time we want to show our machine learning based model is better than rule based model in terms of dealing with new types of website. Note that some patterns are very different to each other even though they are news websites. The research has been publihsed at [KST International Conference](http://ieeexplore.ieee.org/document/7886134/) (*SVM-based Web Content Mining with Leaf Classification Unit from DOM-tree*). Also, there is the [presentation](https://1drv.ms/p/s!AllPqyV9kKUrgieYsNFYWKqCvjKo) for the conference. 


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

#### Example
![](/assets/example.PNG)


## Contribution
* Web content mining using machine learning model
* Robustly deal with new data drawn by new website, which rule based can't predict (shown from outer test)
* Given small amount of dataset with high variance, SVM with kernel was best among logistic regression and multilayer neural network, etc. 

## Summary
* Build the machine learning pipeline for this task.
* Based on HTML file structure (using BeautifulSoup), design feature vectors in which elements are either contiuous or discrete.
* Not optimize hyperparameters of model in order to predict well new data from new websites.
* Test with n-fold cross validation for showing robustness for new websites.

_about resources_
* Some materials about html files are: [[.xlsx]](https://1drv.ms/x/s!AllPqyV9kKUrg3qOK2DE7P-TSWW3)(tag name and attribute list), [[.doc]](https://1drv.ms/w/s!AllPqyV9kKUrg3mmTjb6YsLD_wzi)(page source for only paragraph), [[.doc]](https://1drv.ms/w/s!AllPqyV9kKUrg3sV7zNGUFPfNsvQ)(page source for date and paragraph)
* Certificate of the conference [[link]](https://github.com/gritmind/web-content-mining/blob/master/assets/certificate_of_contributions.pdf)

## Acknowledgement
Korea Institute of Science and Technology Information (KISTI) <br>
University of Science and Technology (UST) <br>
2016.7 ~ 2016.10

