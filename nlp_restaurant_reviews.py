#import all packages required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

#use pandas to read tsv file , and tag is used as delimiter to distinguish both attributes.
data=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

#stemming is done by creating object
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#data cleaning steps
corp=[]
for i in range(1000):
    review=re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('English'))]
    review=' '.join(review)
    corp.append(review)

#tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

#creating a sparse matrix and converting to array
x=cv.fit_transform(corp)
y=data.iloc[:,1]
X=x.toarray()

#seperation of training and testing data
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#training the data
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

#creating a confusion matrix
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

