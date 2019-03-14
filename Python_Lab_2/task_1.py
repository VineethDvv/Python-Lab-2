from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#reading the data set train.csv
data_set =pd.read_csv('train.csv')

#replacing all the alphabet values with the numerical values
data_set['Embarked'] = data_set['Embarked'].map({'Q': 1, 'S': 0 ,'C':2})

#replacing the male and female with 0 and 1
data_set['Sex'] = data_set['Sex'].map({'female': 1, 'male': 0})

#replacing all the zero values with the average values
data_set.select_dtypes(include=[np.number]).interpolate().dropna()

#printinting the survived mean values  of two different sex column values which we changed in to numerical
print(data_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean())

#printinting the survived mean values  of the different Pclass values
print(data_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())

data_set['Age'].fillna(data_set['Age'].mean(),inplace =True)
data_set = data_set.dropna(how='any',axis=0)

#printinting the survived mean values  of the different Embarked values which we changed in to numerical
print(data_set[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

#removing the attribute which are not used
data_set=data_set.drop(columns=['Name','SibSp','Parch','Ticket','Fare','Cabin'])
train,test=train_test_split(data_set,test_size=0.2)
train_label=train['Survived']
train=train.drop(columns=['Survived'])
test_label=test['Survived']
test=test.drop(columns=['Survived'])

#Naive Bayes Classifier
NBclf=GaussianNB()
NBclf.fit(train,train_label)

#Support Vector Classifier
SVCclf=SVC()
SVCclf.fit(train,train_label)

#K Nearest Neighbour classifier
KNNclf=KNeighborsClassifier(n_neighbors=2)
KNNclf.fit(train,train_label)
nbscore= NBclf.score(test,test_label)
svcscore=SVCclf.score(test,test_label)
knnscore=KNNclf.score(test,test_label)

#printing the scores
print("Score for Naive Bayes is: ", nbscore)
print("Score for Support Vector Classifier is: ",svcscore)
print("Score for KNN Classifier is : ",knnscore)
