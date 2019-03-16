import pandas as pd
import copy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#reading dataset
advertising=pd.read_csv('Advt_data.csv')

#dividing data into training data and test data
train,test=train_test_split(advertising,test_size=0.4)

train_label=train['sales']
test_label=test['sales']

#remove the null values
train_eda=train.dropna(how='any',axis=0)
train_eda_label=train_eda['sales']
train=train.drop(columns=['sales'])
train_eda=train_eda.drop(columns=['sales'])

test=test.drop(columns=['sales'])

#Multiple Linear regression without EDA
classification_1 = LinearRegression()
classification_1.fit(train,train_label)

#Linear regression with EDA
classification_2=LinearRegression()
classification_2.fit(train_eda,train_eda_label)

#results on test data
result_1=classification_1.predict(test)
result_2=classification_2.predict(test)

#mean squared error and r2 score with EDA
mean_squared_error_eda=mean_squared_error(test_label,result_2)
r2_score_eda=r2_score(test_label,result_2)

#mean squared error and r2 score without EDA
mean_squared_error = mean_squared_error(test_label, result_1)
r2_score = r2_score(test_label,result_1)


print("mean squared error without EDA is :",mean_squared_error)
print("R2 score with EDA is :",r2_score)
print("mean Squared error with EDA is : ",mean_squared_error_eda)
print("R2 score with EDA is : ",r2_score_eda)
