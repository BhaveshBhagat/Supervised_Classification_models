import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import math
import numpy as np


''' Reading the Data file using the pandas lib '''
dataset = pd.read_csv('C:/Users/Bhagat/Documents/Python/ML/DataSets/heart.csv')
print(dataset.describe())


''' Scaling down the features variable as we are using the KNN classifier for making prediction '''
std_scaler = StandardScaler()
x_scale = std_scaler.fit_transform(x)

''' finding the most correlated features for best prediction '''
cor = dataset.corr()
print(cor)

''' Feature selection based on correlation values between features and target variables '''

x = dataset[['cp','thalach','exang','oldpeak','slope','ca','thal','age','sex']]
y = dataset['target']

''' we can also check the relationship by visualizing the features using seaborn or pyplot graphs '''
sns.countplot(x='sex' , hue = 'target' , data = dataset)
plt.show()

sns.countplot(x='age' , hue = 'target' , data = dataset)
plt.show()

sns.countplot(x='thalach' , hue = 'target' , data = dataset)
plt.show()



''' dividing the dataset into training and testing part using train_test_split method of model_selection '''
xtrain ,xtest ,ytrain, ytest = train_test_split(x_scale , y ,test_size = 0.30 , random_state = 0)


''' check for different K values which provide least RMSE value in the KNN model 
for k in range(30):
    k =k+1
    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(xtrain,ytrain)

    ypred = knn.predict(xtest)

    print('RMSE Error :', k , metrics.mean_squared_error(ytest,ypred))


'''

''' By using the above method we decided to use K value as 6 '''
''' Creating and train the model and predicting the values '''
knn = KNeighborsClassifier(n_neighbors = 6)
knn.fit(xtrain,ytrain)

ypred = knn.predict(xtest)



''' Check for Model Accuracy Score '''
df = pd.DataFrame({ 'Actual': ytest , 'Predicted': ypred
    })
print(df)

print('Accuracy :' , metrics.accuracy_score(ytest ,ypred))

''' Using Cross validation to check the exact performance of our model '''
from sklearn.model_selection import cross_val_score

score = cross_val_score(knn , x_scale , y , cv=5)
print(score)
print(np.mean(score))


