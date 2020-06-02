import numpy as np
import pandas as pd
from sklearn import metrics, tree , ensemble
from sklearn.model_selection import train_test_split



DTC = tree.DecisionTreeClassifier(criterion = 'entropy' ,random_state = 0 , max_depth = 4)
RFC = ensemble.RandomForestClassifier(n_estimators = 100 , max_depth = 10 , random_state = 0)

def splits(x,y):
     X_train, X_test, Y_train, Y_test = train_test_split(x , y, test_size = 0.30 , random_state = 42)
     return X_train, Y_train, X_test, Y_test


def train_model(x, y):
    
    DTC.fit(x,y)
    RFC.fit(x,y)

def predicts(x):
    DTC_ypred = DTC.predict(x)
    RFC_ypred = RFC.predict(x)
    return DTC_ypred , RFC_ypred


def outcome(predict , actual , classifier):
    print(classifier,' : \n ' ,metrics.confusion_matrix(actual, predict))
    print(classifier,' : \n' ,metrics.classification_report(actual, predict))
    print(classifier,' : \n' ,metrics.accuracy_score(actual, predict))

def filter(data):
    gnd = pd.get_dummies(data['gender'], drop_first = True)
    spliz = pd.get_dummies(data['specialisation'], drop_first= True)
    dgt = pd.get_dummies(data['degree_t'], drop_first= True)
    stats = pd.get_dummies(data['status'], drop_first=True)
    workexp = pd.get_dummies(data['workex'], drop_first=True)
    sscboard = pd.get_dummies(data['ssc_b'],drop_first=True)
    hscboard = pd.get_dummies(data['hsc_b'], drop_first=True)
    hscs = pd.get_dummies(data['hsc_s'], drop_first=True)

    data=pd.concat([data,gnd,spliz,dgt,stats,workexp,sscboard,hscboard,hscs], axis =1)

    data.drop(['gender','degree_t','workex','specialisation','status','sl_no','ssc_b','hsc_b','hsc_s'], axis=1, inplace=True)
    data.info()
    x= data.drop(['Placed','salary','M','Others','Science','Commerce','Mkt&HR','Sci&Tech'], axis=1)
    y= data['Placed']
    return x,y


def main():
    sample = pd.read_csv("Placement_Data_Full_Class.csv")
    temp_x, temp_y = filter(sample)
    #print(temp_x, temp_y)
    X,Y,XT,YT = splits(temp_x, temp_y)
    
    train_model(X,Y)
    DTC_op , RFC_op = predicts(XT)
    outcome(DTC_op , YT , 'DecisionTreeClassifier' )
    outcome(RFC_op , YT , 'RandomForestClassifier' )
    

main()
    
