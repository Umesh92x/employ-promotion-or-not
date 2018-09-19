import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder


dataset=pd.read_csv('train.csv')

all_column=dataset.columns

data_des=dataset.describe()

X= dataset.drop(['is_promoted'],axis=1)
X=X.drop(['employee_id','region','recruitment_channel','gender'],axis=1)

XX=dataset.iloc[:,-1].values

y=dataset['is_promoted']


data_des=X.describe()

# **********************************************************
'''                                 FILLING NAN VALUES                         '''
nan_values=X.isna().sum()
#eduction_caterogies=X.groupby('education').count()
#departement_categories=X.groupby('department').count()

edu_count=X['education'].value_counts()
max_id=edu_count.idxmax()

#X['department'].value_counts()
y.value_counts()
#pd.core.categorical.Categorical.fillna()

X['education']=X['education'].fillna(max_id)

mean_of_rating=int(X['previous_year_rating'].mean())

X['previous_year_rating']=X['previous_year_rating'].fillna(mean_of_rating)
X1=X

data_des=X.describe()

X1=pd.get_dummies(X1)
X1=X1.drop(['gender_m'],axis=1)
#X1=X1.drop(['gender_f'],axis=1)
X1=X1.drop(['department_Operations'],axis=1)
X1=X1.drop(['education_Below Secondary'],axis=1)
#X1=X1.drop(['awards_won?'],axis=1)

#X1=X1.drop(['region_region_1'],axis=1)


X1=X1.drop(['age'],axis=1)
X1=X1.drop(['length_of_service'],axis=1)
X1=X1.drop(['department_Finance'],axis=1)
X1=X1.drop(['department_Legal'],axis=1)

#X1=X1.drop(['recruitment_channel_other'],axis=1)
X1=X1.drop(['gender_f'],axis=1)
X1=X1.drop(["education_Master's & above"],axis=1)


'''   ********************************************************************  '''
#onehot=OneHotEncoder(categorical_features='all')
#X=onehot.fit_transform(X['department']).toarray()
2,4,9,17,18,19

X_train, X_test, y_train, y_test = train_test_split(X1, y,
                                                    stratify=y, 
                                                    test_size=0.20,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=.15,random_state=42)


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score
xgb_model=XGBClassifier()
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)
f1_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
xgb_model.feature_importances_

y.to_csv('xgb.csv',sep=',')

dataset['employee_id'].to_csv('emply_id.csv',sep=',')

from sklearn.naive_bayes import GaussianNB
naive_classifier = GaussianNB()
naive_classifier.fit(X_train,y_train)
y_pred_NB=naive_classifier.predict(X_test)
f1_score(y_test,y_pred)

from sklearn.ensemble import GradientBoostingClassifier
GBM_model=GradientBoostingClassifier(random_state=10)
GBM_model.fit(X_train,y_train)

y_pred_GBM=GBM_model.predict(X_test)
accuracy_score(y_test,y_pred_GBM)


from sklearn.ensemble import RandomForestClassifier
forest_classifier=RandomForestClassifier(n_estimators=500,criterion='entropy',verbose=1,random_state=0)
forest_classifier.fit(X_train,y_train)
y_pred_forest=forest_classifier.predict(X_test)
f1_score(y_test,y_pred_forest)


