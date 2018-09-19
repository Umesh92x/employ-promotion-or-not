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
X=X.drop(['employee_id','region'],axis=1)

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

X['department'].value_counts()
y.value_counts()
#pd.core.categorical.Categorical.fillna()

X['education']=X['education'].fillna(max_id)

mean_of_rating=int(X['previous_year_rating'].mean())

X['previous_year_rating']=X['previous_year_rating'].fillna(mean_of_rating)
X1=X
data_des=X.describe()

X1=pd.get_dummies(X1)
X1=X1.drop(['recruitment_channel_sourcing'],axis=1)
X1=X1.drop(['gender_m'],axis=1)
#X1=X1.drop(['gender_f'],axis=1)

X1=X1.drop(['department_Operations'],axis=1)
X1=X1.drop(['education_Below Secondary'],axis=1)
'''   ********************************************************************  '''
#onehot=OneHotEncoder(categorical_features='all')
#X=onehot.fit_transform(X['department']).toarray()
X_train,X_test,y_train,y_test=train_test_split(X1,y,test_size=.20,random_state=0)

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

from xgboost import XGBClassifier
xgb_model=XGBClassifier()

from sklearn.linear_model import LogisticRegression
logistic_regressor=LogisticRegression()
logistic_regressor.fit(X_train,y_train)
y_pred_logistic=logistic_regressor.predict(X_test)
f1_score(y_test,y_pred_logistic)


from sklearn.feature_selection import RFE
cv=RFE(xgb_model,15)
cv.fit(X_train,y_train)
cv.support_
cv.ranking_

17,18,19,1,11
2,4,9,17,18,19

from sklearn.feature_selection import RFECV
cv=RFECV(logistic_regressor,cv=10,scoring=None,verbose=1)
cv.fit(X_train,y_train)
cv.support_
cv.ranking_
cv.n_features_
X.feature_names









