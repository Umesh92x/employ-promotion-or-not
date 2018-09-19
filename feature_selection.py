# Feature selection, important variable selection
from sklearn.feature_selection import RFE
cv=RFE(xgb_model,15)
cv.fit(X_train,y_train)
cv.support_
cv.ranking_

from sklearn.feature_selection import RFECV
cv=RFECV(logistic_regressor,cv=10,scoring=None,verbose=1)
cv.fit(X_train,y_train)
cv.support_
cv.ranking_
cv.n_features_
X.feature_names
