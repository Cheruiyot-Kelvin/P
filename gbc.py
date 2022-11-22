from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv('income_evaluation.csv')
X_train, X_test, y_train, y_test=train_test_split(df.drop('income', axis=1), df['income'], test_size=0.2)
gbc=GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
gbc.fit(X_train,y_train)
print(confusion_matrix(y_test, gbc.predict(X_test)))
print("GBC accuracy is %2.2f"% accuracy_score(y_test, gbc.predict(X_test)))
from sklearn.metrics import classification_report
pred=gbc.predict(X_test)
print(classification_report(y_test, pred))
from sklearn.model_selection import GridSearchCV
grid={
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':np.arrange(100,500,100),
}
gb=GradientBoostingClassifier()
gb_cv=GridSearchCV(gb, grid, cv=4)
gb_cv.fit(X_train, y_train)
print("Best Parameters:", gb_cv.best_params_)
print("Train Score", gb_cv.best_score_)
print("Test Score:", gb_cv.score(X_test, y_test))
grid={'max_depth':[2,3,4,5,6,7]}
gb=GradientBoostingClassifier(learning_rate=0.01, n_estimators=400)
gb_cv=GridSearchCV(gb, grid, cv=4)
gb_cv.fit(X_train,y_train)
print("Best Parameters:", gb_cv.best_params_)
print("Train Score:", gb_cv.best_score_)
print("Test Score:", gb_cv.score(X_test, y_test))
