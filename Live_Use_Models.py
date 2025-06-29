import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

data_dict = pickle.load(open('Sign-Language-Interpreter-main/Data/data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# <<==============================================================================================================>>

from sklearn.ensemble import RandomForestClassifier

model_RFC = RandomForestClassifier()

model_RFC.fit(x_train, y_train)

y_predict_RFC = model_RFC.predict(x_test)

# <<==============================================================================================================>>

from sklearn.tree import DecisionTreeClassifier

model_DTC = DecisionTreeClassifier()

model_DTC.fit(x_train, y_train)

y_predict_DTC = model_DTC.predict(x_test)

# <<==============================================================================================================>>

from sklearn.svm import SVC

model_SVM = SVC(probability=True)

model_SVM.fit(x_train, y_train)

y_predict_SVM = model_SVM.predict(x_test)

# <<==============================================================================================================>>

from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors = 30)

model_KNN.fit(x_train, y_train)

y_predict_KNN = model_KNN.predict(x_test)

# <<==============================================================================================================>>

from sklearn.ensemble import GradientBoostingClassifier

model_GBC = GradientBoostingClassifier()

model_GBC.fit(x_train, y_train)

y_predict_GBC = model_GBC.predict(x_test)


# <<==============================================================================================================>>

from sklearn.metrics import accuracy_score

SVM_test_acc = accuracy_score(y_test, y_predict_SVM)
KNN_test_acc = accuracy_score(y_test, y_predict_KNN)
DTC_test_acc = accuracy_score(y_test, y_predict_DTC)
RFC_test_acc = accuracy_score(y_test, y_predict_RFC)
GBC_test_acc = accuracy_score(y_test, y_predict_GBC)

models = pd.DataFrame({
    'Model' : ['SVC', 'KNN', 'Decision Tree', 'Random Forest','Gradient Boost'],
    'Score' : [SVM_test_acc, KNN_test_acc, DTC_test_acc, RFC_test_acc, GBC_test_acc]
})

models.sort_values(by = 'Score', ascending = False)

import plotly.express as px

px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', 
       title = 'Models Comparison')
# <<==============================================================================================================>>

f = open('ML_Models/SVM_model.p', 'wb')
pickle.dump({'model': model_SVM}, f)
f.close()

f = open('ML_Models/KNN_model.p', 'wb')
pickle.dump({'model': model_KNN}, f)
f.close()

f = open('ML_Models/DTC_model.p', 'wb')
pickle.dump({'model': model_DTC}, f)
f.close()

f = open('ML_Models/RFC_model.p', 'wb')
pickle.dump({'model': model_RFC}, f)
f.close()

f = open('ML_Models/GBC_model.p', 'wb')
pickle.dump({'model': model_GBC}, f)
f.close()