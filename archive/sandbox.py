from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, f1_score, adjusted_mutual_info_score
from sklearn.datasets import load_breast_cancer 
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
import pandas as pd
from lib.mlflow import log_to_mlflow


#load the dataset and split it into training and testing sets
dataset = load_breast_cancer()
X=dataset.data
Y=dataset.target
X_train, X_test, y_train, y_test = train_test_split( 
                        X,Y,test_size = 0.30, random_state = 101) 
# train the model on train set without using GridSearchCV 
model = SVC() 
model.fit(X_train, y_train) 
   
   
# print prediction results 
predictions = model.predict(X_test) 
log_to_mlflow(
    model,
    experiment_name='SVC_log',
    model_name='svc',
    tags={'type': 'baseline'},
    params=model.get_params(),
    metrics={
        'accuracy': accuracy_score(y_test, predictions),
        'adjusted_mutual_info_score': adjusted_mutual_info_score(y_test, predictions),
        'f1-score': f1_score(y_test, predictions)
    },
    input_example=X_test[:1],
    input_data=pd.DataFrame(X_train, columns=dataset.feature_names),
    dataset_name="breast_cancer",
)

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['scale', 'auto'],
              'kernel': ['linear']}  
   
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,n_jobs=-1, cv=2) 
   
# fitting the model for grid search 
grid.fit(X_train, y_train) 
 
# print best parameter after tuning 
grid_predictions = grid.predict(X_test) 
   
log_to_mlflow(
    grid,
    experiment_name='Gridsearch_log',
    model_name='gridsearch',
    input_example=X_test[:1],
    input_data=pd.DataFrame(X, columns=dataset.feature_names),
    dataset_name="breast_cancer",
)

# print classification report 


