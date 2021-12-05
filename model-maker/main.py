# model.py
import numpy as np
from sklearn import datasets
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
def train(X,y):

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    knn = KNeighborsClassifier(n_neighbors=1)

    # fit the model
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Successfully trained model with an accuracy of {acc:.2f}')

    return knn

if __name__ == '__main__':

    iris_data = datasets.load_iris()
    X = iris_data['data']
    y = iris_data['target']
    diabetes_data = pd.read_csv("../diabetes.csv")
    print(diabetes_data)
    df = diabetes_data
    # replacing 0 values with median of that column
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())  # normal distribution
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())  # normal distribution
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())  # skewed distribution
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())  # skewed distribution
    df['BMI'] = df['BMI'].replace(0, df['BMI'].median())  # skewed distribution

    df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')
    diabetes_data = df_selected
    x = diabetes_data
    quantile = QuantileTransformer()
    X = quantile.fit_transform(x)
    df_new = quantile.transform(X)
    df_new = pd.DataFrame(X)
    df_new.columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
    target_name = 'Outcome'
    y = df_new[target_name]  # given predictions - training data
    X = df_new.drop(target_name, axis=1)  # dropping the Outcome column and keeping all other columns as X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # List Hyperparameters to tune
    knn = KNeighborsClassifier()
    n_neighbors = list(range(15, 25))
    p = [1, 2]
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']

    # convert to dictionary
    hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights=weights, metric=metric)

    # Making model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1', error_score=0)
    best_model = grid_search.fit(X_train, y_train)
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    knn_pred = best_model.predict(X_test)

    print("Classification Report is:\n", classification_report(y_test, knn_pred))
    print("\n F1:\n", f1_score(y_test, knn_pred))
    print("\n Precision score is:\n", precision_score(y_test, knn_pred))
    print("\n Recall score is:\n", recall_score(y_test, knn_pred))
    print("\n Confusion Matrix:\n")
    print(confusion_matrix(y_test, knn_pred))


    mdl = train(X,y)
    #
    # # serialize model
    joblib.dump(mdl, '../predictor api/diabetes.mdl')
