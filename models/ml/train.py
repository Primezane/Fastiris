from joblib import dump
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris(return_X_y=True)

X, y = iris[0], iris[1]

knn_pipeline = [('scaling',MinMaxScaler()), ('knn',KNeighborsClassifier(n_neighbors=3))]

pipeline = Pipeline(knn_pipeline)

pipeline.fit(X, y)

dump(pipeline, "./iris_dt_v1.joblib")