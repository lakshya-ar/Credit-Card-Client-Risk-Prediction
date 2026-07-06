from plot.py import plot_results
from train.py import train_models
from data_preprocess import preprocess_data
import pandas as pd

data = pd.read_excel('dataset.xls', header=1)
X_train, X_test, y_train, y_test = preprocess_data(data)

results = train_models(X_train, X_test, y_train, y_test)
plot_results(results)
