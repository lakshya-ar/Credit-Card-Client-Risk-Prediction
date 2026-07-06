import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

def plot_results(results):
    metrics_df = pd.DataFrame(results).T
    metrics_df.sort_values(by='F1-Score', ascending=False, inplace=True)

    metrics_df[['Accuracy', 'F1-Score']].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Defaulter', 'Defaulter'], yticklabels=['Non-Defaulter', 'Defaulter'])
    plt.title(f'{model} Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
