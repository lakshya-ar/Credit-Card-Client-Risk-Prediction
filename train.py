from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

def train_models(X_train, X_test, y_train, y_test):
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, C=0.01, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'SVM': SVC(C=1, kernel='rbf', random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LightGBM': LGBMClassifier(random_state=42)
    }

    results = {}
    
    for m_name, m in models.items():
        print(f"Training {m_name}...")
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        results[m_name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred)
        }
        print(classification_report(y_test, y_pred))

    return results
