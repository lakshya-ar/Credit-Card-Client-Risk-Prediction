from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as imPipeline
from imblearn.over_sampling import SMOTE
import pandas as pd

def preprocess_data(data):
    data = data[~data['EDUCATION'].isin([0, 5, 6])]
    data = data[data['MARRIAGE'] != 0]
    data['SEX'] = data['SEX'].astype(str)
    data['EDUCATION'] = data['EDUCATION'].astype(str)
    data['MARRIAGE'] = data['MARRIAGE'].astype(str)
    
    X = data.drop(['Default', 'ID'], axis=1)
    y = data['Default']
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']),
            ('cat', OneHotEncoder(drop='first'), ['SEX', 'EDUCATION', 'MARRIAGE']),
            ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    preprocess_pipeline = imPipeline(steps=[
        ('preprocessor', column_transformer)
    ])
    preprocess_pipeline.fit(X_train, y_train)
    X_train_processed = preprocess_pipeline.transform(X_train)
    X_test_processed = preprocess_pipeline.transform(X_test)
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_processed, y_train)
    return X_train_oversampled, X_test_processed, y_train_oversampled, y_test
