import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    df = df.dropna()
    df = df[df['price'] > 0]

    le_model = LabelEncoder()
    df['model_enc'] = le_model.fit_transform(df['model'])

    X = df[['model_enc', 'year', 'mileage', 'tax', 'mpg', 'engineSize']]
    y = df['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, le_model, scaler
