import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsRegressor
from preprocess import preprocess_data

def train_and_save():
    df = pd.read_csv("Toyota (1).csv")
    X_scaled, y, le_model, scaler = preprocess_data(df)

    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_scaled, y)

    with open("knn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("le_model.pkl", "wb") as f:
        pickle.dump(le_model, f)

    print("✅ Model berhasil dilatih dan disimpan!")

if __name__ == "__main__":
    train_and_save()
