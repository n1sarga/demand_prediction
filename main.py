import os
import pandas as pd
import joblib

from src.data.preprocessing import (
    load_data,
    clean_and_engineer_features,
    split_features_and_target,
    scale_features,
    apply_pca
)

def main():
    raw_path = 'data/raw/retail_store_inventory.csv'
    processed_dir = 'data/processed'
    model_dir = 'models'

    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load and preprocess
    dataset = load_data(raw_path)
    dataset = clean_and_engineer_features(dataset)

    # Split
    X, y = split_features_and_target(dataset)

    # Scale
    X_scaled, scaler = scale_features(X)

    # PCA
    X_pca, pca = apply_pca(X_scaled)

    # Save preprocessed datasets
    pd.DataFrame(X_scaled, columns=X.columns).to_csv(f'{processed_dir}/X_scaled.csv', index=False)
    pd.DataFrame(y, columns=['Demand']).to_csv(f'{processed_dir}/y.csv', index=False)
    pd.DataFrame(X_pca).to_csv(f'{processed_dir}/X_pca.csv', index=False)

    # Save scaler and PCA model
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(pca, f'{model_dir}/pca.pkl')

    print(f"[INFO] Scaled features saved to: {processed_dir}/X_scaled.csv")
    print(f"[INFO] Target saved to: {processed_dir}/y.csv")
    print(f"[INFO] PCA features saved to: {processed_dir}/X_pca.csv")
    print(f"[INFO] Scaler saved to: {model_dir}/scaler.pkl")
    print(f"[INFO] PCA model saved to: {model_dir}/pca.pkl")
    print(f"[INFO] Original features: {X.shape[1]} → PCA components: {X_pca.shape[1]}")

if __name__ == "__main__":
    main()
