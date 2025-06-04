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
    filepath = 'data/raw/retail_store_inventory.csv'

    # Load and preprocess
    dataset = load_data(filepath)
    dataset = clean_and_engineer_features(dataset)

    # Split
    X, y = split_features_and_target(dataset)

    # Scale
    X_scaled, scaler = scale_features(X)

    # PCA
    X_pca, pca = apply_pca(X_scaled)

    # Create directory for processed data
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Save scaled features and target
    pd.DataFrame(X_scaled, columns=X.columns).to_csv(f'{processed_dir}/X_scaled.csv', index=False)
    pd.DataFrame(y, columns=['Demand']).to_csv(f'{processed_dir}/y.csv', index=False)

    # Save PCA-transformed features
    pd.DataFrame(X_pca).to_csv(f'{processed_dir}/X_pca.csv', index=False)

    # Save scaler and PCA objects
    joblib.dump(scaler, f'{processed_dir}/scaler.pkl')
    joblib.dump(pca, f'{processed_dir}/pca.pkl')

    print(f"Saved scaled features to {processed_dir}/X_scaled.csv")
    print(f"Saved target to {processed_dir}/y.csv")
    print(f"Saved PCA features to {processed_dir}/X_pca.csv")
    print(f"Saved scaler to {processed_dir}/scaler.pkl")
    print(f"Saved PCA model to {processed_dir}/pca.pkl")
    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Feature names: {list(X.columns)}")

if __name__ == "__main__":
    main()
