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

    print(f"Original features: {X.shape[1]}")
    print(f"PCA components: {X_pca.shape[1]}")
    print(f"Feature names: {list(X.columns)}")

if __name__ == "__main__":
    main()
