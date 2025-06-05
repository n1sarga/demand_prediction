import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_and_engineer_features(dataset):
    dataset.drop(columns=['Store ID', 'Product ID'], inplace=True)
    dataset.rename(columns={
        'Inventory Level': 'Inventory',
        'Units Sold': 'Sales',
        'Units Ordered': 'Order',
        'Demand Forecast': 'Demand',
        'Weather Condition': 'Weather',
        'Holiday/Promotion': 'Promotion',
        'Competitor Pricing': 'Competitor Price'
    }, inplace=True)

    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset['Year'] = dataset['Date'].dt.year
    dataset['Month'] = dataset['Date'].dt.month
    dataset['Day'] = dataset['Date'].dt.day
    dataset.drop(columns=['Date'], inplace=True)

    dataset = pd.get_dummies(dataset, columns=['Category', 'Region', 'Weather', 'Seasonality'], drop_first=True).astype(int)
    return dataset

def split_features_and_target(dataset, target_column='Demand'):
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_pca(X_scaled, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca
