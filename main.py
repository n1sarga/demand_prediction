import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocessing import load_data, clean_and_engineer_features, split_features_and_target, scale_features, apply_pca
from src.models.train_models import train_models
from src.evaluation.metrics import evaluate_model
from src.evaluation.plot_curves import plot_learning_curve
from src.explainability.lime_explainer import explain_with_lime

# Paths
data_path = 'data/raw/retail_store_inventory.csv'
processed_path = 'data/preprocessed'
metrics_path = 'reports/figures/metrics'
plots_path = 'reports/figures/plots'
lime_path = 'reports/figures/lime'

# Ensure directories exist
os.makedirs(processed_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)
os.makedirs(plots_path, exist_ok=True)
os.makedirs(lime_path, exist_ok=True)

# Preprocessing
dataset = load_data(data_path)
dataset = clean_and_engineer_features(dataset)
X, y = split_features_and_target(dataset)
X_scaled, scaler = scale_features(X)
X_pca, pca = apply_pca(X_scaled)

# Save processed data
pd.DataFrame(X_scaled, columns=X.columns).to_csv(f'{processed_path}/X_scaled.csv', index=False)
pd.DataFrame(y, columns=['Demand']).to_csv(f'{processed_path}/y.csv', index=False)
pd.DataFrame(X_pca).to_csv(f'{processed_path}/X_pca.csv', index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
models = train_models(X_train, y_train)

# Evaluate + Visualize
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_test, y_test, metrics_path)
    plot_learning_curve(model, X_train, y_train, name, plots_path)
    explain_with_lime(model, X_train, X_test, X.columns.tolist(), name, lime_path)
