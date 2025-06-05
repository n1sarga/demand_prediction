import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluate_model(name, model, X_train, y_train, X_test, y_test, output_dir):
    # Test set predictions
    y_pred = model.predict(X_test)

    # Standard metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation RMSE (note: scoring='neg_root_mean_squared_error' gives negative values)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse_mean = np.mean(np.abs(cv_scores))
    cv_rmse_std = np.std(np.abs(cv_scores))

    # Combine into metrics dictionary
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_RMSE_Mean': cv_rmse_mean,
        'CV_RMSE_Std': cv_rmse_std
    }

    # Save as DataFrame
    df = pd.DataFrame([metrics])
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/{name}_metrics.csv", index=False)

    print(f"[INFO] Saved evaluation + cross-validation metrics for {name} to {output_dir}")
