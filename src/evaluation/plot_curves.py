import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, model_name, output_dir, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    # Define metrics with their corresponding titles
    metric_titles = {
        "MSE": "Mean Squared Error (MSE)",
        "RMSE": "Root Mean Squared Error (RMSE)", 
        "R²": "R-squared (R²)",
        "MAE": "Mean Absolute Error (MAE)"
    }
    
    # Define colors for training and validation curves
    colors = {"train": 'r', "val": 'g'}
    
    # Compute learning curves for MSE
    train_sizes, train_scores_mse, val_scores_mse = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_squared_error'
    )
    
    # Convert negative MSE to positive MSE
    train_scores_mse = -train_scores_mse
    val_scores_mse = -val_scores_mse
    
    # Compute RMSE from MSE
    train_scores_rmse = np.sqrt(train_scores_mse)
    val_scores_rmse = np.sqrt(val_scores_mse)
    
    # Compute learning curves for R²
    train_sizes, train_scores_r2, val_scores_r2 = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='r2'
    )
    
    # Compute learning curves for MAE
    train_sizes, train_scores_mae, val_scores_mae = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='neg_mean_absolute_error'
    )
    
    # Convert negative MAE to positive MAE
    train_scores_mae = -train_scores_mae
    val_scores_mae = -val_scores_mae
    
    # Store all metrics data
    metrics = {
        "MSE": (train_scores_mse, val_scores_mse),
        "RMSE": (train_scores_rmse, val_scores_rmse),
        "R²": (train_scores_r2, val_scores_r2),
        "MAE": (train_scores_mae, val_scores_mae)
    }
    
    # Calculate mean and standard deviation for all metrics
    means_stds = {metric: (np.mean(train, axis=1), np.std(train, axis=1),
                           np.mean(val, axis=1), np.std(val, axis=1))
                  for metric, (train, val) in metrics.items()}
    
    # Create the plot with 4 subplots
    plt.figure(figsize=(30, 8))
    
    for i, (metric, (train_mean, train_std, val_mean, val_std)) in enumerate(means_stds.items()):
        plt.subplot(1, 4, i + 1)
        
        # Plot mean curves
        plt.plot(train_sizes, train_mean, 'o-', color=colors["train"], label=f'Training {metric}')
        plt.plot(train_sizes, val_mean, 'o-', color=colors["val"], label=f'Validation {metric}')
        
        # Add confidence intervals with fill_between
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.1, color=colors["train"])
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.1, color=colors["val"])
        
        plt.xlabel('Training Set Size')
        plt.ylabel(metric_titles[metric])
        plt.title(f'{metric_titles[metric]} Learning Curve for {type(estimator).__name__}')
        plt.legend(loc='best')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_learning_curve.png")
    plt.close()