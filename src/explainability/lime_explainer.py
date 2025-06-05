import os
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

def explain_with_lime(estimator, X_train, X_test, feature_names, model_name, output_dir, instance_index=0, num_features=10):
    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="regression"
    )

    # Select the specified instance from the test set for explanation
    instance = X_test[instance_index]

    # Explain the prediction for the selected instance
    exp = explainer.explain_instance(instance, estimator.predict, num_features=num_features)

    # Customize and display Matplotlib plot
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(8, 6)
    plt.title(f"LIME Explanation for {model_name} - Test Instance {instance_index}", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Feature Contribution to Prediction", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f"{output_dir}/{model_name}_lime_instance{instance_index}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

    return exp