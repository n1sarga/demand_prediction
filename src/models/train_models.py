from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def train_models(X_train, y_train):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=10, fit_intercept=True, solver='saga', max_iter=5000, tol=1e-4, random_state=42),
        'Lasso': Lasso(alpha=3, random_state=42),
        'ElasticNet': ElasticNet(alpha=1, l1_ratio=0.1, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
