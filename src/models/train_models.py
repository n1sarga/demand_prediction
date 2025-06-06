from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import xgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor

def train_models(X_train, y_train):
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=10, fit_intercept=True, solver='saga', max_iter=5000, tol=1e-4, random_state=42),
        'Lasso': Lasso(alpha=3, random_state=42),
        'ElasticNet': ElasticNet(alpha=1, l1_ratio=0.1, random_state=42),
        'SupportVectorRegressor': SVR(kernel='rbf'),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, validation_fraction=0.1, random_state=42),
        'KNNRegressor': KNeighborsRegressor(n_neighbors=5),
        'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'XGBoostRegressor': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=123),
        'LightGBMRegressor': LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
        'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
