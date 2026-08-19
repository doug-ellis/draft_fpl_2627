from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2

def _build_model(model_func):
    if model_func == XGBRegressor:
        return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model_func == LinearRegression:
        return LinearRegression()
    if model_func == Ridge:
        return Ridge(alpha=1.0)
    if model_func == Lasso:
        return Lasso(alpha=0.1)
    if model_func == ElasticNet:
        return ElasticNet(alpha=0.1, l1_ratio=0.5)
    raise ValueError("Unsupported model function")

def create_model(training_df, features, model_func, test):
    model_dict = {}
    rmse_dict = {}
    scaler_dict = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        model = _build_model(model_func)
        training_df_pos = training_df.query('position==@pos').dropna(subset=features + ['total_points_nw']).copy()
        X = training_df_pos[features].copy()
        y = training_df_pos['total_points_nw']
        scaler = StandardScaler()

        if test is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_dict[pos] = model.fit(X_train_scaled, y_train)
            mae, rmse, r2 = evaluate_model(X_test_scaled, y_test, model_dict[pos])
            rmse_dict[pos] = round(rmse, 3)
        else:
            X_scaled = scaler.fit_transform(X)
            model_dict[pos] = model.fit(X_scaled, y)
            rmse_dict[pos] = None
        scaler_dict[pos] = scaler
    return model_dict, rmse_dict, scaler_dict

def predict_scores(prediction_df, features, model_dict, scaler_dict):
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        prediction_df_pos = prediction_df.query('position==@pos').copy()
        X_pred = prediction_df_pos[features]
        X_pred_scaled = scaler_dict[pos].transform(X_pred)
        prediction_df.loc[prediction_df['position']==pos, 'predicted_points'] = model_dict[pos].predict(X_pred_scaled)
    return prediction_df