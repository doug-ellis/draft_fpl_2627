import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def evaluate_model(X, y, model):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mae, rmse, r2

_DEFAULT_PARAMS = {
    XGBRegressor: {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
    LinearRegression: {},
    Ridge: {'alpha': 1.0},
    Lasso: {'alpha': 0.1},
    ElasticNet: {'alpha': 0.1, 'l1_ratio': 0.5},
}

def _build_model(model_func, params=None):
    if model_func not in _DEFAULT_PARAMS:
        raise ValueError("Unsupported model function")
    merged_params = {**_DEFAULT_PARAMS[model_func], **(params or {})}
    return model_func(**merged_params)

def create_model(training_df, features, model_func, test, random_state=42, model_params=None):
    model_dict = {}
    metrics_dict = {}
    scaler_dict = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        model = _build_model(model_func, model_params)
        training_df_pos = training_df.query('position==@pos').dropna(subset=features + ['total_points_nw']).copy()
        X = training_df_pos[features].copy()
        y = training_df_pos['total_points_nw']
        scaler = StandardScaler()

        if test is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_dict[pos] = model.fit(X_train_scaled, y_train)
            mae, rmse, r2 = evaluate_model(X_test_scaled, y_test, model_dict[pos])
            metrics_dict[pos] = {
                'mae': round(mae, 3),
                'rmse': round(rmse, 3),
                'r2': round(r2, 3),
                'n_test': len(y_test),
                # Stashed so callers (e.g. permutation importance) can reuse this exact
                # held-out split instead of re-splitting/re-fitting.
                'X_test': X_test_scaled,
                'y_test': y_test,
            }
        else:
            X_scaled = scaler.fit_transform(X)
            model_dict[pos] = model.fit(X_scaled, y)
            metrics_dict[pos] = None
        scaler_dict[pos] = scaler
    return model_dict, metrics_dict, scaler_dict

def summarize_metrics(metrics_dict):
    """Sample-count-weighted overall MAE/RMSE/R2 across positions.

    RMSE is combined by pooling squared error (weighted mean of rmse**2, then sqrt),
    the correct way to combine per-group RMSEs (a plain weighted mean of RMSE values
    is not, since RMSE isn't a linear reduction). MAE's weighted mean is directly
    correct since MAE is a linear reduction (mean of |error|). R2 is reported as a
    weighted mean for a rough combined readout, not a true pooled R2 (that would need
    raw residuals rather than the already-reduced per-position values).
    """
    valid = {pos: m for pos, m in metrics_dict.items() if m is not None}
    total_n = sum(m['n_test'] for m in valid.values())
    if total_n == 0:
        return None
    weighted_mae = sum(m['mae'] * m['n_test'] for m in valid.values()) / total_n
    weighted_mse = sum((m['rmse'] ** 2) * m['n_test'] for m in valid.values()) / total_n
    weighted_r2 = sum(m['r2'] * m['n_test'] for m in valid.values()) / total_n
    return {
        'mae': round(weighted_mae, 3),
        'rmse': round(weighted_mse ** 0.5, 3),
        'r2': round(weighted_r2, 3),
        'n_test': total_n,
    }

def compute_permutation_importance(model_dict, metrics_dict, features, n_repeats=10, random_state=42):
    """Per-position permutation importance on each position's held-out test split.

    Values are in "test RMSE increase when this feature is randomly shuffled" units,
    so higher = more important. Model-agnostic: works the same way regardless of
    which model type was used, unlike raw model coefficients.
    """
    importance_dict = {}
    for pos, m in metrics_dict.items():
        if m is None or 'X_test' not in m:
            continue
        result = permutation_importance(
            model_dict[pos], m['X_test'], m['y_test'],
            scoring='neg_root_mean_squared_error',
            n_repeats=n_repeats, random_state=random_state,
        )
        importance_dict[pos] = dict(zip(features, result.importances_mean))
    return importance_dict

def summarize_importance(importance_dict, metrics_dict):
    """Sample-count-weighted overall importance per feature across positions."""
    total_n = sum(metrics_dict[pos]['n_test'] for pos in importance_dict if metrics_dict[pos] is not None)
    if total_n == 0:
        return {}
    weighted = {}
    for pos, imp in importance_dict.items():
        n = metrics_dict[pos]['n_test']
        for feat, val in imp.items():
            weighted[feat] = weighted.get(feat, 0) + val * n / total_n
    return weighted

def predict_scores(prediction_df, features, model_dict, scaler_dict):
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        prediction_df_pos = prediction_df.query('position==@pos').copy()
        X_pred = prediction_df_pos[features]
        X_pred_scaled = scaler_dict[pos].transform(X_pred)
        prediction_df.loc[prediction_df['position']==pos, 'predicted_points'] = model_dict[pos].predict(X_pred_scaled)
    return prediction_df

def compute_feature_contributions(prediction_df, features, model_dict, scaler_dict):
    """Per-player, per-feature breakdown of a linear model's prediction: each
    feature's column holds scaled_value * coef_ (its contribution, in points, to
    that player's predicted score), plus an 'intercept' column and a final
    'predicted_points' column equal to the row's contributions + intercept.

    Only meaningful for linear models exposing .coef_/.intercept_ (ridge, lasso,
    elasticnet, linear) -- raises TypeError for anything else (e.g. xgboost),
    since a tree ensemble has no single per-feature coefficient to multiply
    through like this (that needs SHAP values instead, not implemented here)."""
    pos_frames = []
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        prediction_df_pos = prediction_df.query('position==@pos')
        if prediction_df_pos.empty:
            continue
        model = model_dict[pos]
        if not hasattr(model, 'coef_'):
            raise TypeError(
                f"compute_feature_contributions needs a linear model with .coef_/.intercept_ "
                f"(got {type(model).__name__} for position {pos})."
            )
        X_scaled = scaler_dict[pos].transform(prediction_df_pos[features])
        contrib = pd.DataFrame(X_scaled * model.coef_, columns=features, index=prediction_df_pos.index)
        contrib.insert(0, 'full_name', prediction_df_pos['full_name'].values)
        contrib.insert(1, 'position', pos)
        contrib.insert(2, 'team', prediction_df_pos['team'].values)
        contrib['intercept'] = model.intercept_
        contrib['predicted_points'] = contrib[features].sum(axis=1) + model.intercept_
        pos_frames.append(contrib)
    result = pd.concat(pos_frames, ignore_index=True)
    return result.set_index('full_name')