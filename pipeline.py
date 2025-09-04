import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report
)

# =========================
# 1) Cargar datos y columnas
# =========================
CSV_PATH = "data.csv"
df = pd.read_csv(CSV_PATH)

cols = [c.strip() for c in df.columns]
col_age = next((c for c in cols if "age" in c.lower() or "edad" in c.lower()), None)
col_salary = next((c for c in cols if "salary" in c.lower() or "estimated" in c.lower() or "salario" in c.lower()), None)
col_purchased = next((c for c in cols if "purch" in c.lower() or "compr" in c.lower() or "bought" in c.lower()), None)


def make_preprocess(numeric_features, categorical_features=None):
    num_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=1, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
    transformers = [("num", Pipeline(num_steps), numeric_features)]
    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
        transformers.append(("cat", Pipeline(cat_steps), categorical_features))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)

# =======================================================
# 3) REGRESIÓN: un único Pipeline; GridSearch prueba N modelos
# =======================================================
X_reg_feats_num = [col_age]
X_reg = df[X_reg_feats_num ].copy()
y_reg = df[col_salary].copy()

pipe_reg = Pipeline([
    ("preprocess", make_preprocess(X_reg_feats_num)),
    ("model", LinearRegression())
])

param_grid_reg = [
    {
        "model": [LinearRegression()],
        "model__fit_intercept": [True, False],
        "preprocess__num__poly__degree": [1, 2],
    },
    {
        "model": [Ridge()],
        "model__alpha": [0.1, 1.0, 10.0],
        "preprocess__num__poly__degree": [1, 2],
    },
    {
        "model": [Lasso(max_iter=10000)],
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "preprocess__num__poly__degree": [1, 2],
    },
    {
        "model": [ElasticNet(max_iter=10000)],
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.2, 0.5, 0.8],
        "preprocess__num__poly__degree": [1, 2],
    },
]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

grid_reg = GridSearchCV(
    estimator=pipe_reg,
    param_grid=param_grid_reg,
    scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
    refit="rmse",
    cv=5,
    n_jobs=-1,
    verbose=0
)
grid_reg.fit(Xr_train, yr_train)
best_reg = grid_reg.best_estimator_

yr_pred = best_reg.predict(Xr_test)
rmse = root_mean_squared_error(yr_test, yr_pred)
mae  = mean_absolute_error(yr_test, yr_pred)
r2   = r2_score(yr_test, yr_pred)

print("\n===== REGRESIÓN (EstimatedSalary) =====")
print("Mejor conjunto:", grid_reg.best_params_)
print(f"[Test] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")

# =======================================================
# 4) CLASIFICACIÓN: un único Pipeline; GridSearch prueba N modelos
# =======================================================
X_clf_feats_num = [col_age, col_salary]
X_clf = df[X_clf_feats_num].copy()
y_clf = df[col_purchased].astype(int).copy()

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

pipe_clf = Pipeline([
    ("preprocess", make_preprocess(X_clf_feats_num)),
    ("model", LogisticRegression(max_iter=1000, solver="liblinear"))
])

param_grid_clf = [
    {
        "model": [LogisticRegression(max_iter=1000, solver="liblinear")],
        "model__C": [0.1, 1, 10],
        "model__penalty": ["l1", "l2"],
        "preprocess__num__poly__degree": [1],
    },
    {
        "model": [SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-3, random_state=42)],
        "model__alpha": [1e-4, 1e-3, 1e-2],
        "preprocess__num__poly__degree": [1],
    },
    {
        "model": [LinearSVC(max_iter=5000)],
        "model__C": [0.1, 1, 10],
        "model__dual": [True, False],
        "preprocess__num__poly__degree": [1],
    },
]

grid_clf = GridSearchCV(
    estimator=pipe_clf,
    param_grid=param_grid_clf,
    scoring={"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"},
    refit="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=0
)
grid_clf.fit(Xc_train, yc_train)
best_clf = grid_clf.best_estimator_

yc_pred = best_clf.predict(Xc_test)
if hasattr(best_clf, "predict_proba"):
    yc_score = best_clf.predict_proba(Xc_test)[:, 1]
elif hasattr(best_clf, "decision_function"):
    yc_score = best_clf.decision_function(Xc_test)
else:
    yc_score = None

acc = accuracy_score(yc_test, yc_pred)
f1  = f1_score(yc_test, yc_pred)
auc = roc_auc_score(yc_test, yc_score) if yc_score is not None else np.nan

print("\n===== CLASIFICACIÓN (Purchased) =====")
print("Mejor conjunto:", grid_clf.best_params_)
print(f"[Test] Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
print("Reporte de clasificación:\n", classification_report(yc_test, yc_pred, digits=3))
