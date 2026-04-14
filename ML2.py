from __future__ import annotations

"""使用LASSO划分的数据集与多因素Logistic筛选特征进行机器学习建模评估。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.interpolate import PchipInterpolator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_INPUT_DIR = Path("outputs/model_analysis")
TRAIN_PATH = BASE_INPUT_DIR / "train_set.csv"
TEST_PATH = BASE_INPUT_DIR / "test_set.csv"
FEATURES_PATH = BASE_INPUT_DIR / "independent_predictors.json"
TARGET_COLUMN = "SSI"
SEED = 42
CV_FOLDS = 10

OUTPUT_DIR = Path("outputs/ml2")
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
ROC_PATH = OUTPUT_DIR / "roc_curves_smoothed.png"
SHAP_SUMMARY_PATH = OUTPUT_DIR / "shap_summary.png"
SHAP_BAR_PATH = OUTPUT_DIR / "shap_feature_importance_bar.png"


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    independent_features = json.loads(FEATURES_PATH.read_text(encoding="utf-8")).get(
        "independent_predictors", []
    )
    if not independent_features:
        raise ValueError("未在 independent_predictors.json 中找到可用特征。")

    missing_cols = [
        feature
        for feature in independent_features + [TARGET_COLUMN]
        if feature not in train_df.columns or feature not in test_df.columns
    ]
    if missing_cols:
        raise ValueError(f"训练集或测试集缺少字段: {missing_cols}")

    return train_df, test_df, independent_features


def prepare_xy(dataframe: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe[features].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())
    y = dataframe[TARGET_COLUMN].astype(int)
    return X, y


def calculate_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "Youden": sensitivity + specificity - 1,
    }


def smooth_roc_curve(fpr: np.ndarray, tpr: np.ndarray, num_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    unique_fpr, unique_indices = np.unique(fpr_sorted, return_index=True)
    unique_tpr = tpr_sorted[unique_indices]

    if len(unique_fpr) < 3:
        return fpr_sorted, tpr_sorted

    x_new = np.linspace(0, 1, num_points)
    interpolator = PchipInterpolator(unique_fpr, unique_tpr, extrapolate=False)
    y_new = interpolator(x_new)
    y_new = np.nan_to_num(y_new, nan=0.0)
    y_new = np.clip(y_new, 0, 1)
    y_new = np.maximum.accumulate(y_new)
    y_new[0] = 0.0
    y_new[-1] = 1.0
    return x_new, y_new


def train_models(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, dict, dict]:
    models = {
        "GLM": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "RF": RandomForestClassifier(class_weight="balanced", random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=SEED),
        "NNET": MLPClassifier(max_iter=1200, random_state=SEED),
    }
    param_grids = {
        "GLM": {"classifier__C": [0.1, 1, 10]},
        "RF": {"classifier__n_estimators": [100, 200], "classifier__max_depth": [None, 10]},
        "SVM": {"classifier__C": [0.1, 1], "classifier__kernel": ["linear", "rbf"]},
        "NNET": {"classifier__hidden_layer_sizes": [(50,), (100,)], "classifier__alpha": [0.0001, 0.001]},
    }

    metrics_results: dict[str, dict[str, float]] = {}
    roc_results: dict[str, dict[str, np.ndarray | float]] = {}
    fitted_models: dict[str, object] = {}

    for name, model in models.items():
        print(f"\n开始训练模型: {name}")
        pipeline = ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=SEED)),
                ("classifier", model),
            ]
        )
        grid = GridSearchCV(
            pipeline, param_grids[name], cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        probs = best_model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, probs)
        metrics["CV_AUC"] = cross_val_score(best_model, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc").mean()

        fpr, tpr, _ = roc_curve(y_test, probs)
        smoothed_fpr, smoothed_tpr = smooth_roc_curve(fpr, tpr)

        metrics_results[name] = metrics
        roc_results[name] = {
            "AUC": metrics["AUC"],
            "fpr": fpr,
            "tpr": tpr,
            "smooth_fpr": smoothed_fpr,
            "smooth_tpr": smoothed_tpr,
        }
        fitted_models[name] = best_model
        print(
            f"{name} 训练完成 | "
            f"Best Params: {grid.best_params_} | "
            f"AUC={metrics['AUC']:.4f}, CV_AUC={metrics['CV_AUC']:.4f}, "
            f"Sensitivity={metrics['Sensitivity']:.4f}, Specificity={metrics['Specificity']:.4f}, "
            f"Accuracy={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, Youden={metrics['Youden']:.4f}"
        )

    neg, pos = np.bincount(y_train)
    print("\n开始训练模型: GBM")
    gbm = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=neg / pos,
        random_state=SEED,
    )
    gbm_grid = GridSearchCV(
        gbm,
        {
            "colsample_bytree": [0.6, 0.8],
            "learning_rate": [0.01, 0.05],
            "max_depth": [3, 5],
            "min_child_weight": [3, 5],
            "n_estimators": [100, 200],
            "subsample": [0.6, 0.8],
        },
        cv=CV_FOLDS,
        scoring="roc_auc",
        n_jobs=-1,
    )
    gbm_grid.fit(X_train, y_train)
    gbm_best = gbm_grid.best_estimator_
    gbm_probs = gbm_best.predict_proba(X_test)[:, 1]

    gbm_metrics = calculate_metrics(y_test, gbm_probs)
    gbm_metrics["CV_AUC"] = cross_val_score(gbm_best, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc").mean()
    gbm_fpr, gbm_tpr, _ = roc_curve(y_test, gbm_probs)
    gbm_smoothed_fpr, gbm_smoothed_tpr = smooth_roc_curve(gbm_fpr, gbm_tpr)

    metrics_results["GBM"] = gbm_metrics
    roc_results["GBM"] = {
        "AUC": gbm_metrics["AUC"],
        "fpr": gbm_fpr,
        "tpr": gbm_tpr,
        "smooth_fpr": gbm_smoothed_fpr,
        "smooth_tpr": gbm_smoothed_tpr,
    }
    fitted_models["GBM"] = gbm_best
    print(
        f"GBM 训练完成 | "
        f"Best Params: {gbm_grid.best_params_} | "
        f"AUC={gbm_metrics['AUC']:.4f}, CV_AUC={gbm_metrics['CV_AUC']:.4f}, "
        f"Sensitivity={gbm_metrics['Sensitivity']:.4f}, Specificity={gbm_metrics['Specificity']:.4f}, "
        f"Accuracy={gbm_metrics['Accuracy']:.4f}, F1={gbm_metrics['F1']:.4f}, Youden={gbm_metrics['Youden']:.4f}"
    )

    return metrics_results, roc_results, fitted_models


def save_metrics(metrics_results: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(metrics_results, orient="index").reset_index().rename(columns={"index": "Model"})
    df = df[["Model", "AUC", "CV_AUC", "Sensitivity", "Specificity", "Accuracy", "F1", "Youden"]]
    df = df.sort_values("AUC", ascending=False)
    df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    return df


def plot_roc_curves(roc_results: dict[str, dict[str, np.ndarray | float]]) -> None:
    plt.figure(figsize=(10, 8))
    for name, result in roc_results.items():
        plt.plot(
            result["smooth_fpr"],
            result["smooth_tpr"],
            linewidth=2,
            label=f"{name} (AUC={result['AUC']:.3f})",
        )
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title("Smoothed ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def run_shap_analysis(best_model_name: str, best_model: object, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    model_for_shap = best_model
    if hasattr(best_model, "named_steps"):
        model_for_shap = best_model.named_steps["classifier"]

    if best_model_name == "GBM" or best_model_name == "RF":
        explainer = shap.TreeExplainer(model_for_shap, X_train)
        shap_values = explainer.shap_values(X_test)
    else:
        background = shap.sample(X_train, min(100, len(X_train)), random_state=SEED)
        eval_data = shap.sample(X_test, min(200, len(X_test)), random_state=SEED)
        explainer = shap.KernelExplainer(lambda x: model_for_shap.predict_proba(x)[:, 1], background)
        shap_values = explainer.shap_values(eval_data)
        X_test = eval_data

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs, index=X_test.columns).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    shap_importance.plot(kind="barh", color="#2c7fb8")
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"SHAP Feature Importance ({best_model_name})")
    plt.tight_layout()
    plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, features = load_inputs()
    X_train, y_train = prepare_xy(train_df, features)
    X_test, y_test = prepare_xy(test_df, features)

    metrics_results, roc_results, fitted_models = train_models(X_train, y_train, X_test, y_test)
    metrics_df = save_metrics(metrics_results)
    plot_roc_curves(roc_results)

    best_model_name = str(metrics_df.iloc[0]["Model"])
    run_shap_analysis(best_model_name, fitted_models[best_model_name], X_train, X_test)

    print("=== ML2建模完成 ===")
    print(f"使用特征: {features}")
    print(metrics_df.to_string(index=False))
    print(f"评估指标文件: {METRICS_PATH}")
    print(f"ROC图文件: {ROC_PATH}")
    print(f"SHAP图文件: {SHAP_SUMMARY_PATH}, {SHAP_BAR_PATH}")


if __name__ == "__main__":
    main()
