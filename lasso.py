from __future__ import annotations

"""分层抽样 + LASSO 特征筛选 + 相关性热力图。"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

DATA_PATH = Path("data1.csv")
TARGET_COLUMN = "PulmonaryInfection"
TEST_SIZE = 0.3
RANDOM_STATE = 42
CORRELATION_THRESHOLD = 0.6
OUTPUT_DIR = Path("analysis_outputs")
TRAIN_PATH = OUTPUT_DIR / "train_set.csv"
TEST_PATH = OUTPUT_DIR / "test_set.csv"
LASSO_COEF_PATH = OUTPUT_DIR / "lasso_coefficients.csv"
LASSO_SELECTED_PATH = OUTPUT_DIR / "lasso_selected_features.json"
CORRELATION_MATRIX_PATH = OUTPUT_DIR / "lasso_spearman_matrix.csv"
CORRELATION_HEATMAP_PATH = OUTPUT_DIR / "lasso_spearman_heatmap.png"
FILTERED_FEATURES_PATH = OUTPUT_DIR / "selected_features_after_correlation.json"
SPLIT_SUMMARY_PATH = OUTPUT_DIR / "split_summary.json"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到 {DATA_PATH}，请先运行 data_clean1.py。")
    return pd.read_csv(DATA_PATH)


def prepare_xy(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe.drop(columns=[TARGET_COLUMN]).copy()
    y = dataframe[TARGET_COLUMN].astype(int)
    for column in tqdm(X.columns, desc="LASSO输入矩阵预处理", leave=False):
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].fillna(X[column].median())
    return X, y


def fit_lasso(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, Pipeline]:
    tqdm.write("开始在训练集上执行LASSO回归特征筛选...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            cv=10,
            scoring="roc_auc",
            max_iter=5000,
            random_state=RANDOM_STATE,
            refit=True,
        )),
    ])
    pipeline.fit(X_train, y_train)
    coefficients = pipeline.named_steps["classifier"].coef_.ravel()
    coefficient_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values(["abs_coefficient", "feature"], ascending=[False, True])
    return coefficient_df, pipeline


def correlation_filter(X_train: pd.DataFrame, coefficient_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame, list[dict[str, float | str]]]:
    selected_df = coefficient_df.loc[coefficient_df["coefficient"] != 0].copy()
    if selected_df.empty:
        raise ValueError("LASSO未筛选出任何非零系数变量。")
    selected_features = selected_df["feature"].tolist()
    corr_matrix = X_train[selected_features].corr(method="spearman").fillna(0)
    coef_lookup = selected_df.set_index("feature")["abs_coefficient"].to_dict()

    to_remove: set[str] = set()
    removed_pairs: list[dict[str, float | str]] = []
    for i, left_feature in enumerate(tqdm(selected_features, desc="相关性过滤", leave=False)):
        if left_feature in to_remove:
            continue
        for right_feature in selected_features[i + 1:]:
            if right_feature in to_remove:
                continue
            corr_value = abs(float(corr_matrix.loc[left_feature, right_feature]))
            if corr_value > CORRELATION_THRESHOLD:
                left_coef = coef_lookup[left_feature]
                right_coef = coef_lookup[right_feature]
                drop_feature = left_feature if left_coef < right_coef else right_feature
                keep_feature = right_feature if drop_feature == left_feature else left_feature
                to_remove.add(drop_feature)
                removed_pairs.append({
                    "keep_feature": keep_feature,
                    "drop_feature": drop_feature,
                    "spearman_abs": corr_value,
                    "keep_abs_coefficient": float(coef_lookup[keep_feature]),
                    "drop_abs_coefficient": float(coef_lookup[drop_feature]),
                })
    final_features = [feature for feature in selected_features if feature not in to_remove]
    return final_features, corr_matrix.loc[selected_features, selected_features], removed_pairs


def save_heatmap(corr_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(max(8, 0.7 * len(corr_matrix.columns)), max(6, 0.7 * len(corr_matrix.columns))))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Spearman Correlation Heatmap for LASSO-selected Features")
    plt.tight_layout()
    plt.savefig(CORRELATION_HEATMAP_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(full_data: pd.DataFrame, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, coefficient_df: pd.DataFrame, final_features: list[str], corr_matrix: pd.DataFrame, removed_pairs: list[dict[str, float | str]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_data.loc[X_train.index].to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    full_data.loc[X_test.index].to_csv(TEST_PATH, index=False, encoding="utf-8-sig")
    coefficient_df.to_csv(LASSO_COEF_PATH, index=False, encoding="utf-8-sig")
    corr_matrix.to_csv(CORRELATION_MATRIX_PATH, encoding="utf-8-sig")
    save_heatmap(corr_matrix)

    LASSO_SELECTED_PATH.write_text(json.dumps({
        "selected_by_lasso": coefficient_df.loc[coefficient_df["coefficient"] != 0, "feature"].tolist(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    FILTERED_FEATURES_PATH.write_text(json.dumps({
        "target": TARGET_COLUMN,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "final_features": final_features,
        "removed_pairs": removed_pairs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    SPLIT_SUMMARY_PATH.write_text(json.dumps({
        "target": TARGET_COLUMN,
        "overall_distribution": full_data[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict(),
        "train_distribution": y_train.value_counts(normalize=True).sort_index().to_dict(),
        "test_distribution": y_test.value_counts(normalize=True).sort_index().to_dict(),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with tqdm(total=5, desc="lasso总进度") as progress:
        dataframe = load_dataset()
        progress.update(1)

        X, y = prepare_xy(dataframe)
        progress.update(1)

        tqdm.write("开始按7:3进行分层抽样...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        progress.update(1)

        coefficient_df, _ = fit_lasso(X_train, y_train)
        final_features, corr_matrix, removed_pairs = correlation_filter(X_train, coefficient_df)
        progress.update(1)

        save_outputs(dataframe, X_train, X_test, y_train, y_test, coefficient_df, final_features, corr_matrix, removed_pairs)
        progress.update(1)

    print("=== LASSO筛选完成 ===")
    print(f"训练集: {len(y_train)}，测试集: {len(y_test)}")
    print(f"LASSO保留变量数: {(coefficient_df['coefficient'] != 0).sum()}")
    print(f"相关性过滤后保留变量数: {len(final_features)}")
