"""读取并清洗数据集.xls，输出建模数据与变量分布统计。

功能概述：
1. 读取 `数据集.xls`，并优先参考 `文章表格.xls` 的字段命名做变量翻译。
2. 统一空值表达、重命名字段、计算住院时长。
3. 按要求优先使用分类字段：BMI分类3、病变部位分类2、术前前白蛋白/术前白蛋白/术前血红蛋白/皮瓣/术前抗菌药物均使用“分类”列。
4. 以“切口感染”为因变量进行缺失值处理与编码。
5. 统计每个变量的分布并输出表格。
6. 所有输出写入独立子目录 `outputs/data_clean/`。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

READ_PATH = Path("数据集.xlsx")
ARTICLE_TABLE_PATH = Path("文章表格.xlsx")
OUTPUT_DIR = Path("outputs/data_clean")
WRITE_PATH = OUTPUT_DIR / "data_cleaned.csv"
MAPPING_PATH = OUTPUT_DIR / "category_mappings.json"
REVIEW_PATH = OUTPUT_DIR / "data_review_summary.txt"
STAT_TABLE_PATH = OUTPUT_DIR / "group_statistics.tsv"
DISTRIBUTION_PATH = OUTPUT_DIR / "variable_distribution.tsv"
TARGET_COLUMN = "SSI"
FEATURE_MISSING_THRESHOLD = 0.3

COLUMN_RENAME_MAP = {
    "性别": "Sex",
    "年龄": "Age,years",
    "BMI": "BMI",
    "既往头颈部疾病手术外伤史（0无，1有）": "HeadNeckSurgeryTraumaHistory",
    "冠心病（0无，1有）": "Coronary heart disease",
    "高血压（0无，1有）": "Hypertension",
    "外周血管病（0无，1有）": "Peripheral vascular disease",
    "免疫性疾病（0无，1有）": "Immune disease",
    "糖尿病（0无，1有）": "Diabetes",
    "高脂血症（0无，1有）": "Hyperlipidemia",
    "激素类（0无，1有）": "Hormones",
    "吸烟史（0无，1有）": "Smoking",
    "饮酒史（0无，1有）": "Drinking",
    "术前前白蛋白PALB": "PALB before surgery",
    "术前白蛋白ALB": "ALB before surgery",
    "术前血红蛋白HGB": "HBG before surgery",
    "术前口咽拭子": "Preoperative Oropharyngeal Swab",
    "病变部位": "site",
    "ASA评分": "ASA",
    "术前抗菌药物原始数据": "Surgical Antimicrobial Prophylaxis (SAP)",
    "术前抗菌药物分类": "SAP Group",
    "术中输血（有1，无0）": "Transfusion",
    "吻合方式": "Anastomosis type",
    "颈清扫（0无，1有）": "Neck Dissection",
    "术前放疗（0无，1有）": "Radiation therapy before surgery",
    "术前化疗（0无，1有）": "Chemotherapy before surgery",
    "术前同步放化疗": "Concurrent chemoradiotherapy before surgery",
    "腔镜（0无，1有）": "Endoscopy",
    "皮瓣": "Flap",
    "气管造瘘（无0，有1）": "Tracheal Fistula",
    "术后病理": "Postoperative pathology",
    "分化": "Differentiation",
    "最新版pTNM": "Stage",
    "多原发（0否、1是）": "Multiple Primary",
    "术后03天白蛋白ALB": "ALB at postoperative day 3",
    "术后03天前白蛋白PALB": "PALB at postoperative day 3",
    "非计划二次手术": "Unplanned reoperation",
    "肺部感染": "Pulmonary infection",
    "吻合口瘘": "Anastomotic fistula",
    "吻合口瘘确认距术后天数": "Days to fistula confirmation",
    "脂肪液化": "Fat liquefaction",
    "切口感染": "SSI",
    "是否多重耐药": "Multidrug resistance",
    "年龄分类": "Age Group",
    "BMI分类2": "BMIClass2",
    "BMI分类3": "BMI Group",
    "ASA评分分类": "ASA Group",
    "术前咽拭子分类": "Preoperative Oropharyngeal Swab Group",
    "术前前白蛋白分类": "PALB Group",
    "术前白蛋白分类": "ALB Group",
    "术前血红蛋白分类": "HBG Group",
    "皮瓣分类": "Flap Group",
    "病变部位分类2": "Site Group",
}


PREFERRED_CLASS_COLUMNS = {
    "Age,years": "Age Group",
    "BMI": "BMI Group",
    "ASA": "ASA Group",
    "Preoperative Oropharyngeal Swab": "Preoperative Oropharyngeal Swab Group",
    "PALB before surgery": "PALB Group",
    "ALB before surgery": "ALB Group",
    "HBG before surgery": "HBG Group",
    "Flap": "Flap Group",
    "site": "Site Group",
    "Surgical Antimicrobial Prophylaxis (SAP)": "SAP Group",
}

DEDUPLICATED_CLASS_COLUMNS = set(PREFERRED_CLASS_COLUMNS.values()) | {"BMIClass2"}

KEEP_COLUMNS = [
    column
    for column in COLUMN_RENAME_MAP.values()
    if column not in DEDUPLICATED_CLASS_COLUMNS
] + ["LengthOfStay"]

DROP_COLUMNS = [

]


def normalize_missing(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "" or cleaned.lower() in {"nan", "none", "null", "na", "n/a", "不详", "未知"}:
            return pd.NA
        return cleaned
    return value


def load_article_name_map(path: Path) -> dict[str, str]:
    """尝试从文章表格中抽取字段翻译映射（若结构不匹配则回退为空）。"""
    if not path.exists():
        return {}
    try:
        article_df = read_excel_with_fallback(path)
    except Exception:
        return {}

    mapping: dict[str, str] = {}
    columns = [str(col) for col in article_df.columns]
    source_col = next((c for c in columns if "原" in c and "名" in c), None)
    target_col = next((c for c in columns if ("英文" in c or "变量" in c) and "名" in c), None)
    if not source_col or not target_col:
        return {}

    temp = article_df[[source_col, target_col]].dropna()
    for _, row in temp.iterrows():
        cn = str(row[source_col]).strip()
        en = str(row[target_col]).strip()
        if cn and en:
            mapping[cn] = en
    return mapping


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.apply(normalize_missing), errors="coerce")

def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.apply(normalize_missing), errors="coerce")

def read_excel_with_fallback(path: Path) -> pd.DataFrame:
    """读取 Excel，优先规避 .xls 依赖 xlrd 导致的导入错误。"""
    last_error: Exception | None = None
    for engine in ("openpyxl", None):
        try:
            if engine is None:
                return pd.read_excel(path)
            return pd.read_excel(path, engine=engine)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise ImportError(
        "读取 .xls 失败：当前环境缺少可用 Excel 引擎。"
        "请安装 `xlrd>=2.0.1`，或将文件另存为 .xlsx 后重试。"
    ) from last_error

    raise ImportError(
        "读取 .xls 失败：当前环境缺少可用 Excel 引擎。"
        "请安装 `xlrd>=2.0.1`，或将文件另存为 .xlsx 后重试。"
    ) from last_error

def add_length_of_stay(df: pd.DataFrame) -> pd.DataFrame:
    if {"AdmissionDate", "DischargeDate"}.issubset(df.columns):
        admission = safe_to_datetime(df["AdmissionDate"])
        discharge = safe_to_datetime(df["DischargeDate"])
        length_of_stay = (discharge - admission).dt.days
        df["LengthOfStay"] = length_of_stay.where(length_of_stay >= 0, pd.NA)
    return df


def apply_preferred_class_columns(df: pd.DataFrame) -> pd.DataFrame:
    for raw_col, class_col in PREFERRED_CLASS_COLUMNS.items():
        if class_col in df.columns:
            df[raw_col] = df[class_col]
    return df


def keep_only_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """仅保留分析使用字段，并删除冗余列（如 BMIClass2/BMIClass3 原列）。"""
    keep_cols = [column for column in KEEP_COLUMNS if column in df.columns]
    return df[keep_cols].copy()


def build_data_review_summary(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> str:
    total_rows, total_columns = df.shape
    lines = [
        "数据处理前审阅摘要",
        "=" * 24,
        f"样本量: {total_rows}",
        f"字段数: {total_columns}",
        "",
        "各字段缺失情况（按缺失率降序）:",
    ]
    missing_stats = (
        pd.DataFrame({"missing_count": df.isna().sum(), "missing_ratio": df.isna().mean()})
        .sort_values(by=["missing_ratio", "missing_count"], ascending=False)
    )
    for column, row in missing_stats.iterrows():
        lines.append(f"- {column}: 缺失 {int(row['missing_count'])} / {total_rows} ({row['missing_ratio']:.1%})")

    if target_column in df.columns:
        lines.append("")
        lines.append(f"因变量: {target_column}")
        lines.append(str(df[target_column].value_counts(dropna=False).to_dict()))
    return "\n".join(lines)


def report_and_drop_high_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    missing_ratio = df.isna().mean()
    high_missing = [c for c, r in missing_ratio.items() if r > FEATURE_MISSING_THRESHOLD and c != TARGET_COLUMN]
    if high_missing:
        df = df.drop(columns=high_missing)
    return df


def preprocess_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    numeric_candidates = ["Age", "OperationDurationMin", "LengthOfStay"]
    available = []
    for c in numeric_candidates:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            available.append(c)
    return df, available


def split_by_target_and_handle_missing(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return df

    pos_df = df[df[TARGET_COLUMN] == 1].copy()
    neg_df = df[df[TARGET_COLUMN] == 0].copy()

    for col in numeric_columns:
        if col in pos_df.columns:
            pos_df[col] = pos_df[col].fillna(pos_df[col].median(skipna=True))
    pos_df = pos_df.fillna(pos_df.mode(dropna=True).iloc[0] if not pos_df.mode(dropna=True).empty else 0)

    if not neg_df.empty:
        neg_df = neg_df.loc[~neg_df.isna().any(axis=1)].copy()

    return pd.concat([pos_df, neg_df], axis=0).sort_index()


def encode_categorical_columns(df: pd.DataFrame, numeric_columns: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    mappings: dict[str, dict[str, int]] = {}
    for col in df.columns:
        if col in numeric_columns:
            continue
        series = df[col].astype(str)
        categories = sorted(series.dropna().unique().tolist())
        mapping = {v: i for i, v in enumerate(categories)}
        df[col] = series.map(mapping).fillna(0).astype(int)
        mappings[col] = mapping
    return df, mappings

def build_group_statistics_table(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()
    analysis = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    infected = analysis[analysis[TARGET_COLUMN] == 1]
    non_infected = analysis[analysis[TARGET_COLUMN] == 0]

def build_group_statistics_table(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()
    analysis = df[df[TARGET_COLUMN].isin([0, 1])].copy()
    infected = analysis[analysis[TARGET_COLUMN] == 1]
    non_infected = analysis[analysis[TARGET_COLUMN] == 0]

    rows: list[dict[str, str]] = []
    for col in analysis.columns:
        if col == TARGET_COLUMN:
            continue
        if col in numeric_columns:
            rows.append({
                "变量": col,
                "切口感染": f"{infected[col].mean():.2f} ± {infected[col].std(ddof=1):.2f}",
                "非切口感染": f"{non_infected[col].mean():.2f} ± {non_infected[col].std(ddof=1):.2f}",
            })
        else:
            rows.append({"变量": col, "切口感染": "", "非切口感染": ""})
            levels = sorted(analysis[col].dropna().astype(str).unique().tolist())
            for lv in levels:
                c1 = int((infected[col].astype(str) == lv).sum())
                c0 = int((non_infected[col].astype(str) == lv).sum())
                rows.append({
                    "变量": f"  {lv}",
                    "切口感染": f"{c1} ({(c1 / max(len(infected), 1)) * 100:.1f}%)",
                    "非切口感染": f"{c0} ({(c0 / max(len(non_infected), 1)) * 100:.1f}%)",
                })
    return pd.DataFrame(rows)


def build_variable_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    n = len(df)
    for col in df.columns:
        value_counts = df[col].value_counts(dropna=False)
        for value, count in value_counts.items():
            records.append({
                "变量": col,
                "取值": "缺失" if pd.isna(value) else str(value),
                "频数": int(count),
                "占比": f"{(count / n * 100):.2f}%" if n > 0 else "0.00%",
            })
    return pd.DataFrame(records)


def clean_data(read_path: Path = READ_PATH) -> tuple[pd.DataFrame, dict[str, dict[str, int]], pd.DataFrame, pd.DataFrame, str]:
    df = read_excel_with_fallback(read_path)
    df = df.applymap(normalize_missing)

    article_map = load_article_name_map(ARTICLE_TABLE_PATH)
    rename_map = COLUMN_RENAME_MAP.copy()
    rename_map.update(article_map)
    rename_map = {c: rename_map[c] for c in df.columns if c in rename_map}
    df = df.rename(columns=rename_map)

    df = add_length_of_stay(df)
    df = apply_preferred_class_columns(df)
    df = keep_only_analysis_columns(df)
    df = report_and_drop_high_missing_features(df)

    drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df, numeric_columns = preprocess_numeric_columns(df)
    stats_table = build_group_statistics_table(df, numeric_columns)
    distribution_table = build_variable_distribution_table(df)

    review_summary = build_data_review_summary(df)
    df = split_by_target_and_handle_missing(df, numeric_columns)
    df, mappings = encode_categorical_columns(df, numeric_columns)
    return df, mappings, stats_table, distribution_table, review_summary


def save_outputs(df: pd.DataFrame, category_mappings: dict[str, dict[str, int]], stats_table: pd.DataFrame, distribution_table: pd.DataFrame, review_summary: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WRITE_PATH, index=False, encoding="utf-8-sig")
    MAPPING_PATH.write_text(json.dumps(category_mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    REVIEW_PATH.write_text(review_summary, encoding="utf-8")
    if not stats_table.empty:
        stats_table.to_csv(STAT_TABLE_PATH, sep="\t", index=False, encoding="utf-8-sig")
    distribution_table.to_csv(DISTRIBUTION_PATH, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    cleaned_df, mappings, stats, distribution, summary = clean_data()
    save_outputs(cleaned_df, mappings, stats, distribution, summary)
    print(f"数据清洗完成，已输出到：{OUTPUT_DIR}")
