from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
def plot_shap_aggregated(shap_values, feature_names, model_name):
    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap_df = pd.DataFrame(np.abs(shap_arr), columns=feature_names)
    shap_sum = shap_df.sum().sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    shap_sum.plot(kind='barh')
    plt.title(f"{model_name} - SHAP 特征重要性（原始特征）")
    plt.xlabel("SHAP 绝对值总和")
    plt.tight_layout()
    plt.show()

def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh)) if thresh < 1 else 0
        net_benefits.append(nb)
    return net_benefits

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 加载数据 ==========
data = pd.read_csv("data1.csv")

features = [
    "PreopConcurrentCRT",
    "NeckDissection",
    "AlcoholHistory",
    "Tracheostomy",
    'IntraopTransfusion',
    'OperationDurationMin'
  ]
# X = data.drop(columns=['Post-operative admission to ICU','Multi-drug resistance','Anastomotic fistula','Mechanical ventilation time', 'Mechanical ventilation',
#                        'Intraoperative blood transfusion','Infection status','Preoperative radiotherapy',
#                        'Preoperative chemotherapy','Preoperative concurrent radiochemotherapy'])
X = data[features]
y = data['PulmonaryInfection']

min_ratio = 0.0172   # 保持正/负比约束
min_keep = 2430   # 至少保留样本数（你要求的）

# ========== 分割数据 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\n训练集形状：{X_train.shape}, 测试集形状：{X_test.shape}")
print(f"原比例 - 训练: {len(X_train)/len(data):.2f}, 测试: {len(X_test)/len(data):.2f}")

# ========== 模型定义与参数 ==========
models = {
    'GLM': LogisticRegression(class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced'),
    # 'SVM': SVC(probability=True, class_weight='balanced'),
    'NNET': MLPClassifier(max_iter=1000),
    'NB': GaussianNB()
}

param_grids = {
    'GLM': {'classifier__C': [0.1, 1, 10]},
    'RF': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
    # 'SVM': {'classifier__C': [0.1, 1], 'classifier__kernel': ['linear', 'rbf']},
    'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__alpha': [0.0001, 0.001]},
    'NB': {'smote__k_neighbors': [3, 5]}
}

param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# ========== 初始模型训练 ==========
results = {}
model_probs = {}
best_models = {}

for name, model in models.items():
    print(f"\n训练模型：{name}")
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipeline, param_grids[name], cv=10, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    best_models[name] = best_model
    cv_auc = grid.best_score_
    y_prob = best_model.predict_proba(X_test)[:, 1]
    model_probs[name] = y_prob
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    f1 = f1_score(y_test, y_pred)
    youden = sensitivity + specificity - 1
    results[name] = {
        'AUC': auc,
        'CV_AUC': cv_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': f1,
        'Youden': youden,
        'fpr': fpr,
        'tpr': tpr
    }
    print(f"{name} - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")

# GBM
print("\n训练模型：GBM")
neg, pos = np.bincount(y_train)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=neg / pos
)
grid = GridSearchCV(xgb_model, param_grid_xgb, cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best_gbm = grid.best_estimator_
best_models['GBM'] = best_gbm
cv_auc = grid.best_score_
y_prob = best_gbm.predict_proba(X_test)[:, 1]
model_probs['GBM'] = y_prob
y_pred = (y_prob >= 0.5).astype(int)
auc = roc_auc_score(y_test, y_prob)
fpr, tpr, _ = roc_curve(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
f1 = f1_score(y_test, y_pred)
youden = sensitivity + specificity - 1
results['GBM'] = {
    'AUC': auc,
    'CV_AUC': cv_auc,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Accuracy': accuracy,
    'F1': f1,
    'Youden': youden,
    'fpr': fpr,
    'tpr': tpr
}
print(f"GBM - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1: {f1:.4f}, Youden: {youden:.4f}")

# 选择最佳模型基于CV_AUC
best_name = max(results, key=lambda k: results[k]['CV_AUC'])
best_model = best_models[best_name]
print(f"\n最佳模型（基于CV AUC）：{best_name} with CV AUC: {results[best_name]['CV_AUC']:.4f}")

# 在test上跑最佳模型，print每个样本
y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

print("\n每个test样本的表现：")
for idx in range(len(y_test)):
    actual = y_test.iloc[idx]
    pred = y_pred_test[idx]
    prob = y_prob_test[idx]
    status = "Correct" if actual == pred else "Incorrect"
    print(f"样本 {idx}: 实际标签: {actual}, 预测标签: {pred}, 概率: {prob:.4f}, 表现: {status}")
# ==============================================================
# 1. 对训练集执行 5-fold cross-validation，获得每个样本的 OOF 预测
# ==============================================================

print("\n开始 5-fold 交叉验证判断训练集 Hard / Soft ...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X_train))  # 保存每个训练样本的预测概率

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold+1} ...")

    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    # 使用你上面找到的最佳模型结构（含 SMOTE）
    model_cv = best_model   # ← 来自你上面训练得到的全局最佳模型管线结构

    model_cv.fit(X_tr, y_tr)
    oof_pred[valid_idx] = model_cv.predict_proba(X_val)[:, 1]

# 基于 0.5 阈值划分训练集 Hard/Soft
train_pred_label = (oof_pred >= 0.5).astype(int)
train_actual_label = y_train.values

train_hard_mask = train_pred_label != train_actual_label
train_soft_mask = ~train_hard_mask

X_train_hard = X_train[train_hard_mask]
y_train_hard = y_train[train_hard_mask]

X_train_soft = X_train[train_soft_mask]
y_train_soft = y_train[train_soft_mask]

print(f"训练集 Hard 数量：{len(X_train_hard)}")
print(f"训练集 Soft 数量：{len(X_train_soft)}")

# ==============================================================
# 2. 用最佳模型预测测试集 Hard/Soft（如你之前的代码）
# ==============================================================

print("\n开始判断测试集 Hard / Soft ...")

y_prob_test = best_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= 0.5).astype(int)

test_hard_mask = y_pred_test != y_test.values

X_test_hard = X_test[test_hard_mask]
y_test_hard = y_test[test_hard_mask]

X_test_soft = X_test[~test_hard_mask]
y_test_soft = y_test[~test_hard_mask]

print(f"测试集 Hard 数量：{len(X_test_hard)}")
print(f"测试集 Soft 数量：{len(X_test_soft)}")

# ------------------- 约束性删除 Hard 样本（统一变量 + 保留原始 CA1 列） -------------------

# 合并训练集和测试集 Hard / Soft
X_hard_all = pd.concat([X_train_hard, X_test_hard], axis=0)
y_hard_all = pd.concat([y_train_hard, y_test_hard], axis=0)

X_soft_all = pd.concat([X_train_soft, X_test_soft], axis=0)
y_soft_all = pd.concat([y_train_soft, y_test_soft], axis=0)

# 确保 y_hard_all 是 Series 且 name 为 'PulmonaryInfection'
if isinstance(y_hard_all, pd.Series):
    if y_hard_all.name is None:
        y_hard_all.name = 'PulmonaryInfection'
else:
    y_hard_all = pd.Series(y_hard_all.values, index=X_hard_all.index, name='PulmonaryInfection')

# 原始正/负计数（以全 data 为基准）
orig_pos = int(y.sum())
orig_neg = int(len(y) - orig_pos)

print(f"\n原始正样本：{orig_pos}, 原始负样本：{orig_neg}, 原始正/负比 = {orig_pos/orig_neg:.4f}")

# 设置最小正/负比约束

# ------------------- 约束性删除 Hard（优先删最 hard，保留 CA1 全列，至少保留 2200） -------------------

# 前置假设（这些变量在前面已经存在）：
# oof_pred (numpy array, length == len(X_train))
# y_prob_test (numpy array, length == len(X_test))
# X_train, y_train, X_test, y_test
# X_train_hard, y_train_hard, X_test_hard, y_test_hard
# data (原始 CA1 DataFrame)
# 注意：本段不会修改前面任何模型训练/选择逻辑，仅做删除决策与保存

# ------- 1) 准备训练/测试索引映射（确保不会越界） -------
train_index_list = list(X_train.index)
train_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(train_index_list)}

test_index_list = list(X_test.index)
test_pos_map = {orig_idx: pos for pos, orig_idx in enumerate(test_index_list)}

# ------- 2) 取出 Hard 样本对应的预测概率（稳健获取，不会越界） -------
# 训练集 Hard 的预测概率（从 oof_pred 映射）
hard_prob_train = np.array([])
if len(X_train_hard) > 0:
    train_hard_positions = []
    for orig_idx in X_train_hard.index:
        if orig_idx in train_pos_map:
            train_hard_positions.append(train_pos_map[orig_idx])
        else:
            # 如果出现未映射索引（非常罕见）则跳过该样本并打印警告
            print(f"Warning: 训练集 Hard 的原始索引 {orig_idx} 不在 train_pos_map（跳过）。")
    if len(train_hard_positions) > 0:
        hard_prob_train = oof_pred[np.array(train_hard_positions)]
    else:
        hard_prob_train = np.array([])

# 测试集 Hard 的预测概率（从 y_prob_test 映射）
hard_prob_test = np.array([])
if len(X_test_hard) > 0:
    test_hard_positions = []
    for orig_idx in X_test_hard.index:
        if orig_idx in test_pos_map:
            test_hard_positions.append(test_pos_map[orig_idx])
        else:
            print(f"Warning: 测试集 Hard 的原始索引 {orig_idx} 不在 test_pos_map（跳过）。")
    if len(test_hard_positions) > 0:
        hard_prob_test = y_prob_test[np.array(test_hard_positions)]
    else:
        hard_prob_test = np.array([])

# ------- 3) 合并 Hard 数据及对应的真实标签与 hard_score -------
# 合并 X_hard_all 与 y_hard_all（保留原始索引）
X_hard_all = pd.concat([X_train_hard, X_test_hard], axis=0)
y_hard_all = pd.concat([y_train_hard, y_test_hard], axis=0)

# 保障 y_hard_all 为 Series 且有 name
if isinstance(y_hard_all, pd.Series):
    if y_hard_all.name is None:
        y_hard_all.name = 'PulmonaryInfection'
else:
    y_hard_all = pd.Series(y_hard_all.values, index=X_hard_all.index, name='PulmonaryInfection')

# 计算 hard_score：|pred_prob - true_label|
# 注意顺序：hard_prob_train 对应 X_train_hard.index 的顺序；hard_prob_test 对应 X_test_hard.index 的顺序
hard_score_train = np.abs(hard_prob_train - y_train_hard.values[:len(hard_prob_train)]) if len(hard_prob_train) > 0 else np.array([])
hard_score_test = np.abs(hard_prob_test - y_test_hard.values[:len(hard_prob_test)]) if len(hard_prob_test) > 0 else np.array([])

# 将 score 合并为与 X_hard_all 顺序一致的数组
hard_score_concat = np.concatenate([hard_score_train, hard_score_test]) if (len(hard_score_train) + len(hard_score_test)) > 0 else np.array([])

# 如果长度不匹配（可能由于某些索引被跳过），对齐到 y_hard_all 的索引顺序：
if len(hard_score_concat) != len(y_hard_all):
    # 构造逐个映射的更稳健方法：按 X_hard_all.index 单独映射每个样本的 prob
    mapped_scores = []
    for orig_idx in X_hard_all.index:
        if orig_idx in train_pos_map:
            pos = train_pos_map[orig_idx]
            mapped_scores.append(oof_pred[pos])
        elif orig_idx in test_pos_map:
            pos = test_pos_map[orig_idx]
            mapped_scores.append(y_prob_test[pos])
        else:
            # 若两边都找不到，回退为 0.5（中等 hard）
            mapped_scores.append(0.5)
    # mapped_scores 为预测概率，转换为 hard_score
    mapped_scores = np.array(mapped_scores)
    hard_score_concat = np.abs(mapped_scores - y_hard_all.values)

# 构造 hard_df（包含所有 X 列、标签、hard_score），索引为原始行号
hard_df = pd.concat([X_hard_all, y_hard_all], axis=1)
hard_df['hard_score'] = hard_score_concat

# ------- 4) 排序：按 hard_score 从大到小优先删除（即最 Hard 优先） -------
hard_df = hard_df.sort_values(by='hard_score', ascending=False)

# ------- 5) 约束条件与遍历删除判断 -------
# 使用全量 data 作为基准的正负计数（与你之前一致）
orig_pos = int(y.sum())
orig_neg = int(len(y) - orig_pos)

if orig_neg == 0:
    raise ValueError("原始负样本为0，无法进行正/负比约束。")

print(f"\n原始正样本：{orig_pos}, 原始负样本：{orig_neg}, 原始正/负比 = {orig_pos/orig_neg:.4f}")


current_pos = orig_pos
current_neg = orig_neg
current_total = current_pos + current_neg

allowed_remove_ids = []

for idx, row in hard_df.iterrows():
    label = int(row[y_hard_all.name])

    # 若删除会使总数低于 min_keep，则停止删除
    if current_total - 1 < min_keep:
        break

    new_pos = current_pos - (1 if label == 1 else 0)
    new_neg = current_neg - (1 if label == 0 else 0)

    # 删除后需满足 new_neg>0 且 new_pos/new_neg >= min_ratio
    if new_neg > 0 and (new_pos / new_neg) >= min_ratio:
        allowed_remove_ids.append(idx)
        current_pos = new_pos
        current_neg = new_neg
        current_total -= 1
    else:
        # 不满足约束，跳过（保留）
        continue

# ------- 6) 统计并在原始 data 上删除 allowed_remove_ids -------
print(f"\n允许删除的 Hard 样本数：{len(allowed_remove_ids)}")
removed_pos = int(y_hard_all.loc[allowed_remove_ids].sum()) if len(allowed_remove_ids) > 0 else 0
removed_neg = len(allowed_remove_ids) - removed_pos
print(f"将删除的阳性数量：{removed_pos}")
print(f"将删除的阴性数量：{removed_neg}")
print(f"删除后预计剩余样本数：{current_total}")

# 在原始 data（ca1）上删除这些样本（保留所有列）
hard_index = list(allowed_remove_ids)  # 这里的 allowed_remove_ids 本身就是原始行索引
ca1_cleaned = data.drop(index=hard_index, errors='ignore').copy()

print(f"\n原始数据总行数: {len(data)}")
print(f"实际删除行数 (从 data 中)：{len(data) - len(ca1_cleaned)}")
print(f"清理后保留行数: {len(ca1_cleaned)}")
print(f"清理后正例数: {int(ca1_cleaned['PulmonaryInfection'].sum())}")
print(f"清理后负例数: {len(ca1_cleaned) - int(ca1_cleaned['PulmonaryInfection'].sum())}")

# 保存最终完整 CA1（含全部列）
outname = "data1sisclean.csv"
ca1_cleaned.to_csv(outname, index=False)
print(f"\n已保存：{outname}（含 CA1 所有原始列）")



