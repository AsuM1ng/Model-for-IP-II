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
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    shap_sum = shap_df.sum()
    shap_sum_normalized = (shap_sum - shap_sum.min()) / (shap_sum.max() - shap_sum.min())
    shap_sum = shap_sum.sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    shap_sum.plot(kind='barh')
    plt.xlabel("SHA绝对值总和")
    plt.tight_layout()
    # plt.xlim(0, 1)
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
data = pd.read_csv("data1sisclean.csv")
# data = pd.read_csv("data1.csv")
features = [
    "OperationDurationMin",
    "PreopConcurrentCRT",
    "NeckDissection",
    "IntraopTransfusion",
    "Tracheostomy"
  ]
X = data[features]
y = data['PulmonaryInfection']
seed = 15
ccvv = 10
# ========== 分割数据 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15, stratify=y)
print(f"\n训练集形状：{X_train.shape}, 测试集形状：{X_test.shape}")
print("===== 数据标签分布检查 =====")
label_counts = y_test.value_counts()
print(f"标签分布：{label_counts.to_dict()}")
label_counts = y_train.value_counts()
print(f"标签分布：{label_counts.to_dict()}")
# features = ['Sex', 'Duration of surgery', 'Postoperative 0-3 days CRP',
#             'Post-operative admission to ICU', 'Multi-drug resistance']
# data_train = pd.read_csv("new_train.csv")
# X_train = data_train[features]
# y_train = data_train['Infection status']
# data_test = pd.read_csv("new_test.csv")
# X_test = data_test[features]
# y_test = data_test['Infection status']

# ========== 模型定义与参数 ==========
models = {
    'GLM': LogisticRegression(class_weight='balanced'),
    'RF': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(probability=True, class_weight='balanced'),
    'NNET': MLPClassifier(max_iter=100),
    # 'NB': GaussianNB()
}

param_grids = {
    'GLM': {'classifier__C': [0.1, 1, 10]},
    'RF': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
    'SVM': {'classifier__C': [0.1, 1], 'classifier__kernel': ['linear', 'rbf']},
    'NNET': {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__alpha': [0.0001, 0.001]},
    'NB': {'smote__k_neighbors': [5]}
}

# ========== 通用模型训练与解释 ==========
results = {}
model_probs = {}  # 存储每个模型的预测概率用于DCA
for name, model in models.items():
    print(f"\n训练模型：{name}")
    pipeline = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
         ('smote', SMOTE(random_state=seed)),
        ('classifier', model)
    ])
    grid = GridSearchCV(pipeline, param_grids[name], cv=ccvv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    model_probs[name] = y_prob  # 保存预测概率
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    cv_auc = cross_val_score(best_model, X_train, y_train, cv=ccvv, scoring='roc_auc').mean()
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
    print(f"{name} - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, "
          f"特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1分数: {f1:.4f}, Youden指数: {youden:.4f}")
    # # SHAP 分析
    # X_train_sample = X_train.sample(200, random_state=42)
    # X_test_sample = X_test.sample(200, random_state=42)
    # classifier = best_model.named_steps['classifier']
    #
    # if name == 'RF':
    #     explainer = shap.TreeExplainer(classifier, X_train_sample, check_additivity=False)
    #     shap_values = explainer.shap_values(X_test_sample, check_additivity=False)
    #     shap.summary_plot(shap_values[1], X_test_sample, plot_type="dot", cmap='coolwarm', show=True)
    #     plot_shap_aggregated(shap_values, feature_names=X_train.columns, model_name=name)
    # else:
    #     explainer = shap.KernelExplainer(lambda x: classifier.predict_proba(x)[:, 1], X_train_sample)
    #     shap_values = explainer.shap_values(X_test_sample)
    #     shap.summary_plot(shap_values, X_test_sample, plot_type="dot", cmap='coolwarm', show=True)
    #     plot_shap_aggregated(shap_values, feature_names=X_train.columns, model_name=name)

# ========== XGBoost 模型（不使用 SMOTE） ==========
print("\n训练模型：GBM")

neg, pos = np.bincount(y_train)
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=neg / pos
)
param_grid_xgb = {'colsample_bytree': [0.6], 'learning_rate': [0.01],
                  'max_depth': [5], 'min_child_weight': [5],
                  'n_estimators': [50], 'subsample': [0.6]}
grid = GridSearchCV(xgb_model, param_grid_xgb, cv=ccvv, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
clf = grid.best_estimator_

# 使用 XGBoost 模型进行预测
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
model_probs['GBM'] = y_prob  # 保存预测概率
threshold = 0.4
y_pred = (y_prob >= threshold).astype(int)
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
cv_auc = cross_val_score(clf, X_train, y_train, cv=ccvv, scoring='roc_auc').mean()
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
print(
    f"GBM - AUC: {auc:.4f}, CV_AUC: {cv_auc:.4f}, 敏感性: {sensitivity:.4f}, 特异性: {specificity:.4f}, 准确性: {accuracy:.4f}, F1分数: {f1:.4f}, Youden指数: {youden:.4f}")


def plot_shap_grouped(shap_values, feature_names):
    # shap_values 转为数组
    shap_arr = shap_values[1] if isinstance(shap_values, list) else shap_values

    # 计算每个变量 |SHAP| 总和
    shap_df = pd.DataFrame(np.abs(shap_arr), columns=feature_names)
    shap_sum = shap_df.sum()

    # 变量分组
    preoperative_vars = ['Multi-drug resistance', 'Sex', 'ASA']
    perioperative_vars = ['Duration of surgery', 'Post-operative admission to ICU', 'Postoperative 0-3 days CRP']

    grouped_values = {
        'Preoperative variables': shap_sum[preoperative_vars].sum(),
        'Perioperative variables': shap_sum[perioperative_vars].sum()
    }

    grouped_series = pd.Series(grouped_values)

    # 归一化
    # grouped_series = (grouped_series - grouped_series.min()) / (grouped_series.max() - grouped_series.min())

    # 画图
    plt.figure(figsize=(6, 5))
    grouped_series.sort_values().plot(kind='barh')
    plt.xlabel("SHAP values|")
    plt.tight_layout()

    plt.show()
# SHAP 分析
explainer = shap.TreeExplainer(clf, X_train, check_additivity=False)
shap_values = explainer.shap_values(X_test, check_additivity=False)
shap.summary_plot(shap_values, X_test, plot_type="dot", cmap='coolwarm', show=True)
plot_shap_aggregated(shap_values, feature_names=X.columns, model_name='GBM')
plot_shap_grouped(shap_values, feature_names=X.columns)

# # ========== 保存结果到 CSV ==========
# results_df = pd.DataFrame({
#     '模型': list(results.keys()),
#     'AUC': [results[m]['AUC'] for m in results],
#     'CV_AUC': [results[m]['CV_AUC'] for m in results],
#     '敏感性': [results[m]['Sensitivity'] for m in results],
#     '特异性': [results[m]['Specificity'] for m in results],
#     '准确性': [results[m]['Accuracy'] for m in results],
#     'F1分数': [results[m]['F1'] for m in results],
#     'Youden指数': [results[m]['Youden'] for m in results]
# })
# # results_df.to_csv('final/result.csv', index=False, encoding='utf-8')
#
# from scipy.signal import savgol_filter
#
# thresholds = np.linspace(0.01, 0.99, 200)
#
# def smooth(values):
#     return savgol_filter(values, 21, 3)  # window=21, poly=3
# # ===== 改进版 ROC 曲线 =====
# plt.figure(figsize=(10, 8))
# for name, res in results.items():
#     plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['AUC']:.2f})")
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel("1 - 特异性")
# plt.ylabel("敏感性")
# plt.legend()
# # plt.savefig('new/roc_curves.png')
# plt.show()

# # ===== 改进版 DCA 曲线 =====
# plt.figure(figsize=(8, 6))
#
# for name, y_prob in model_probs.items():
#     nb = calculate_net_benefit(y_test.values, y_prob, thresholds)
#     nb_s = smooth(nb)
#     plt.plot(thresholds, nb_s, linewidth=2.0, label=name)
#
# # treat-none
# plt.plot(thresholds, np.zeros_like(thresholds), 'k--', label='Treat None')
#
# # treat-all
# prevalence = y_test.mean()
# treat_all = thresholds * prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
# plt.plot(thresholds, treat_all, 'k:', label='Treat All')
#
# plt.xlabel("Threshold Probability", fontsize=12)
# plt.ylabel("Net Benefit", fontsize=12)
# plt.title("Decision Curve Analysis", fontsize=14)
# plt.grid(alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()





