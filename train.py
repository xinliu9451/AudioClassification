import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# 加载已经提取好的特征数据
X = np.load('processed_data/X_common_voice_15_0.npy')
y = np.load('processed_data/y_common_voice_15_0.npy')

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# 训练随机森林模型
# rf_model = RandomForestClassifier(n_estimators=180, criterion='entropy', max_depth=20, max_features='sqrt', random_state=42)
# rf_model.fit(X_train, y_train)

# 训练梯度提升树模型
# gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=20, max_features='sqrt', random_state=42)
# gb_model.fit(X_train, y_train)


# 在测试集上评估模型
xgb_pred = xgb_model.predict(X_test)

le = joblib.load('models/label_encoder.joblib')
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Classification Report:")
print(classification_report(y_test, xgb_pred, target_names=le.classes_))


# 保存模型
def save_models(xgb_model, le, path='models/'):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

    joblib.dump(xgb_model, path + 'XGBoost_model.joblib')
    joblib.dump(le, path + 'label_encoder.joblib')
    print("Models and LabelEncoder saved successfully.")

# 保存模型
save_models(xgb_model, le)