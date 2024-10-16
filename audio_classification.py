import numpy as np
import pandas as pd
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import io

# 加载数据集
dataset = load_dataset("NathanRoll/commonvoice_train_gender_accent_16k", split="train")  # 使用公开的数据来快速测试，实际应用中应该使用自己的数据集

def extract_features(audio):
    # 将音频数据转换为numpy数组
    y = audio['array'].astype(np.float32) / 32768.0
    sr = audio['sampling_rate']

    # 提取特征
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

# 提取特征和标签
features = []
labels = []

for i in range(len(dataset)): # len(dataset)
    print(i)
    sample = dataset[i]
    feature = extract_features(sample['audio'])
    features.append(feature)
    labels.append(sample['gender'])

# 将特征和标签转换为numpy数组
X = np.array(features)
y = np.array(labels)

# 编码标签
le = LabelEncoder()
y = le.fit_transform(y)  # 1是男

print(np.count_nonzero(y == 0))

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 训练梯度提升树模型
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)

print(len(X_test))

# 在测试集上评估模型
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

print("\nGradient Boosting Accuracy:", accuracy_score(y_test, gb_pred))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred, target_names=le.classes_))

# 保存模型
def save_models(rf_model, gb_model, le, path='models/'):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

    joblib.dump(rf_model, path + 'random_forest_model.joblib')
    joblib.dump(gb_model, path + 'gradient_boosting_model.joblib')
    joblib.dump(le, path + 'label_encoder.joblib')
    print("Models and LabelEncoder saved successfully.")

# 加载模型
def load_models(path='models/'):
    rf_model = joblib.load(path + 'random_forest_model.joblib')
    gb_model = joblib.load(path + 'gradient_boosting_model.joblib')
    le = joblib.load(path + 'label_encoder.joblib')
    print("Models and LabelEncoder loaded successfully.")
    return rf_model, gb_model, le

# 保存模型
save_models(rf_model, gb_model, le)

# 加载模型示例
loaded_rf_model, loaded_gb_model, loaded_le = load_models()

# 使用模型进行预测
def predict_gender(audio, model):
    features = extract_features(audio)
    prediction = model.predict([features])[0]
    return loaded_le.inverse_transform([prediction])[0]

# 示例：预测新音频文件的性别
new_audio = dataset[0]['audio']  # 使用数据集中的第一个样本作为示例
print('真实标签：', dataset[0]['gender'])
rf_prediction = predict_gender(new_audio, loaded_rf_model)
gb_prediction = predict_gender(new_audio, loaded_gb_model)

print(f"Random Forest predicts: {rf_prediction}")
print(f"Gradient Boosting predicts: {gb_prediction}")

