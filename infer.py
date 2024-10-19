import torchaudio
import joblib
import numpy as np
import librosa

def load_models(model_path, le_path):
    rf_model = joblib.load(model_path)
    le = joblib.load(le_path)
    return rf_model, le

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

def predict_gender(audio, model):
    features = extract_features(audio)
    prediction = model.predict([features])[0]
    return le.inverse_transform([prediction])[0]

# 示例：预测新音频文件的性别
audio, sr = torchaudio.load('/path/to/your/audio.mp3')
audio = audio.squeeze(0).numpy()
new_audio = {'array': audio, 'sampling_rate': sr}

xgb_model, le = load_models(model_path='models/XGBoost_model.joblib', le_path='models/label_encoder.joblib')

xgb_prediction = predict_gender(new_audio, xgb_model)

print(f"XGBoost predicts: {xgb_prediction}")
