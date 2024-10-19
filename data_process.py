from datasets import load_dataset, Audio, concatenate_datasets
import random
from collections import Counter
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
from tqdm import tqdm

# 加载数据
datasets = load_dataset("mozilla-foundation/common_voice_15_0", 'zh-CN', split=["train", "test", "validation", "invalidated", "other"])

# 去除不要的列，我的目标是预测性别，所以只保留性别和音频列
train_dataset = datasets[0].remove_columns(['client_id', 'path','sentence','up_votes','down_votes','age','accent','locale','segment','variant'])
test_dataset = datasets[1].remove_columns(['client_id', 'path','sentence','up_votes','down_votes','age','accent','locale','segment','variant'])
validation_dataset = datasets[2].remove_columns(['client_id', 'path','sentence','up_votes','down_votes','age','accent','locale','segment','variant'])
invalidated_dataset = datasets[3].remove_columns(['client_id', 'path','sentence','up_votes','down_votes','age','accent','locale','segment','variant'])
other_dataset = datasets[4].remove_columns(['client_id', 'path','sentence','up_votes','down_votes','age','accent','locale','segment','variant'])

# 重采样为16000
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16_000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
validation_dataset = validation_dataset.cast_column("audio", Audio(sampling_rate=16_000))
invalidated_dataset = invalidated_dataset.cast_column("audio", Audio(sampling_rate=16_000))
other_dataset = other_dataset.cast_column("audio", Audio(sampling_rate=16_000))

# 过滤掉没有gender标签的数据，因为很多数据没有性别的标签，这些数据对于我的目标是没有用的
def filter_empty_gender_numpy(examples):
    return np.array(examples['gender']) != ''

# 应用过滤，这里的batch_size是为了加速
filtered_train_dataset = train_dataset.filter(filter_empty_gender_numpy, batched=True, batch_size=100)
filtered_test_dataset = test_dataset.filter(filter_empty_gender_numpy, batched=True, batch_size=100)
filtered_validation_dataset = validation_dataset.filter(filter_empty_gender_numpy, batched=True, batch_size=100)
filtered_invalidated_dataset = invalidated_dataset.filter(filter_empty_gender_numpy, batched=True, batch_size=100)
filtered_other_dataset = other_dataset.filter(filter_empty_gender_numpy, batched=True, batch_size=100)

# 合并数据集
filtered_dataset = concatenate_datasets([filtered_train_dataset,filtered_test_dataset,filtered_validation_dataset, filtered_invalidated_dataset, filtered_other_dataset])

def balance_gender_data(dataset):
    # 获取所有样本的索引
    male_indices = [i for i, gender in enumerate(dataset['gender']) if gender == 'male']
    female_indices = [i for i, gender in enumerate(dataset['gender']) if gender == 'female']

    # 随机选择与female样本数量相同的male样本
    male_downsampled = random.sample(male_indices, len(female_indices))

    # 合并下采样后的male样本索引和所有female样本索引
    balanced_indices = male_downsampled + female_indices

    # 创建新的平衡数据集
    balanced_dataset = dataset.select(balanced_indices)

    # 打乱数据集的顺序
    balanced_dataset = balanced_dataset.shuffle(seed=42)

    # 打印平衡后的性别分布
    gender_counts = Counter(balanced_dataset['gender'])
    print("Balanced gender distribution:")
    print(f"Male: {gender_counts['male']}")
    print(f"Female: {gender_counts['female']}")

    return balanced_dataset

# 平衡一下两个标签对应的数据量，因为数据集中在某一个标签会导致模型过拟合
dataset = balance_gender_data(filtered_dataset)


def extract_features(audio):
    y = audio['array'].astype(np.float32) / 32768.0
    sr = audio['sampling_rate']

    # 提取特征
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def process_sample(sample):
    feature = extract_features(sample['audio'])
    return feature, sample['gender']

# 使用多进程加速，并显示进度条
def parallel_feature_extraction(dataset, num_workers=2):
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_sample, dataset), total=len(dataset)))

    # 分离特征和标签
    features, labels = zip(*results)
    return np.array(features), np.array(labels)

# 提取特征和标签
X, y_labels = parallel_feature_extraction(dataset, num_workers=2)

# 编码标签
le = LabelEncoder()
y = le.fit_transform(y_labels)  # 1是男

# 将处理好的数据保存到本地
np.save('processed_data/X_common_voice_15_0.npy', X)
np.save('processed_data/y_common_voice_15_0.npy', y)