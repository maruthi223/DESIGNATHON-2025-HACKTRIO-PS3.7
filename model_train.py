import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
import gc
import tensorflow as tf

DATA_ROOT = Path("C:/Users/kalya/Downloads/model_audio/openmic-2018")  # Correct path
file_path = DATA_ROOT / "openmic-2018.npz"

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

OPENMIC = np.load(file_path, allow_pickle=True)
X, Y_true, Y_mask, sample_key = OPENMIC['X'], OPENMIC['Y_true'], OPENMIC['Y_mask'], OPENMIC['sample_key']

n_fft = 1024
hop_length = 512
n_mfcc = 40

# Clear RAM before starting
gc.collect()
tf.keras.backend.clear_session()

# MFCC Feature Extraction
mfcc_features = []

for i in tqdm(range(len(sample_key))):
    file = sample_key[i]
    dir = file[0:3]
    audio, sr = librosa.load(DATA_ROOT.joinpath('audio', dir, f"{file}.ogg"), sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_features.append(mfcc.T.astype(np.float32))  # Use float32 to reduce memory usageg

X = np.array(mfcc_features, dtype=np.float32)  # Use float32 to reduce memory usage

# Load Class Map
with open(DATA_ROOT.joinpath('class-map.json'), 'r') as f:
    class_map = json.load(f)

split_train = pd.read_csv(DATA_ROOT.joinpath('partitions/split01_train.csv'), header=None)[0]
split_test = pd.read_csv(DATA_ROOT.joinpath('partitions/split01_test.csv'), header=None)[0]

train_set = set(split_train)
test_set = set(split_test)

idx_train, idx_test = [], []

for idx, n in enumerate(sample_key):
    if n in train_set:
        idx_train.append(idx)
    elif n in test_set:
        idx_test.append(idx)

idx_train = np.asarray(idx_train)
idx_test = np.asarray(idx_test)

X_train = X[idx_train]
X_test = X[idx_test]

Y_true_train = Y_true[idx_train]
Y_true_test = Y_true[idx_test]
Y_mask_train = Y_mask[idx_train]
Y_mask_test = Y_mask[idx_test]

models = {}
save_dir = Path(DATA_ROOT.joinpath('models'))
save_dir.mkdir(parents=True, exist_ok=True)

for instrument in class_map:
    print(f"Training {instrument}")
    inst_num = class_map[instrument]

    train_inst = Y_mask_train[:, inst_num]
    X_train_inst = X_train[train_inst]
    Y_true_train_inst = (Y_true_train[train_inst, inst_num] > 0.5).astype(int)

    input_shape = (X_train_inst.shape[1], X_train_inst.shape[2], 1)

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))  # Reduced filters
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))  # Smaller dense layers
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_inst, Y_true_train_inst, epochs=15, batch_size=8, validation_split=0.2)  # Smaller batch size and epochs

    model.save(save_dir.joinpath(f'{instrument}_model.h5'))
    print(f"Saved {instrument} model")
    gc.collect()  # Clear RAM after each model
