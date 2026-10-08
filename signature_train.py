import os
import random
import itertools
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, \
    Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

DATA_PATH = "data/kaggle_sign_data_1"
TARGET_SIZE = (128, 96)
NUM_USERS = 179
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4


def read_image(path):
    img = Image.open(path).convert("L")
    img = img.resize(TARGET_SIZE)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def l2_normalize(t):
    return tf.math.l2_normalize(t, axis=1)


def get_pairs():
    positive = []
    negative = []
    for user in range(1, NUM_USERS + 1):
        real_folder = os.path.join(DATA_PATH, str(user).zfill(3))
        forg_folder = os.path.join(DATA_PATH, str(user).zfill(3) + "_forg")
        if not os.path.exists(real_folder) or not os.path.exists(forg_folder):
            continue
        real_files = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.lower().endswith(".jpg")]
        forg_files = [os.path.join(forg_folder, f) for f in os.listdir(forg_folder) if f.lower().endswith(".jpg")]

        comb = list(itertools.combinations(real_files, 2))
        random.shuffle(comb)
        for f1, f2 in comb:
            positive.append((read_image(f1), read_image(f2), 1))

        random.shuffle(real_files)
        random.shuffle(forg_files)
        n = min(len(real_files), len(forg_files))
        for i in range(n):
            negative.append((read_image(real_files[i]), read_image(forg_files[i]), 0))

    pairs = positive + negative
    random.shuffle(pairs)
    X1, X2, Y = [], [], []
    for a, b, y in pairs:
        X1.append(a)
        X2.append(b)
        Y.append(y)
    return np.array(X1, dtype=np.float32), np.array(X2, dtype=np.float32), np.array(Y, dtype=np.float32)


X1, X2, Y = get_pairs()

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    X1, X2, Y, test_size=0.25, stratify=Y, random_state=42)


def build_base_network():
    inp = Input(shape=(96, 128, 1))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(128)(x)
    x = Lambda(l2_normalize, output_shape=(128,), name="embedding")(x)
    return Model(inp, x)


base_network = build_base_network()

input_a = Input(shape=(96, 128, 1))
input_b = Input(shape=(96, 128, 1))

embedding_a = base_network(input_a)
embedding_b = base_network(input_b)


def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-9)


distance = Lambda(euclidean_distance, output_shape=(1,), name="distance")([embedding_a, embedding_b])

siamese = Model([input_a, input_b], distance)


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


siamese.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=contrastive_loss)

callbacks = [
    ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
]

history = siamese.fit(
    [x1_train, x2_train], y_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

pred = siamese.predict([x1_test, x2_test], batch_size=BATCH_SIZE).ravel()

best_acc = 0
for threshold in np.arange(0.05, 2.0, 0.01):
    acc = np.mean((pred < threshold) == y_test)
    if acc > best_acc:
        best_acc = acc

print("Точность:", round(best_acc, 4))

siamese.save("signature_siamese.keras", save_format="keras")
siamese.save_weights("model.weights.h5")
print("Модель успешно сохранена!")

scores = -pred.ravel()
auc = roc_auc_score(y_test, scores)
print(f"ROC-AUC: {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, scores)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Signatures')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ROC-AUC.png', dpi=150)
plt.show()