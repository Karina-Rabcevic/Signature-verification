import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

original_from_config = Dense.from_config

@classmethod
def patched_from_config(cls, config):
    config.pop('quantization_config', None)
    return original_from_config(config)

Dense.from_config = patched_from_config

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(
        tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-9
    )


def l2_normalize(t):
    return tf.math.l2_normalize(t, axis=1)


def preprocess(path):
    img = Image.open(path).convert("L")
    img = img.resize((128, 96), Image.Resampling.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


model = load_model(
    "d:/BSU/ODbKuUS/Lab_4_signs/signature_siamese.keras",
    custom_objects={
        "euclidean_distance": euclidean_distance,
        "l2_normalize": l2_normalize,
        "embedding": l2_normalize
    },
    compile=False,
    safe_mode=False
)


path1 = r"D:\BSU\ODbKuUS\Lab_4_signs\data\our_signatures\K\real\Снимок экрана 2026-04-24 104457.png"
path2 = r"D:\BSU\ODbKuUS\Lab_4_signs\data\our_signatures\K\fake\Снимок экрана 2026-04-24 105446.png"
img1 = preprocess(path1)
img2 = preprocess(path2)

distance = model.predict([img1, img2], verbose=0)[0][0]

THRESHOLD = 0.01

if distance < THRESHOLD:
    confidence = max(0, min(100, (1 - distance / THRESHOLD) * 100))
    print("Подпись оригинальная")
    print(f"Уверенность: {confidence:.2f}%")
else:
    confidence = min(100, ((distance - THRESHOLD) / THRESHOLD) * 100)
    print("Подпись поддельная")
    print(f"Расхождение: {confidence:.2f}%")