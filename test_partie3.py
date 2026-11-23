import os
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam
from tf_keras.utils import register_keras_serializable
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from sklearn.model_selection import train_test_split

# Classes fournies dans le notebook
@register_keras_serializable()
class SelectFirstToken(layers.Layer):
    def call(self, inputs):
        return inputs[:, 0, :]

@register_keras_serializable()
class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config

@register_keras_serializable(package="clip")
class ClipLossLayer(layers.Layer):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        img, txt = inputs
        logits = tf.matmul(img, txt, transpose_b=True) / self.temperature
        labels = tf.range(tf.shape(logits)[0])
        loss_i = keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        loss_t = keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
        loss = tf.reduce_mean(loss_i + loss_t) / 2.0
        self.add_loss(loss)
        return (img, txt)

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

# Configuration
dataset_dir = "./flickr_subset2"
image_dir = os.path.join(dataset_dir, "images")
model_dir = "./models_forclip"
os.makedirs(model_dir, exist_ok=True)

class_names = ["ball", "bike", "dog", "water"]
image_size = (224, 224)
image_shape = image_size + (3,)
text_model_name = "distilbert-base-uncased"
max_len = 32
embed_dim = 128

tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_name)

# ============================================================
# 1. Définition des encodeurs et du modèle CLIP
# ============================================================

def build_image_encoder(input_shape=(224, 224, 3), embed_dim=128):
    inputs = Input(shape=input_shape, name="image")

    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = MaxPooling2D((2, 2), name="pool3")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv4")(x)
    x = MaxPooling2D((2, 2), name="pool4")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(embed_dim, activation=None, name="image_projection")(x)
    embeds = L2Normalize(name="image_embedding")(x)

    return Model(inputs=inputs, outputs=embeds, name="image_encoder")


def build_text_encoder(max_len=32, model_name="distilbert-base-uncased", embed_dim=128):
    distilbert = TFDistilBertModel.from_pretrained(model_name, from_pt=True)
    distilbert.trainable = True

    input_ids = Input(shape=(max_len,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype="int32", name="attention_mask")

    outputs = distilbert({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    cls_token = SelectFirstToken(name="cls_token")(outputs.last_hidden_state)
    x = Dense(embed_dim, activation=None, name="text_projection")(cls_token)
    embeds = L2Normalize(name="text_embedding")(x)

    return Model(
        inputs=[input_ids, attention_mask],
        outputs=embeds,
        name="text_encoder"
    )


def build_clip_model(image_encoder, text_encoder, temperature=0.07):
    # Les entrées du modèle CLIP correspondent aux entrées de chaque encodeur
    image_input = image_encoder.input
    text_inputs = text_encoder.inputs

    # Calcul des embeddings
    img_embeds = image_encoder(image_input)
    txt_embeds = text_encoder(text_inputs)

    # Couche de perte contrastive
    loss_out = ClipLossLayer(temperature=temperature, name="clip_loss")(
        [img_embeds, txt_embeds]
    )

    clip_model = Model(
        inputs=[image_input, text_inputs[0], text_inputs[1]],
        outputs=loss_out,
        name="clip_model"
    )

    return clip_model

print("=== Construction du modèle CLIP ===")
img_enc = build_image_encoder(image_shape, embed_dim)
txt_enc = build_text_encoder(max_len, text_model_name, embed_dim)
clip = build_clip_model(img_enc, txt_enc, temperature=0.07)
clip.summary()

# Compilation
clip.compile(optimizer=Adam(learning_rate=1e-4))

# ============================================================
# 2. Préparation du jeu de données CLIP
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def make_clip_dataset(
    dataframe,
    tokenizer,
    dataset_root,
    batch_size=32,
    max_len=32,
    image_size=(224, 224),
    shuffle=True,
    seed=42
):
    image_paths = [
        str(Path(dataset_root) / p)
        for p in dataframe["image_path"].astype(str)
    ]
    captions = dataframe["caption"].fillna("").astype(str).tolist()

    encoded = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    input_ids = encoded["input_ids"].astype("int32")
    attention_masks = encoded["attention_mask"].astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((image_paths, input_ids, attention_masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    def load_sample(img_path, input_ids, attention_mask):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, image_size)
        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds

df = pd.read_csv(os.path.join(dataset_dir, "captions.csv"))
df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

train_clip_ds = make_clip_dataset(df_train, tokenizer, dataset_dir, batch_size=32, max_len=max_len, image_size=image_size, shuffle=True)
val_clip_ds = make_clip_dataset(df_val, tokenizer, dataset_dir, batch_size=32, max_len=max_len, image_size=image_size, shuffle=False)

# Vérification du batch
sample_batch = next(iter(train_clip_ds))
print("\n=== Vérification du batch CLIP ===")
print("Keys           :", sample_batch.keys())
print("Image shape    :", sample_batch["image"].shape)
print("Input_ids shape:", sample_batch["input_ids"].shape)

# ============================================================
# 3. Entraînement test (1 epoch)
# ============================================================
print("\n=== Lancement de l'entraînement test CLIP ===")
best_path = os.path.join(model_dir, "best_clip_model.weights.h5")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
]

history = clip.fit(
    train_clip_ds,
    validation_data=val_clip_ds,
    epochs=1,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 4. Vérification rechargement des poids
# ============================================================
print("\n=== Rechargement des poids ===")
image_encoder_reload = build_image_encoder(image_shape, embed_dim)
text_encoder_reload = build_text_encoder(max_len, text_model_name, embed_dim)

clip_reloaded = build_clip_model(
    image_encoder_reload,
    text_encoder_reload,
    temperature=0.07
)
clip_reloaded.load_weights(best_path)
print("Poids rechargés avec succès !")

# ============================================================
# 5. Calcul des embeddings et test de recherche
# ============================================================
print("\n=== Test d'inférence bidirectionnelle ===")
# Test embedding image
sample_img = sample_batch["image"][:1]
img_emb = image_encoder_reload(sample_img)
print(f"Embedding image shape: {img_emb.shape} | Norme: {np.linalg.norm(img_emb[0]):.4f}")

# Test embedding texte
q_text = "A dog running in the water"
encoded_q = tokenizer([q_text], padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
txt_emb = text_encoder_reload({
    "input_ids": encoded_q["input_ids"].astype("int32"),
    "attention_mask": encoded_q["attention_mask"].astype("int32")
})
print(f"Embedding texte shape: {txt_emb.shape} | Norme: {np.linalg.norm(txt_emb[0]):.4f}")

sim = float(np.dot(img_emb[0], txt_emb[0]))
print(f"Similarité cosinus calculée: {sim:.4f}")

print("\n>> TOUS LES TESTS SONT VALIDES POUR LA PARTIE 3 <<")
