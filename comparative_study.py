import os
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam
from tf_keras.utils import register_keras_serializable
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from sklearn.model_selection import train_test_split

# Classes Keras personnalisées
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

# Configuration commune
image_size = (224, 224)
image_shape = image_size + (3,)
text_model_name = "distilbert-base-uncased"
max_len = 32
embed_dim = 128
batch_size = 32
model_dir = "./models_forclip"
os.makedirs(model_dir, exist_ok=True)

tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_name)
AUTOTUNE = tf.data.AUTOTUNE

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
    outputs = distilbert({"input_ids": input_ids, "attention_mask": attention_mask})
    cls_token = SelectFirstToken(name="cls_token")(outputs.last_hidden_state)
    x = Dense(embed_dim, activation=None, name="text_projection")(cls_token)
    embeds = L2Normalize(name="text_embedding")(x)
    return Model(inputs=[input_ids, attention_mask], outputs=embeds, name="text_encoder")

def build_clip_model(image_encoder, text_encoder, temperature=0.07):
    image_input = image_encoder.input
    text_inputs = text_encoder.inputs
    img_embeds = image_encoder(image_input)
    txt_embeds = text_encoder(text_inputs)
    loss_out = ClipLossLayer(temperature=temperature, name="clip_loss")([img_embeds, txt_embeds])
    return Model(inputs=[image_input, *text_inputs], outputs=loss_out, name="clip_model")

def make_clip_dataset(dataframe, tokenizer, dataset_root, batch_size=32, max_len=32, image_size=(224, 224), shuffle=True, seed=42):
    caption_col = "captions" if "captions" in dataframe.columns else "caption"
    image_paths = [str(Path(dataset_root) / p) for p in dataframe["image_path"].astype(str)]
    captions = dataframe[caption_col].fillna("").astype(str).tolist()

    encoded = tokenizer(captions, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
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
        return {"image": img, "input_ids": input_ids, "attention_mask": attention_mask}

    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds.prefetch(AUTOTUNE)

def evaluate_retrieval(image_encoder, text_encoder, df_val, dataset_root):
    caption_col = "captions" if "captions" in df_val.columns else "caption"
    img_paths = [str(Path(dataset_root) / p) for p in df_val["image_path"].astype(str)]
    captions = df_val[caption_col].fillna("").astype(str).tolist()
    
    # 1. Embeddings des textes
    encoded = tokenizer(captions, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    txt_embeds = text_encoder.predict({
        "input_ids": encoded["input_ids"].astype("int32"),
        "attention_mask": encoded["attention_mask"].astype("int32")
    }, batch_size=32, verbose=0)
    
    # 2. Embeddings des images
    def load_img(p):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, image_size)
    
    ds_imgs = tf.data.Dataset.from_tensor_slices(img_paths).map(load_img, num_parallel_calls=AUTOTUNE).batch(32)
    img_embeds = image_encoder.predict(ds_imgs, verbose=0)
    
    # 3. Matrice de similarité cosinus (N x N)
    sim_matrix = img_embeds @ txt_embeds.T
    n = len(df_val)
    
    # Top-1 et Top-5 Image -> Texte
    i2t_top1 = 0
    i2t_top5 = 0
    for i in range(n):
        ranked = np.argsort(sim_matrix[i])[::-1]
        if ranked[0] == i:
            i2t_top1 += 1
        if i in ranked[:5]:
            i2t_top5 += 1
            
    # Top-1 et Top-5 Texte -> Image
    t2i_top1 = 0
    t2i_top5 = 0
    for j in range(n):
        ranked = np.argsort(sim_matrix[:, j])[::-1]
        if ranked[0] == j:
            t2i_top1 += 1
        if j in ranked[:5]:
            t2i_top5 += 1
            
    top1 = (i2t_top1 + t2i_top1) / (2.0 * n) * 100.0
    top5 = (i2t_top5 + t2i_top5) / (2.0 * n) * 100.0
    return top1, top5

def run_experiment(name, dataset_dir, epochs=5):
    print(f"\n========================================================")
    print(f"  Expérimentation : {name} ({dataset_dir})")
    print(f"========================================================")
    
    csv_path = os.path.join(dataset_dir, "captions.csv")
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"Nombre total d'échantillons: {total_samples}")
    
    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    train_ds = make_clip_dataset(df_train, tokenizer, dataset_dir, batch_size=batch_size, max_len=max_len, image_size=image_size, shuffle=True)
    val_ds = make_clip_dataset(df_val, tokenizer, dataset_dir, batch_size=batch_size, max_len=max_len, image_size=image_size, shuffle=False)
    
    img_enc = build_image_encoder(image_shape, embed_dim)
    txt_enc = build_text_encoder(max_len, text_model_name, embed_dim)
    model = build_clip_model(img_enc, txt_enc, temperature=0.07)
    model.compile(optimizer=Adam(learning_rate=1e-4))
    
    weights_path = os.path.join(model_dir, f"clip_{name}.weights.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(weights_path, monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1)
    ]
    
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    final_val_loss = min(history.history["val_loss"])
    epochs_trained = len(history.history["loss"])
    
    # Évaluation Top-1 et Top-5 retrieval
    top1, top5 = evaluate_retrieval(img_enc, txt_enc, df_val, dataset_dir)
    print(f"Résultats {name} -> Val Loss: {final_val_loss:.4f} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
    
    return {
        "name": name,
        "n_samples": total_samples,
        "history": history.history,
        "final_val_loss": final_val_loss,
        "epochs": epochs_trained,
        "top1": top1,
        "top5": top5
    }

if __name__ == "__main__":
    results = []
    
    # 1. Test sur Data_200
    if os.path.exists("./data_200"):
        res_200 = run_experiment("Data_200", "./data_200", epochs=5)
        results.append(res_200)
    
    # 2. Test sur Data_400 (si présent)
    if os.path.exists("./data_400"):
        res_400 = run_experiment("Data_400", "./data_400", epochs=5)
        results.append(res_400)
        
    # 3. Test sur Data_600 (si présent)
    if os.path.exists("./data_600"):
        res_600 = run_experiment("Data_600", "./data_600", epochs=5)
        results.append(res_600)
        
    print("\n================== SYNTHÈSE COMPARATIVE ==================")
    print(f"{'Jeu de données':<15} | {'N paires':<10} | {'Val Loss':<10} | {'Top-1 (%)':<10} | {'Top-5 (%)':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:<15} | {r['n_samples']:<10} | {r['final_val_loss']:<10.4f} | {r['top1']:<10.2f} | {r['top5']:<10.2f}")
        
    # Tracé des courbes de perte comparatives
    plt.figure(figsize=(9, 4))
    for r in results:
        plt.plot(r["history"]["val_loss"], marker='o', label=f"{r['name']} (N={r['n_samples']})")
    plt.title("Comparaison de la convergence (Val Loss) selon le volume de données")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("comparaison_perte_tailles.png", dpi=200)
    print("\nFigure sauvegardée : comparaison_perte_tailles.png")
