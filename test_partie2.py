import os
os.environ["MPLCONFIGDIR"] = "/tmp"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.layers import Input, Dense, Dropout
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam
from tf_keras.utils import register_keras_serializable
from transformers import DistilBertTokenizerFast, TFDistilBertModel
from sklearn.model_selection import train_test_split

# Définition de la classe SelectFirstToken telle que définie en cellule 17 du notebook
@register_keras_serializable()
class SelectFirstToken(layers.Layer):
    """
    Récupère la représentation du premier token de la séquence.
    """
    def call(self, inputs):
        return inputs[:, 0, :]

# Paramètres
dataset_dir = "./flickr_subset2"
captions_csv = os.path.join(dataset_dir, "captions.csv")
model_dir = "./models_forclip"
os.makedirs(model_dir, exist_ok=True)

class_names = ["ball", "bike", "dog", "water"]
text_model_name = "distilbert-base-uncased"
max_len = 32
batch_size = 16

print("=== 1. Chargement et préparation des données ===")
df = pd.read_csv(captions_csv)
print(f"Nombre total de légendes: {len(df)}")

# Mapping des classes vers entiers selon l'ordre alphabétique
label_to_id = {name: i for i, name in enumerate(class_names)}
df["label_id"] = df["label"].map(label_to_id)

texts = df["caption"].fillna("").astype(str).tolist()
labels = df["label_id"].values

# Découpage stratifié apprentissage / validation (80% / 20%)
X_train_txt, X_val_txt, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print(f"Taille apprentissage: {len(X_train_txt)} | Taille validation: {len(X_val_txt)}")

# Tokenisation
tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_name)

def make_text_dataset(texts, labels, batch_size=16, shuffle=False):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )
    
    ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids": encoded["input_ids"].astype("int32"),
            "attention_mask": encoded["attention_mask"].astype("int32")
        },
        labels.astype("int32")
    ))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_text_dataset(X_train_txt, y_train, batch_size=batch_size, shuffle=True)
val_ds = make_text_dataset(X_val_txt, y_val, batch_size=batch_size, shuffle=False)

# Vérification d'un batch
sample_x, sample_y = next(iter(train_ds))
print(f"Batch input_ids shape     : {sample_x['input_ids'].shape}")
print(f"Batch attention_mask shape: {sample_x['attention_mask'].shape}")
print(f"Batch labels shape        : {sample_y.shape}")

print("\n=== 2. Construction du classifieur DistilBERT ===")
def build_text_classifier(max_len=32, model_name="distilbert-base-uncased", num_classes=4, lr=2e-5):
    distilbert = TFDistilBertModel.from_pretrained(model_name, from_pt=True)
    distilbert.trainable = True

    input_ids = Input(shape=(max_len,), dtype="int32", name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype="int32", name="attention_mask")

    outputs = distilbert({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    cls_token = SelectFirstToken(name="cls_token")(outputs.last_hidden_state)

    x = Dropout(0.2, name="dropout1")(cls_token)
    x = Dense(128, activation="relu", name="dense_text")(x)
    x = Dropout(0.2, name="dropout2")(x)
    out = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(
        inputs=[input_ids, attention_mask],
        outputs=out,
        name="text_classifier"
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

text_model = build_text_classifier(max_len=max_len, model_name=text_model_name, num_classes=len(class_names))
text_model.summary()

print("\n=== 3. Entraînement test (2 epochs rapides) ===")
checkpoint_path = os.path.join(model_dir, "best_text_classifier.keras")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]

history = text_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=callbacks,
    verbose=1
)

print("\n=== 4. Évaluation sur le jeu de validation ===")
val_loss, val_acc = text_model.evaluate(val_ds, verbose=0)
print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

print("\n=== 5. Test du rechargement du modèle sauvegardé ===")
reloaded_text_model = keras.models.load_model(
    checkpoint_path,
    custom_objects={"SelectFirstToken": SelectFirstToken, "TFDistilBertModel": TFDistilBertModel}
)
test_loss, test_acc = reloaded_text_model.evaluate(val_ds, verbose=0)
print(f"Modèle rechargé avec succès ! Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
print("\n>> TOUS LES TESTS SONT VALIDES POUR LA PARTIE 2 <<")
