import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras.optimizers import Adam
import matplotlib.pyplot as plt

# Configuration
dataset_dir = "./flickr_subset2"
image_dir = os.path.join(dataset_dir, "images")
model_dir = "./models_forclip"
os.makedirs(model_dir, exist_ok=True)

image_size = (224, 224)
image_shape = image_size + (3,)
class_names = ["ball", "bike", "dog", "water"]

print(f"=== Vérification de l'environnement ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs disponibles  : {tf.config.list_physical_devices('GPU')}")

# 1. Générateurs
print("\n=== 1. Initialisation des générateurs ===")
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    image_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_gen = datagen.flow_from_directory(
    image_dir,
    target_size=image_size,
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

batch_x, batch_y = next(train_gen)
print(f"Shape batch images: {batch_x.shape} (attendu: (32, 224, 224, 3))")
print(f"Shape batch labels: {batch_y.shape} (attendu: (32, 4))")
print(f"Classes détectées : {list(train_gen.class_indices.keys())}")

# 2. Modèle
print("\n=== 2. Construction du modèle CNN ===")
def build_image_classifier(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape, name="image_input")

    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="conv1")(inputs)
    x = MaxPooling2D((2, 2), name="pool1")(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv3")(x)
    x = MaxPooling2D((2, 2), name="pool3")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="conv4")(x)
    x = MaxPooling2D((2, 2), name="pool4")(x)

    x = Flatten(name="flatten")(x)
    x = Dense(128, activation="relu", name="dense1")(x)
    x = Dropout(0.5, name="dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="image_classifier")

image_model = build_image_classifier(input_shape=image_shape, num_classes=len(class_names))
image_model.summary()

# 3. Compilation
image_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_path = os.path.join(model_dir, "best_image_classifier.keras")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1)
]

# 4. Entraînement (test rapide sur 5 epochs pour valider la chaîne)
print("\n=== 3. Lancement de l'entraînement test (5 epochs) ===")
epochs = 5
history = image_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

# 5. Évaluation et validation de sauvegarde
print("\n=== 4. Évaluation sur validation ===")
val_loss, val_acc = image_model.evaluate(val_gen, verbose=0)
print(f"Validation loss: {val_loss:.4f} | Validation accuracy: {val_acc:.4f}")

print("\n=== 5. Test du rechargement du modèle ===")
reloaded_model = keras.models.load_model(checkpoint_path)
test_loss, test_acc = reloaded_model.evaluate(val_gen, verbose=0)
print(f"Modèle rechargé avec succès ! Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")
print("\n>> TOUS LES TESTS SONT VALIDES POUR LA PARTIE 1 <<")
