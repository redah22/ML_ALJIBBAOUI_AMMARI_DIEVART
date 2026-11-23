# Projet HAI923 : Réalisation d'un Modèle Multimodal CLIP

**Auteurs :**
* Reda AMMARI
* Tala ALJIBBAOUI
* Colin DIEVART

Projet réalisé dans le cadre de l'UE **HAI923 - Apprentissage Automatique 2 / Deep Learning** (Master Informatique, Université de Montpellier).

Ce projet implémente un modèle multimodal inspiré de **CLIP** (*Contrastive Language-Image Pre-training*, Radford et al., 2021) permettant d'aligner des représentations visuelles et textuelles au sein d'un même espace latent partagé.

---

## 🎯 Objectifs

1. **Construire un encodeur visuel** : Conception d'un CNN avec extraction d'embeddings normalisés L2.
2. **Construire un encodeur textuel** : Fine-tuning de **DistilBERT** (`distilbert-base-uncased`) avec extraction de la représentation de phrase via le token `[CLS]`.
3. **Apprentissage contrastif (CLIP)** : Projection conjointe dans un espace commun de dimension $D = 128$, avec fonction de perte contrastive symétrique ($\tau = 0.07$).
4. **Inférence bidirectionnelle Top-5** :
   * **Texte $\rightarrow$ Images** : Recherche des 5 images les plus représentatives d'une requête textuelle.
   * **Image $\rightarrow$ Textes** : Recherche des 5 descriptions textuelles les plus pertinentes pour une image donnée.
5. **Étude comparative** : Évaluation de l'impact de la taille du jeu de données sur la généralisation de l'alignement (`Data_sheep_cat_elephant` à 200, 400 et 600 images/classe).

---

## 📁 Structure du Projet

```text
├── Notebook à utiliser pour le projet-20261123/
│   ├── EnonceNotebookProjetClip.ipynb   # Notebook principal complet (Parties 1, 2, 3)
│   └── EnonceNotebookProjetClip.pdf     # Sujet du notebook
├── rapport_clip.tex                     # Rapport académique en LaTeX (Template officiel)
├── rapport_clip.pdf                     # Rapport compilé (format de soumission)
├── test_partie1.py                      # Script de validation : Classifieur CNN d'images
├── test_partie2.py                      # Script de validation : Classifieur DistilBERT
├── test_partie3.py                      # Script de validation : Assemblage et inférence CLIP
├── .gitignore                           # Fichiers ignorés (datasets, poids, caches, TPs)
└── README.md                            # Documentation du projet
```

---

## 🛠️ Installation et Exécution

### Option 1 : Google Colab (Recommandé pour l'entraînement)
1. Déposer le dossier du projet sur votre Google Drive.
2. Ouvrir le notebook `EnonceNotebookProjetClip.ipynb` dans Google Colab.
3. Activer un accélérateur GPU : `Exécution` > `Modifier le type d'exécution` > `GPU T4`.
4. Exécuter les cellules dans l'ordre. Les paquets et le jeu de données s'initialisent automatiquement.

### Option 2 : En local (Terminal macOS / Linux)

```bash
# 1. Créer et activer un environnement virtuel Python 3.10 ou 3.11
uv venv --python 3.10 .venv
source .venv/bin/activate

# 2. Installer les dépendances avec les versions compatibles Keras 2 / Transformers 4.x
uv pip install tensorflow tf-keras "transformers<4.46" torch pandas numpy matplotlib scikit-learn pillow

# 3. Lancer les scripts de test unitaires
python test_partie1.py   # Test CNN d'images
python test_partie2.py   # Test DistilBERT texte
python test_partie3.py   # Test intégration CLIP & retrieval bidirectionnel
```

---

## 🧠 Architecture Technique

### 1. Encodeur d'images
* Entrée : Image RGB $(224 \times 224 \times 3)$ normalisée dans $[0, 1]$.
* Backbone : 4 blocs convolutifs ($32 \rightarrow 64 \rightarrow 128 \rightarrow 128$ filtres $3 \times 3$, ReLU, `MaxPooling2D(2, 2)`).
* Projection : `Dense(128, activation=None)` suivie de `L2Normalize`.

### 2. Encodeur de texte
* Entrée : Légende tokenisée via `DistilBertTokenizerFast` ($L = 32$, `input_ids` et `attention_mask`).
* Backbone : `TFDistilBertModel` dégelé (`trainable=True`).
* Agrégation : Couche personnalisée `SelectFirstToken` extrayant le token `[CLS]` (dimension 768).
* Projection : `Dense(128, activation=None)` suivie de `L2Normalize`.

### 3. Modèle CLIP et Perte Contrastive
* Les embeddings normalisés $\mathbf{u}$ (image) et $\mathbf{v}$ (texte) sont comparés via une matrice de similarités cosinus mises à l'échelle :
  $$S_{i, j} = \frac{\mathbf{u}_i^\top \mathbf{v}_j}{\tau} \quad (\tau = 0.07)$$
* La couche `ClipLossLayer` calcule la perte d'entropie croisée symétrique (image $\rightarrow$ texte et texte $\rightarrow$ image) et l'injecte via `self.add_loss()`.

---

## 📄 Compilation du Rapport LaTeX

Le rapport respecte le template officiel `template.tex` (marges 3.2 cm, 8 pages max hors annexes) :

```bash
/Library/TeX/texbin/pdflatex -interaction=nonstopmode rapport_clip.tex
```

---

## 📦 Consignes pour le Rendu Final

Le fichier d'archive à déposer sur Moodle doit être nommé `<num_groupe>.zip` (ex: `1.zip`) et contenir :
1. **Rapport PDF** : `1.pdf` (8 pages max + 2 pages d'annexes max).
2. **Notebooks** : `1_projet_clip.ipynb` et sa version exportée `1_projet_clip.pdf`.
3. **Lien vidéo** : Lien YouTube (ou équivalent) vers une présentation synthétique de 8 minutes max (avec apparition initiale des membres du groupe).
