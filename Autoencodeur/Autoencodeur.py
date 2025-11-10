# ===============================================
# 🔍 Modèle Autoencoder pour la détection d'anomalies
# ===============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================
# 1. Chargement des données
# ===============================================
# Utilise les mêmes fichiers que pour les autres modèles
df = pd.read_csv("merged_dataset.csv")  # ton dataset fusionné (Afriware + JDE)
print("✅ Données chargées :", df.shape)

# Variables explicatives et cible
X = df.drop(columns=['is_missing'])   # 'is_missing' = 1 si transaction manquante
y = df['is_missing']

# ===============================================
# 2. Normalisation des variables d’entrée
# ===============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================================
# 3. Construction de l’Autoencoder
# ===============================================
input_dim = X_scaled.shape[1]
encoding_dim = 8  # tu peux ajuster selon la complexité des données

input_layer = Input(shape=(input_dim,))
# Encodeur
encoder = Dense(32, activation='relu')(input_layer)
encoder = Dense(16, activation='relu')(encoder)
encoder = Dense(encoding_dim, activation='relu')(encoder)
# Décodeur
decoder = Dense(16, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='linear')(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# ===============================================
# 4. Entraînement du modèle sur les données normales
# ===============================================
# Entraîne uniquement sur les transactions "présentes"
X_train = X_scaled[y == 0]
history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# ===============================================
# 5. Calcul du score de reconstruction
# ===============================================
X_pred = autoencoder.predict(X_scaled)
reconstruction_error = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Ajout du score au dataframe
df['reconstruction_error'] = reconstruction_error

# ===============================================
# 6. Détermination du seuil optimisé (Tukey)
# ===============================================
Q1 = df['reconstruction_error'].quantile(0.25)
Q3 = df['reconstruction_error'].quantile(0.75)
IQR = Q3 - Q1
seuil_tukey = Q3 + 1.5 * IQR
print(f"📏 Seuil optimisé (Tukey) : {seuil_tukey}")

# Détection des anomalies
df['y_pred'] = (df['reconstruction_error'] > seuil_tukey).astype(int)

# ===============================================
# 7. Évaluation du modèle
# ===============================================
print("\n=== Rapport de classification Autoencoder ===")
print(classification_report(y, df['y_pred']))
print("\nMatrice de confusion :")
print(confusion_matrix(y, df['y_pred']))

# ===============================================
# 8. Visualisation du score d’anomalie
# ===============================================
plt.figure(figsize=(7,5))
sns.boxplot(x=y, y=reconstruction_error)
plt.axhline(y=seuil_tukey, color='r', linestyle='--', label='Seuil optimisé')
plt.title("Autoencoder - Score de reconstruction par classe réelle")
plt.xlabel("Classe réelle (0 = présente, 1 = manquante)")
plt.ylabel("Erreur de reconstruction")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================================
# 9. Sauvegarde des résultats
# ===============================================
df[['numfacture', 'reconstruction_error', 'y_pred']].to_csv("Autoencoder_results.csv", index=False)
print("\n💾 Résultats sauvegardés : Autoencoder_results.csv")
