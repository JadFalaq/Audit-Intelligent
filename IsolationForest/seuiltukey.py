# ===============================
#  Isolation Forest - Détection d’anomalies
#  Jad Falaq | Projet de détection de transactions manquantes
# ===============================

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1️⃣ Chargement des données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- 2️⃣ Création des labels : 1 = transaction manquante ---
book1['in_jde911'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_jde311'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- 3️⃣ Nettoyage des dates et types ---
for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')

if 'TypeFacture' in book1.columns:
    book1['TypeFacture'] = book1['TypeFacture'].astype(str)

# --- 4️⃣ Sélection des variables numériques ---
X = book1[['CodeClient', 'CompteProduit', 'CentreAnalyse', 'MontantHT', 'MontantTTC', 'Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- 5️⃣ Application du modèle Isolation Forest ---
iso = IsolationForest(
    n_estimators=200,
    contamination='auto',  # proportion d'anomalies estimée automatiquement
    random_state=42
)
iso.fit(X)

# --- 6️⃣ Calcul du score d’anomalie ---
book1['anomaly_score'] = iso.decision_function(X)
book1['anomaly_flag'] = iso.predict(X)  # -1 = anomalie, 1 = normale
book1['anomaly_flag'] = book1['anomaly_flag'].map({1: 0, -1: 1})  # 1 = anomalie

# --- 7️⃣ Calibration du seuil à partir des vraies classes ---
y_true = book1['in_jde911'].values  # transactions manquantes (1) ou présentes (0)
scores_inv = -book1['anomaly_score'].values  # inversion (car bas = anomalie)

precisions, recalls, thresholds = precision_recall_curve(y_true, scores_inv)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-6)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"\n✅ Seuil optimisé (basé sur F1-score) = {best_thresh:.5f}")
print(f"→ Précision : {precisions[best_idx]:.3f} | Rappel : {recalls[best_idx]:.3f} | F1-score : {f1_scores[best_idx]:.3f}")

# --- 8️⃣ Classification selon le seuil optimisé ---
book1['predicted_anomaly'] = (scores_inv >= best_thresh).astype(int)

# --- 9️⃣ Évaluation du modèle ---
print("\n=== Rapport de classification Isolation Forest (seuil optimisé) ===")
print(classification_report(y_true, book1['predicted_anomaly']))

cm = confusion_matrix(y_true, book1['predicted_anomaly'])
print("\nMatrice de confusion :\n", cm)

# --- 🔟 Visualisation ---
sns.boxplot(x=y_true, y=book1['anomaly_score'])
plt.axhline(y=-best_thresh, color='r', linestyle='--', label='Seuil optimisé')
plt.title("Isolation Forest - Score d'anomalie par classe réelle")
plt.xlabel("Classe réelle (0 = présente, 1 = manquante)")
plt.ylabel("Score d'anomalie")
plt.legend()
plt.show()

# --- 11️⃣ Sauvegarde des résultats ---
book1[['NumFacture', 'in_jde911', 'anomaly_score', 'predicted_anomaly']].to_csv('IsolationForest_results.csv', index=False)
print("\n💾 Fichier des résultats sauvegardé : IsolationForest_results.csv")
