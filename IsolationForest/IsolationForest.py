import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Charger les données ===
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# === 2. Créer les indicateurs (1 = manquante) ===
book1['in_jde911'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_jde311'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# === 3. Nettoyage des données ===
features = ['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']
X = book1[features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === 4. Isolation Forest Optimisé ===
# Paramètres optimisés : contamination='auto' + nombre d'estimateurs élevé
iso = IsolationForest(
    n_estimators=500,       # plus d’arbres pour une meilleure stabilité
    contamination='auto',   # ajustement automatique selon la distribution
    max_samples='auto',
    random_state=42,
    bootstrap=True,
    n_jobs=-1
)
iso.fit(X)

# === 5. Calcul du score d’anomalie ===
book1['anomaly_score'] = iso.decision_function(X)

# === 6. Optimisation dynamique du seuil de Tukey ===
# Calcul des quartiles
Q1 = book1['anomaly_score'].quantile(0.25)
Q3 = book1['anomaly_score'].quantile(0.75)
IQR = Q3 - Q1

# Optimisation du coefficient de Tukey selon la distribution
# On peut tester plusieurs k : 1.5 (classique), 2, 2.5…
best_k = None
best_sep = -np.inf

for k in [0.5,1,1.5,4]:
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    
    # Mesure de séparation entre classes (différence moyenne des moyennes)
    sep = abs(book1.loc[book1['in_jde911']==1, 'anomaly_score'].mean() - 
              book1.loc[book1['in_jde911']==0, 'anomaly_score'].mean())
    
    if sep > best_sep:
        best_sep = sep
        best_k = k

# Seuil final optimisé
tukey_threshold = Q1 - best_k * IQR

# === 7. Visualisation ===
sns.set(style="whitegrid", palette="muted", font_scale=1.1)
plt.figure(figsize=(8,6))
sns.boxplot(x=book1['in_jde911'], y=book1['anomaly_score'])
plt.axhline(y=tukey_threshold, color='gray', linestyle='--')
plt.title("Isolation Forest anomaly score by dichotomous class (Tukey optimized)")
plt.xlabel("Classe réelle (0 = présente, 1 = manquante)")
plt.ylabel("Score d'anomalie")
plt.show()

print(f"\n✅ Seuil de Tukey optimisé (k={best_k}): {tukey_threshold:.4f}")
