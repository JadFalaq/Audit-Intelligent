import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# --- Chargement des données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Création des labels ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Sélection des features ---
features = book1.drop(columns=['TypeFacture','DateFacture','DateCreation','DateModification','DateEDI','ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)
y = features['in_book3']  # Exemple pour Book3

# --- Standardisation ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Rééchantillonnage pour équilibrer les classes (train uniquement) ---
df = pd.DataFrame(X_scaled, columns=X.columns)
df['target'] = y.values

# Séparer les classes
df_majority = df[df.target==0]
df_minority = df[df.target==1]

# Échantillonnage aléatoire de la classe majoritaire pour équilibrer
df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)

# Fusionner pour créer le dataset équilibré
df_balanced = pd.concat([df_majority_downsampled, df_minority])
X_balanced = df_balanced.drop(columns='target').values
y_balanced = df_balanced['target'].values

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# --- Modèle Naïve Bayes ---
nb_model = GaussianNB(var_smoothing=1e-9)
nb_model.fit(X_train, y_train)

# --- Prédictions ---
y_pred = nb_model.predict(X_test)

# --- Évaluation ---
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
