import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Chargement des données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# Sélection des features numériques
features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)
y = features['in_book3']  # exemple pour Book3, tu peux changer pour Book2

# --- Standardisation ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --- KNN optimisé ---
knn = KNeighborsClassifier(
    n_neighbors=1,        # tester plusieurs valeurs de n_neighbors
    weights='distance',   # pondération par distance
    metric='minkowski',   # distance euclidienne
    p=2
)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# --- Évaluation ---
print("KNN Optimisé Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Sauvegarde des prédictions ---
results = pd.DataFrame({
    'NumFacture': book1.loc[y_test.index, 'NumFacture'],
    'KNN_Predicted': y_pred,
    'Actual': y_test.values
})
results.to_csv('transaction_match_predictions_book3_KNN.csv', index=False)
