import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Charger les données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering ---
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Data cleaning ---
for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')

# --- Sélection et encodage des features ---
features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 
                               'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)
y = features['in_book3']

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- LinearSVC avec calibration pour probabilités ---
svc = LinearSVC(max_iter=5000, random_state=42)
calibrated_svc = CalibratedClassifierCV(svc, method='sigmoid')  # méthode sigmoid pour calibrer
calibrated_svc.fit(X_train, y_train)

# --- Prédictions probabilistes ---
y_prob = calibrated_svc.predict_proba(X_test)[:,1]  # probabilité de la classe 1 "Manquante"

# --- Ajustement du seuil ---
threshold = 0.6  # vous pouvez tester plusieurs valeurs entre 0.5 et 0.7
y_pred = (y_prob >= threshold).astype(int)

# --- Évaluation du modèle ---
print(f"=== LinearSVM Calibré (threshold={threshold}) ===")
print(classification_report(y_test, y_pred, target_names=['Présente','Manquante']))
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Sauvegarde des prédictions ---
results = pd.DataFrame({
    'NumFacture': book1.loc[X_test.index, 'NumFacture'],
    'LinearSVM_Predicted': y_pred,
    'LinearSVM_Actual': y_test.values
})
results.to_csv('transaction_predictions_book3_LinearSVM.csv', index=False)
