import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# --- Charger les données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering : 1 = NOT present ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Features ---
features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
X_df = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Standardisation ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns, index=X_df.index)

# --- Fonction pour Logistic Regression avec seuil ajustable ---
def train_logreg_adjust_threshold(X, y, model_name="Book", threshold=0.5):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_index = X_test.index

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)

    # Probabilités
    proba = lr_model.predict_proba(X_test)[:,1]  # probabilité de la classe minoritaire
    y_pred = (proba >= threshold).astype(int)     # ajustement du seuil

    # Metrics
    print(f"\n=== {model_name} Logistic Regression (threshold={threshold}) ===")
    print(classification_report(y_test, y_pred, target_names=['Présente','Manquante']))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Sauvegarde des résultats
    results = pd.DataFrame({
        'NumFacture': book1.loc[test_index, 'NumFacture'],
        f'{model_name}_Predicted_LR': ['Manquante' if i==1 else 'Présente' for i in y_pred],
        f'{model_name}_Actual': ['Manquante' if i==1 else 'Présente' for i in y_test]
    })
    results.to_csv(f'transaction_predictions_{model_name}_LR_threshold.csv', index=False)

    return lr_model

# --- Entraînement pour Book3 avec seuil ajusté ---
y3 = book1['in_book3']
# On peut tester un seuil plus bas comme 0.3 pour améliorer le recall
lr_model3 = train_logreg_adjust_threshold(X_scaled, y3, model_name="Book3", threshold=0.56
)

# Sauvegarde du modèle
joblib.dump(lr_model3, "lr_model3_adjusted.pkl")
