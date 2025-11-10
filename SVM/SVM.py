import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Charger les données ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Nettoyage des dates et types ---
if 'TypeFacture' in book1.columns:
    book1['TypeFacture'] = book1['TypeFacture'].astype(str)

for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')

# --- Sélection des features numériques ---
features = book1.drop(columns=['TypeFacture','DateFacture','DateCreation','DateModification','DateEDI','ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Fonction pour entraîner SVM linéaire ---
def train_linear_svm(X, y, model_name="Book"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # LinearSVC avec max_iter augmenté pour convergence
    svm_model = LinearSVC(random_state=42, max_iter=10000,class_weight='balanced')
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    print(f"\n=== {model_name} Linear SVM ===")
    print(classification_report(y_test, y_pred, target_names=['Présente','Manquante']))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Sauvegarde des résultats
    results = pd.DataFrame({
        'NumFacture': book1.loc[X_test.index, 'NumFacture'],
        f'{model_name}_Predicted_SVM': ['Manquante' if i==1 else 'Présente' for i in y_pred],
        f'{model_name}_Actual': ['Manquante' if i==1 else 'Présente' for i in y_test]
    })
    results.to_csv(f'transaction_predictions_{model_name}_SVM.csv', index=False)
    
    return svm_model

# --- Entraînement pour Book2 ---
y2 = features['in_book2']
svm_model2 = train_linear_svm(X, y2, model_name="Book2")

# --- Entraînement pour Book3 ---
y3 = features['in_book3']
svm_model3 = train_linear_svm(X, y3, model_name="Book3")
