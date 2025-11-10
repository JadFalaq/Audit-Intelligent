import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- Load data ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering: 1 = NOT present ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Data cleaning ---
for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')
if 'TypeFacture' in book1.columns:
    book1['TypeFacture'] = book1['TypeFacture'].astype(str)

# --- Features ---
features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 
                               'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Function to train and evaluate Random Forest ---
def train_random_forest(X, y, model_name="Book", random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    print(f"\n=== {model_name} RandomForest ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save predictions
    results = pd.DataFrame({
        'NumFacture': book1.loc[X_test.index, 'NumFacture'],
        f'{model_name}_Predicted_RF': y_pred,
        f'{model_name}_Actual': y_test.values
    })
    results.to_csv(f'transaction_predictions_{model_name}_RF.csv', index=False)
    
    return rf

# --- Train for Book2 ---
y2 = features['in_book2']
rf_model2 = train_random_forest(X, y2, model_name="Book2")

# --- Train for Book3 ---
y3 = features['in_book3']
rf_model3 = train_random_forest(X, y3, model_name="Book3")
