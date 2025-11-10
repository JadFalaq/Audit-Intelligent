import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# --- Load data ---
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# --- Feature engineering ---
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)

# --- Encode features ---
features = book1.drop(columns=['NumLigne','TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
X = features[['CodeClient','CompteProduit','CentreAnalyse']].apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Fonction pour entraîner et évaluer un arbre avec dataset équilibré ---
def train_decision_tree_balanced(X, y, model_name="Book"):
    # Split train/test avec stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Combiner X_train et y_train pour le sur-échantillonnage
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    # Séparer classes
    df_majority = train_df[train_df['target'] == 1]  # Manquante
    df_minority = train_df[train_df['target'] == 0]  # Présente
    
    # Sur-échantillonnage de la classe minoritaire
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),  # équilibrage parfait
                                     random_state=42)
    
    # Recomposer le dataset équilibré
    train_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    X_train_bal = train_balanced[X_train.columns]
    y_train_bal = train_balanced['target']
    
    # Entraînement du Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_bal, y_train_bal)
    
    # Prédictions sur le jeu de test original
    y_pred = dt.predict(X_test)
    
    # Affichage des metrics
    print(f"\n=== {model_name} DecisionTree (Balanced Train) ===")
    print(classification_report(y_test, y_pred, target_names=['Présente','Manquante']))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Sauvegarde des résultats
    results = pd.DataFrame({
        'NumFacture': book1.loc[X_test.index, 'NumFacture'],
        f'{model_name}_Predicted': ['Manquante' if i==1 else 'Présente' for i in y_pred],
        f'{model_name}_Actual': ['Manquante' if i==1 else 'Présente' for i in y_test]
    })
    results.to_csv(f'transaction_predictions_{model_name}_balanced.csv', index=False)
    
    return dt

# --- Entraînement pour Book2 ---
y2 = features['in_book2']
dt_model2 = train_decision_tree_balanced(X, y2, model_name="Book2")

# --- Entraînement pour Book3 ---
y3 = features['in_book3']
dt_model3 = train_decision_tree_balanced(X, y3, model_name="Book3")
