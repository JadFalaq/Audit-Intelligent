import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
book1 = pd.read_csv('afriware_train.csv', low_memory=False)
book2 = pd.read_csv('final-stage/data/Book2.csv', low_memory=False)
book3 = pd.read_csv('final-stage/data/Book3.csv', low_memory=False)

# Feature engineering: 1 = NOT present
book1['in_book2'] = 1 - book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = 1 - book1['NumFacture'].isin(book3['RPDOC']).astype(int)
print(book1['in_book2'].value_counts())
print(book1['in_book3'].value_counts())

# Data cleaning
if 'TypeFacture' in book1.columns:
    book1['TypeFacture'] = book1['TypeFacture'].astype(str)

for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')

# Feature engineering
features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')

# Encode features
new_features = features[['CodeClient','CompteProduit','CentreAnalyse','MontantHT','MontantTTC','Taxes']].copy()
new_features = new_features.apply(pd.to_numeric, errors='coerce').fillna(0)

# --- Decision Tree for Book2 ---
X2 = new_features.copy()
y2 = features['in_book2']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=50)
dt_model2 = DecisionTreeClassifier(random_state=60)
dt_model2.fit(X2_train, y2_train)
pred2_dt = dt_model2.predict(X2_test)
print('Book2 DecisionTree Accuracy:', accuracy_score(y2_test, pred2_dt))
print(classification_report(y2_test, pred2_dt))

# --- Decision Tree for Book3 ---
X3 = new_features.copy()
y3 = features['in_book3']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
dt_model3 = DecisionTreeClassifier(random_state=60)
dt_model3.fit(X3_train, y3_train)
pred3_dt = dt_model3.predict(X3_test)
print('Book3 DecisionTree Accuracy:', accuracy_score(y3_test, pred3_dt))
print(classification_report(y3_test, pred3_dt))

# Save predictions for Book2
results_book2 = pd.DataFrame({
    'NumFacture': book1.loc[X2_test.index, 'NumFacture'],
    'Book2_Predicted_DT': pred2_dt,
    'Book2_Actual': y2_test.values
})
results_book2.to_csv('transaction_match_predictions_book2_DT.csv', index=False)

# Save predictions for Book3
results_book3 = pd.DataFrame({
    'NumFacture': book1.loc[X3_test.index, 'NumFacture'],
    'Book3_Predicted_DT': pred3_dt,
    'Book3_Actual': y3_test.values
})
results_book3.to_csv('transaction_match_predictions_book3_DT.csv', index=False)

