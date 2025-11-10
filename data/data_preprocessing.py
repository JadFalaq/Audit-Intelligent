import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#load data
book1=pd.read_csv('./data/Book1.csv', low_memory=False)
book2=pd.read_csv('./data/Book2.csv', low_memory=False)
book3=pd.read_csv('./data/Book3.csv', low_memory=False)

book5=book1.drop_duplicates()

#feature engineering
book1['in_book2'] = book1['NumFacture'].isin(book2['GLDOC']).astype(int)
book1['in_book3'] = book1['NumFacture'].isin(book3['RPDOC']).astype(int)

#Data cleaning 
if 'TypeFacture' in book1.columns:
    book1['TypeFacture'] = book1['TypeFacture'].astype(str)

for date_col in ['DateCreation', 'DateModification', 'DateEDI', 'DateFacture']:
    if date_col in book1.columns:
        book1[date_col] = pd.to_datetime(book1[date_col], errors='coerce')

#feature engineering 
features = book1.drop(columns=['NumLigne','TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')

# Compute and display correlation matrix
correlation_matrix = features.corr()
# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix for book1 Columns')
plt.tight_layout()
plt.show()

# Journal Entry Encoding

new_features = features[['CodeClient','CompteProduit','CentreAnalyse','in_book2','in_book3']].copy()
new_features = new_features.apply(pd.to_numeric, errors='coerce').fillna(0)
print(new_features.head())

new_features.to_csv('./data/features.csv', index=False)