
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns



book1=pd.read_csv('./data/book1_with_labels.csv', low_memory=False)
# Group by NumFacture and count occurrences
numfacture_counts = book1.groupby('NumFacture').size().reset_index(name='count')
print(numfacture_counts.head())


features = book1.drop(columns=['TypeFacture', 'DateFacture', 'DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI'], errors='ignore')
# Only keep records where in_book2 == 0

features_not_in_book2 = features[book1['in_book2'] == 0].copy()
# Group by NumFacture and count occurrences
grouped_not_in_book2 = features_not_in_book2.groupby('NumFacture').size().reset_index(name='count')
print(grouped_not_in_book2.head())

print(grouped_not_in_book2.shape)

""""
sum1=book1['in_book2'].sum()
sum2=book1['in_book3'].sum()
print(f"Total transactions in Book2: {sum1}")
print(f"Total transactions in Book3: {sum2}")"""
""""
# Sum the count column
total_count = grouped_not_in_book2['count'].sum()
print(f"Total count of records not in book2: {total_count}")
"""

book5=book1[(book1['CodeClient']==12008) & (book1['CompteProduit']==449910.92)]
book5.to_csv('filtered_book5.csv', index=False)