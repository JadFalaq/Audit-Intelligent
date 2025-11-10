import pandas as pd

#load data
book1=pd.read_csv('./data/Book1.csv', low_memory=False)
book2=pd.read_csv('./data/Book2.csv', low_memory=False)
book3=pd.read_csv('./data/Book3.csv', low_memory=False)
book4=pd.read_csv('./data/book1_with_labels.csv', low_memory=False)

#print first few rows
print(book1.head())
print(book2.head())
print(book3.head())

#Dataset columns with data types
i=0
for col in book1.columns:
    print(f"{i} Column: {col},Type: {book1[col].dtype}")
    i+=1

#Unique value counts for categorical columns
book5=book1.drop(columns=['MontantHT','MontantTTC','DateFacture', 'DateCreation', 'DateModification', 'DateEDI'], errors='ignore')
for col in book5.columns:
    print(f"Column: {col},Unique Values: {book1[col].nunique()}")

