import pandas as pd
import numpy as np

df_afriware = pd.read_csv('final-stage/data/Book1.csv',  low_memory=False)
columns_to_drop = ['NumLigne']
df_afriware.drop(columns=columns_to_drop, inplace=True, errors='ignore')


df_grouped = df_afriware.groupby(['NumFacture','TypeFacture','DateFacture'], as_index=False).agg({
    'MontantHT': 'sum',
    'MontantTTC': 'sum',
    'Taxes': 'sum',
    'DateModification': 'first',
    'DateCreation': 'first',
    'DateEDI': 'first',
    'ReferenceEDI': 'first',
    'CodeClient': 'first',
    'CentreAnalyse': 'first',
    'CompteProduit': 'first'


})

df_grouped = df_grouped[['TypeFacture', 'NumFacture', 'MontantHT', 'MontantTTC', 'Taxes', 'DateFacture','DateCreation', 'DateModification', 'DateEDI', 'ReferenceEDI','CodeClient','CentreAnalyse','CompteProduit']]
df_grouped.sort_values('NumFacture', ascending=True, inplace=True)
df_grouped['MontantHT']=df_grouped['MontantHT'].round(2)
df_grouped['MontantTTC']=df_grouped['MontantTTC'].round(2)
df_grouped['Taxes']=df_grouped['Taxes'].round(2)

df_grouped.to_csv('afriware_train.csv', index=False)