import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('final-stage/data/Book1.csv')

df["NumFacture"] = pd.to_numeric(df["NumFacture"], errors="coerce")

# Trier par la colonne NumFacture
df_sorted = df.sort_values(by="NumFacture")

# Sauvegarder le fichier trié
df_sorted.to_csv("fichier_trie.csv", index=False)

print("✅ Le fichier a été trié par NumFacture et enregistré sous le nom 'fichier_trie.csv'")