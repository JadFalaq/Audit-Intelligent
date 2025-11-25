# 🧠 Audit Intelligent  
### _Détection d’anomalies dans les journaux comptables via Machine Learning_

---

## 🏢 Contexte du projet

Ce projet a été réalisé dans le cadre du **stage de fin d’année** au sein de la **Direction des Systèmes d’Information (DSI) d’AKWA Group**, sous l’encadrement de :

- 🎓 **Encadrant académique** : M. **Abdellatif EL AFIA** (ENSIAS)  
- 🧑‍💼 **Maître de stage** : M. **Kamal GHANAM** (AKWA Group)  
- 📅 **Période** : du 15 juin au 5 septembre 2025  
- 🏫 **Établissement** : École Nationale Supérieure d’Informatique et d’Analyse des Systèmes (ENSIAS), filière **Ingénierie en Intelligence Artificielle (2IA)**

---

## 🚀 Description

**Audit Intelligent** est un système de détection d’anomalies conçu pour **identifier automatiquement les transactions comptables incohérentes ou manquantes** entre différents systèmes de gestion — notamment entre les fichiers **Afriware** et **JDE**.  

Le projet s’appuie sur des **modèles de Machine Learning supervisés** pour assister les auditeurs internes dans leurs missions de contrôle, en réduisant les risques d’erreurs humaines et les coûts d’audit.

### 🎯 Objectifs principaux

- Développer un **modèle d’apprentissage automatique** pour la détection d’anomalies dans les journaux comptables.  
- Identifier les **transactions présentes dans Afriware mais absentes dans JDE**.  
- Fournir un **outil d’aide à la décision** pour prioriser les zones à risque.  
- Contribuer à la **digitalisation des processus d’audit interne** d’AKWA Group.

---

## 🧠 Approche méthodologique

Le projet suit la méthodologie **CRISP-DM**, standard en data science :

1. **Business Understanding** – Définir les enjeux et critères d’audit.  
2. **Data Understanding** – Collecter et explorer les données (Afriware, JDE_F0911, JDE_F03B11).  
3. **Data Preparation** – Nettoyer, harmoniser et sélectionner les variables pertinentes.  
4. **Modeling** – Expérimenter plusieurs modèles supervisés.  
5. **Evaluation** – Comparer les performances et interpréter les résultats.  
6. **Deployment** – Intégrer les modèles à une application web.

---

## 📊 Jeux de données

Trois principales sources comptables ont été exploitées :

| Fichier | Description | Exemple de colonnes |
|----------|--------------|---------------------|
| **Afriware** | Factures et écritures comptables principales | TypeFacture, NumeroFacture, CodeClient, MontantHT, CentreAnalyse |
| **JDE_F0911** | Grand Livre Général | GLDOC, GLMCU, GLOBJ, GLFY, GLPN |
| **JDE_F03B11** | Comptes Clients | RPDOC, RPAG, RPAAP, RPFY, RPPN |

Les anomalies visées correspondent aux **transactions présentes dans Afriware mais absentes dans les fichiers JDE**.

---

## 🧮 Modèles de Machine Learning testés

Les modèles supervisés suivants ont été évalués :

| Modèle | Type | Performance (Rappel macro) |
|--------|------|-----------------------------|
| **Logistic Regression** | Linéaire | 0.73 |
| **SVM (Support Vector Machines)** | Linéaire | 0.63 |
| **Decision Tree** | Arborescent | 0.90 |
| **Random Forest** | Ensemble Learning | 🔹 **Meilleur modèle (F1-score global = 0.97)** |
| **K-Nearest Neighbors (kNN)** | Distance-based | 0.85 |

Les **modèles non supervisés** (Isolation Forest, Autoencoder) ont également été explorés mais se sont révélés moins adaptés au volume et à la nature des données.

---

## 📈 Résultats clés

- 🔍 **49 846 anomalies** détectées parmi plus de **480 000 transactions**.  
- 🌲 Le modèle **Random Forest** obtient les meilleures performances globales.  
- 🧮 Le **taux de rappel de 0.82** pour les anomalies manquantes avec Decision Tree.  
- 🕒 Réduction significative du temps d’audit et amélioration de la fiabilité des contrôles.

---

## ⚙️ Technologies utilisées

| Domaine | Technologies |
|----------|---------------|
| **Langage principal** | Python |
| **Bibliothèques ML** | Scikit-learn, Pandas, NumPy |
| **Analyse & Prétraitement** | Jupyter, Matplotlib, Seaborn |
| **Base de données** | PostgreSQL |
| **Déploiement web** | VibeCoding |
| **Versionnement** | Git, GitHub |

---

## 💡 Perspectives futures

- Développer un **système de détection en temps réel** pour les nouvelles transactions.  
- Explorer des **réseaux neuronaux** (LSTM, Autoencoders profonds).  
- Étendre la détection à d’autres anomalies (TVA, fraude fiscale).  
- Intégrer un **dashboard interactif** de visualisation.

---

## 👨‍💻 Auteur

**Jad Falaq**  
Étudiant ingénieur en **Intelligence Artificielle** à l’ENSIAS  
📧 [jadfalaq@gmail.com](mailto:jadfalaq@gmail.com)  
🔗 [GitHub - JadFalaq](https://github.com/JadFalaq)

---

## 🪪 Licence

Ce projet est distribué sous licence **MIT**.  
Vous pouvez librement le réutiliser, le modifier et le redistribuer, en mentionnant la source.

---

## 🏆 Remerciements

> Je remercie chaleureusement **AKWA Group**, **M. Kamal Ghanam**,  
> ainsi que **M. Abdellatif El Afia** pour leur accompagnement, leur encadrement  
> et leur confiance tout au long de ce projet.

