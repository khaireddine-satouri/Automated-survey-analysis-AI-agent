# Streamlit — Analyse d'enquêtes
Synthèse + Chatbot + Storytelling + Analyse verbatims

## Nouveautés

### Chatbot
- Analyses descriptives déterministes :
  - Comptages
  - Répartition Top-K
  - Valeurs uniques
  - Recherche "contiennent ..."
  - Pourcentage d’une valeur
  - Statistiques numériques
- RAG / LLM pour l’analyse des questions qualitatives

### Fonctionnalités principales
- Comptage exact : "Combien ont répondu … <valeur>" sur toute la colonne
- Suggestion automatique de colonnes à ignorer (IDs, noms, e-mails...) avec option d’application / réinitialisation
- Analyse des verbatims :
  - Tri décroissant des labels
  - Filtres par labels / sentiments
  - Emojis d’humeur
  - Sentiments en camembert
  - Pagination (20 réponses / page)
  - Graphiques persistants
- Mini-synthèse :
  - Wordcloud sur colonnes texte
  - Camembert pour le sentiment (masqué si aucun résultat)

## Prérequis

Installer les dépendances nécessaires :

```bash
pip install streamlit pandas numpy scikit-learn unidecode openai openpyxl altair wordcloud markdown
