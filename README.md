#!/usr/bin/env bash
set -euo pipefail

APP_FILE="${1:-app.py}"
PY_BIN="${PY_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Vérification de Python..."
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "Erreur: Python introuvable. Installez Python 3 et réessayez."
  exit 1
fi

echo "==> Création de l'environnement virtuel ($VENV_DIR)..."
"$PY_BIN" -m venv "$VENV_DIR"

# Activer l'environnement selon OS / shell
ACTIVATE=""
if [ -f "$VENV_DIR/bin/activate" ]; then
  ACTIVATE="$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  ACTIVATE="$VENV_DIR/Scripts/activate"
fi

# shellcheck disable=SC1090
source "$ACTIVATE"

echo "==> Mise à jour de pip..."
python -m pip install --upgrade pip

echo "==> Installation des dépendances..."
pip install \
  streamlit \
  pandas \
  numpy \
  scikit-learn \
  unidecode \
  openai \
  openpyxl \
  altair \
  wordcloud \
  markdown

echo "==> Vérification du fichier de l'application: $APP_FILE"
if [ ! -f "$APP_FILE" ]; then
  echo "Attention: $APP_FILE n'existe pas dans le dossier courant."
  echo "Création d'un squelette minimal ($APP_FILE) pour démarrer."
  cat > "$APP_FILE" <<'PY'
import streamlit as st

st.set_page_config(page_title="Analyse d'enquêtes", layout="wide")

st.title("Streamlit — Analyse d'enquêtes")
st.markdown("""
Cette application attend vos données et votre logique d'analyse.
Remplacez ce squelette par votre implémentation.
""")

st.header("Fonctionnalités (à implémenter)")
st.markdown("""
- Chatbot : analyses descriptives déterministes et RAG/LLM pour questions qualitatives  
- Comptage exact sur colonnes  
- Suggestion de colonnes à ignorer  
- Analyse des verbatims (tri, filtres, sentiments, pagination, graphiques persistants)  
- Mini-synthèse (wordcloud, camembert de sentiment)
""")
PY
fi

echo "==> Lancement de Streamlit..."
exec streamlit run "$APP_FILE"
