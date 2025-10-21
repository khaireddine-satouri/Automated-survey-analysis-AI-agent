#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit — Analyse d'enquêtes (synthèse + chatbot + storytelling + analyse verbatims)
--------------------------------------------------------------------------------------

Nouveautés :
- Chatbot : analyses descriptives DÉTERMINISTES (comptages, répartition Top-K, uniques,
  "contiennent ...", pourcentage d'une valeur, stats numériques) + RAG/LLM pour les questions qualitatives.
- Comptage exact "Combien ont répondu ... <valeur>" sur TOUTE la colonne.
- Suggestion de colonnes à ignorer (IDs, noms, e-mails...) + application/réinitialisation.
- Analyse verbatims : tri décroissant des labels, filtres par labels/sentiments, smiles 😀😐😞,
  sentiments en camembert, 20 réponses/page, graphiques persistants.
- Mini-synthèse : wordcloud sur colonnes texte, camembert pour le sentiment (masqué si 0).

Prérequis :
    pip install streamlit pandas numpy scikit-learn unidecode openai openpyxl altair wordcloud markdown
"""

from __future__ import annotations
import os
import re
import io
import json
import math
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import streamlit as st
import altair as alt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# WordCloud (optionnel)
try:
    from wordcloud import WordCloud, STOPWORDS as WC_STOPWORDS
    _HAS_WC = True
except Exception:
    _HAS_WC = False
    WC_STOPWORDS = set()

# OpenAI (SDK >= 1.0)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Markdown -> HTML
try:
    import markdown as md
    _HAS_MD_CONV = True
except Exception:
    _HAS_MD_CONV = False

def _wrap_html(body_html: str, title: str = "Storytelling") -> str:
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;
         line-height:1.6; max-width:900px; margin:2rem auto; padding:0 1rem; color:#111827}}
    h1,h2,h3{{margin-top:1.6rem}}
    blockquote{{border-left:4px solid #e5e7eb; padding:.5rem 1rem; color:#374151; background:#f9fafb}}
    code{{background:#f3f4f6; padding:.2rem .4rem; border-radius:4px}}
    pre code{{display:block; padding:1rem}}
    ul,ol{{padding-left:1.25rem}}
    hr{{border:none;border-top:1px solid #e5e7eb;margin:1.5rem 0}}
  </style>
</head>
<body>
{body_html}
</body>
</html>"""


# =======================================
# Configuration (seuils et constantes)
# =======================================
CAT_UNIQUE_RATIO_MAX = 0.20
CAT_UNIQUE_ABS_MAX = 30
DISCRETE_UNIQUE_MAX = 25
CONTINUOUS_UNIQUE_RATIO_MIN = 0.30
MIN_TEXT_LEN_FOR_SENTIMENT = 12
TOP_N_CATEGORIES = 25

LIKERT_SETS = [
    ["très insatisfait","insatisfait","plutôt insatisfait","neutre","plutôt satisfait","satisfait","très satisfait"],
    ["pas du tout d'accord","plutôt pas d'accord","ni d'accord ni pas d'accord","plutôt d'accord","tout à fait d'accord"],
    ["très mauvais","mauvais","moyen","bon","très bon"],
]

BINARY_SYNONYMS = {
    "oui": {"oui","o","y","yes","true","vrai","1"},
    "non": {"non","n","no","false","faux","0"},
}

# Smiles (pas de ballons)
SMILEYS = {"positif": "😀", "neutre": "😐", "négatif": "😞"}

FR_STOPWORDS = {
    "le","la","les","de","des","du","un","une","et","à","a","au","aux","en","dans","pour","par","sur",
    "avec","sans","ce","cet","cette","ces","qui","que","quoi","dont","où","ou","ne","pas","plus","moins",
    "très","tres","se","sa","son","ses","leur","leurs","on","nous","vous","ils","elles","est","sont","été",
    "etre","être","etait","était","comme","car","donc","or","ni","mais","si","l","d","c","j","y","aujourd","hui","ainsi","alors",""
}

# =====================
# Utilitaires généraux
# =====================
def ensure_openai() -> Optional[OpenAI]:
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        pass
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key or not isinstance(api_key, str) or not api_key.startswith("sk-"):
        return None
    if OpenAI is None:
        return None
    return OpenAI(api_key=api_key)

def strip_pii(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\+?\d[\d\s().-]{6,}", "[PHONE]", text)
    text = re.sub(r"@\w+", "@[HANDLE]", text)
    return text

def _norm(s: Any) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def _try_parse_number(s: Any) -> Optional[float]:
    if pd.isna(s):
        return None
    try:
        return float(str(s).replace(",", ".").strip())
    except Exception:
        return None
def _prep_export_frame(df_src: pd.DataFrame, full_index: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({
        "row_id": full_index.iloc[:len(df_src)].values,  # index d'origine dans la feuille Excel
        "reponse": df_src["text"].astype(str),
        "labels": df_src["labels"].map(lambda x: "; ".join(map(str, x)) if isinstance(x, list) else str(x))
    })
    if "sentiment" in df_src.columns:
        out["sentiment"] = df_src["sentiment"].astype(str)
    return out

def _to_bytes_auto(df_export: pd.DataFrame) -> tuple[str, bytes, str]:
    """
    Essaie Excel (openpyxl). Si échec → CSV UTF-8 SIG.
    Retourne (ext, bytes, mime).
    """
    try:
        import openpyxl  # noqa
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="verbatims")
        return ("xlsx", bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        data = df_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        return ("csv", data, "text/csv")

# ==========================
# Détection du type de champ
# ==========================
@dataclass
class ColMeta:
    name: str
    n: int
    n_nonnull: int
    n_unique: int
    unique_ratio: float
    is_numeric: bool
    is_integer_like: bool
    supertype: str
    subtype: str
    categories: Optional[List[str]] = None
    order: Optional[List[str]] = None
    is_sentiment_candidate: bool = False

def detect_column_meta(series: pd.Series, name: str) -> ColMeta:
    s = series.dropna()
    n = len(series)
    n_nonnull = len(s)
    n_unique = s.nunique(dropna=True)
    unique_ratio = (n_unique / n_nonnull) if n_nonnull else 0.0

    # Numérique ?
    is_numeric_dtype = pd.api.types.is_numeric_dtype(series)
    parsed_nums = None
    if not is_numeric_dtype:
        parsed_nums = [_try_parse_number(v) for v in s]
        if all(v is not None for v in parsed_nums) and n_nonnull > 0:
            is_numeric_dtype = True

    if is_numeric_dtype:
        if parsed_nums is None:
            parsed_nums = [float(v) if not pd.isna(v) else None for v in s]
        is_integer_like = all(v is None or float(v).is_integer() for v in parsed_nums)
        if (n_unique <= DISCRETE_UNIQUE_MAX) or (unique_ratio < CONTINUOUS_UNIQUE_RATIO_MIN and n_unique <= 50):
            subtype = "discrète"
        else:
            subtype = "continue"
        return ColMeta(name, n, n_nonnull, n_unique, unique_ratio, True, is_integer_like, "quantitative", subtype)

    # Texte vs catégoriel : basé sur ratio d'unicité
    texts = s.astype(str).str.strip()
    norm_texts = texts.map(_norm)

    # Binaire ?
    uniq_norm = list(pd.unique(norm_texts))
    binary_map = set()
    for u in uniq_norm:
        if u in BINARY_SYNONYMS["oui"]:
            binary_map.add("oui")
        elif u in BINARY_SYNONYMS["non"]:
            binary_map.add("non")
        else:
            binary_map.add(u)
    if len(binary_map) == 2 and binary_map.issubset({"oui", "non"}):
        return ColMeta(name, n, n_nonnull, n_unique, unique_ratio, False, False, "qualitative", "binaire", ["Non","Oui"], None, False)

    # Ordinal ?
    def _maybe_ordinal(values: List[str]) -> Optional[List[str]]:
        vals = list({v for v in values if v != ""})
        for template in LIKERT_SETS:
            if set(vals).issubset(set(template)) and len(vals) >= max(3, len(vals)):
                return [v for v in template if v in vals]
        parsed = []
        for v in vals:
            num = _try_parse_number(v)
            if num is None or not float(num).is_integer():
                break
            parsed.append(int(num))
        else:
            if 3 <= len(parsed) <= 11:
                ordered = sorted(set(parsed))
                return [str(i) for i in ordered]
        return None

    ord_order = _maybe_ordinal(norm_texts.tolist())
    if ord_order is not None:
        order_original = []
        for key in ord_order:
            for orig in pd.unique(texts):
                if _norm(orig) == key:
                    order_original.append(orig)
                    break
        return ColMeta(name, n, n_nonnull, n_unique, unique_ratio, False, False, "qualitative", "ordinale",
                       order_original, order_original, False)

    is_categorical = (unique_ratio <= CAT_UNIQUE_RATIO_MAX) or (n_unique <= CAT_UNIQUE_ABS_MAX)
    if is_categorical:
        categories = texts.dropna().value_counts().head(TOP_N_CATEGORIES).index.astype(str).tolist()
        return ColMeta(name, n, n_nonnull, n_unique, unique_ratio, False, False, "qualitative", "catégorielle",
                       categories, None, False)

    is_sentiment_candidate = (texts.str.len().mean() if n_nonnull else 0) >= MIN_TEXT_LEN_FOR_SENTIMENT
    return ColMeta(name, n, n_nonnull, n_unique, unique_ratio, False, False, "qualitative", "texte", None, None, is_sentiment_candidate)

# ================
# Ignore columns
# ================
def suggest_ignored_columns(df: pd.DataFrame) -> List[str]:
    """Heuristiques : nom de colonne (id, nom, prénom, email, phone, etc.) + ratio d'unicité très élevé."""
    suggestions: List[str] = []
    name_tokens_block = {
        "id","identifiant","identifiants","uuid","guid","matricule","code","cle","clé","key",
        "numero","numéro","no","n","ref","reference","référence","user","userid","username","compte",
        "email","mail","courriel","e-mail","telephone","tel","tél","mobile","gsm","phone",
        "nom","name","lastname","last","prenom","prénom","firstname","first"
    }
    for c in df.columns:
        cn = _norm(c)
        tokens = set(re.split(r"[^a-z0-9]+", cn))
        if tokens & name_tokens_block:
            suggestions.append(c)
            continue
        s = df[c].astype(str).str.strip()
        non_empty = s.replace("", np.nan).dropna()
        if len(non_empty) > 20:
            ur = non_empty.nunique() / len(non_empty)
            if ur >= 0.95:  # quasi toutes uniques -> identifiant probable
                suggestions.append(c)
    # garder l’ordre d’origine, sans doublons
    seen, out = set(), []
    for c in df.columns:
        if c in suggestions and c not in seen:
            out.append(c); seen.add(c)
    return out

# =========================
# Embeddings + RAG helpers
# =========================
@st.cache_resource(show_spinner=False)
def get_nn_index(embs: np.ndarray) -> NearestNeighbors:
    k = min(50, len(embs)) if len(embs) > 0 else 1
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="auto")
    if len(embs) > 0:
        nn.fit(embs)
    return nn

def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    embs: List[List[float]] = []
    B = 256
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=batch)
        embs.extend([e.embedding for e in resp.data])
    X = np.array(embs, dtype=np.float32)
    if len(X) == 0:
        return X
    X = normalize(X)
    return X

def retrieve(query: str, client: OpenAI, df: pd.DataFrame, X: np.ndarray, nn: NearestNeighbors, top_k: int) -> pd.DataFrame:
    if len(df) == 0:
        cols = list(df.columns)
        return pd.DataFrame(columns=cols + ["similarity", "rank"])  # type: ignore
    q_emb = embed_texts(client, [query])
    distances, indices = nn.kneighbors(q_emb, n_neighbors=min(top_k, len(df)))
    sims = 1.0 - distances[0]
    rows = []
    for rank, (idx, sim) in enumerate(zip(indices[0], sims), start=1):
        row = df.iloc[int(idx)].to_dict()
        row["similarity"] = float(sim)
        row["rank"] = rank
        rows.append(row)
    return pd.DataFrame(rows)

def build_context(question_label: str, retrieved_df: pd.DataFrame, max_chars: int = 12000) -> str:
    lines = [f"QUESTION: {question_label}", "EXTRAITS (classés par pertinence décroissante):"]
    acc = 0
    for _, r in retrieved_df.iterrows():
        txt = strip_pii(str(r["text"]).replace("\n", " ").strip())
        chunk = f"- (sim={r['similarity']:.3f}) {txt}"
        if acc + len(chunk) > max_chars:
            break
        lines.append(chunk)
        acc += len(chunk)
    return "\n".join(lines)

# =========================
# Appels LLM (SDK stable)
# =========================
def _supports_temperature(model: str) -> bool:
    # Ajuste la liste si d'autres modèles limitent la température
    return not model.startswith("gpt-5")  # gpt-5 / gpt-5-mini : pas de temperature param

def _chat_complete(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
    kwargs = {"model": model, "messages": messages}
    if temperature is not None and _supports_temperature(model):
        kwargs["temperature"] = float(temperature)

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Fallback 1 : si échec et on avait mis une température, on réessaie sans
        if isinstance(temperature, (int, float)):
            try:
                resp = client.chat.completions.create(model=model, messages=messages)
            except Exception:
                raise
        else:
            raise

    return resp.choices[0].message.content or ""


def _try_parse_json(text: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        return json.loads(text), None
    except Exception:
        pass
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.S)
    if m:
        block = m.group(1)
        try:
            return json.loads(block), None
        except Exception as e:
            return None, f"Echec parse bloc json: {e}"
    m2 = re.search(r"(\[.*\]|\{.*\})", text, flags=re.S)
    if m2:
        try:
            return json.loads(m2.group(1)), None
        except Exception as e:
            return None, f"Echec parse objet heuristique: {e}"
    return None, "Aucun JSON détecté"

def ask_llm_structured(client: OpenAI, model: str, context: str, user_query: str) -> Dict[str, Any]:
    system_prompt = (
        "Tu es analyste d'enquêtes. Réponds UNIQUEMENT à partir des extraits fournis.\n"
        "Retourne un JSON avec les clés EXACTES :\n"
        "  themes (liste d'objets {label, share}),\n"
        "  sentiment (objet {positif, neutre, négatif}),\n"
        "  insights (liste de chaînes),\n"
        "  quotes (liste de chaînes),\n"
        "  actions (liste de chaînes).\n"
        "Si une info manque, mets [] ou {}.\n"
        "IMPORTANT : réponds UNIQUEMENT avec un JSON valide, sans texte autour."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion d'analyse: {user_query}"},
    ]
    txt = _chat_complete(client, model, messages, temperature=0.0)
    data, err = _try_parse_json(txt)
    if data is not None:
        return data
    return {"raw": txt, "parse_error": err}

def ask_llm_chat(client: OpenAI, model: str, context: str, user_query: str) -> str:
    system = (
        "Tu es un analyste d'étude. Réponds en français, de manière concise et actionnable.\n"
        "Appuie-toi UNIQUEMENT sur les extraits fournis, cite-les si utile."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {user_query}"},
    ]
    return _chat_complete(client, model, messages, temperature=0.0)

# ===============================
# Helpers ANALYSES DÉTERMINISTES
# ===============================
def is_count_query(q: str) -> bool:
    if not q:
        return False
    qn = _norm(q)
    patterns = [
        r"\bcombien\b.*\b(r[eé]ponses?|valeurs?|entr[ée]es?|lignes?)\b",
        r"\bnombre\b.*\b(r[eé]ponses?|valeurs?|entr[ée]es?|lignes?)\b",
        r"\b(nb|nbr)\b.*\b(r[eé]ponses?|valeurs?|entr[ée]es?|lignes?)\b",
        r"\bhow\s+many\b.*\b(answers?|responses?|entries?|rows?|values?)\b",
        r"\bcount\b.*\b(answers?|responses?|entries?|rows?|values?)\b",
        r"\btotal\b.*\b(answers?|responses?)\b",
    ]
    return any(re.search(p, qn) for p in patterns)

def column_counts(series: pd.Series) -> Dict[str, int]:
    total_rows = len(series)
    non_null = int(series.notna().sum())
    non_empty = int(series.astype(str).str.strip().ne("").sum())
    empty_or_blank = total_rows - non_empty
    unique_non_empty = int(series.astype(str).str.strip().replace("", np.nan).dropna().nunique())
    return {
        "total_rows": total_rows,
        "non_null": non_null,
        "non_empty": non_empty,
        "empty_or_blank": empty_or_blank,
        "unique_non_empty": unique_non_empty,
    }

def _unique_clean_values(series: pd.Series) -> List[str]:
    s = series.astype(str).str.strip()
    s = s.replace("", np.nan).dropna()
    seen, out = set(), []
    for v in s:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def _best_value_from_query(query: str, series: pd.Series) -> Optional[str]:
    qn = _norm(query)
    # Entre guillemets ?
    quoted = re.findall(r"[\"'«]([^\"'»]+)[\"'»]", query)
    quoted = [q.strip() for q in quoted if q.strip()]
    uniques = _unique_clean_values(series)
    if quoted:
        target = quoted[-1]
        tn = _norm(target)
        best = None; best_len = -1
        for v in uniques:
            vn = _norm(v)
            if tn == vn or tn in vn or vn in tn:
                L = max(len(tn), len(vn))
                if L > best_len:
                    best, best_len = v, L
        if best:
            return best
    # Après préposition
    m = re.search(r"\b(par|pour|avec|en|de)\s+([A-Za-zÀ-ÿ0-9 ._'\-\/]+)", query, flags=re.I)
    if m:
        after = m.group(2).strip().strip("?.!,;:)")
        tn = _norm(after)[:80]
        best = None; best_score = 0.0
        q_tokens = set(re.split(r"\W+", tn)) - FR_STOPWORDS
        for v in uniques:
            vn = _norm(v)
            if vn in tn or tn in vn:
                return v
            v_tokens = set(re.split(r"\W+", vn)) - FR_STOPWORDS
            inter = len(q_tokens & v_tokens)
            denom = max(len(q_tokens), len(v_tokens), 1)
            score = inter / denom
            if score > best_score:
                best, best_score = v, score
        if best_score >= 0.6:
            return best
    # Inclusion brute
    best = None; best_len = -1
    for v in uniques:
        vn = _norm(v)
        if vn and (vn in qn or qn in vn):
            if len(vn) > best_len:
                best, best_len = v, len(vn)
    return best

def count_value_in_column(series: pd.Series, meta: ColMeta, target_value: str) -> Dict[str, Any]:
    s = series.astype(str)
    s_clean = s.str.strip()
    non_empty_mask = s_clean.ne("")
    non_empty = int(non_empty_mask.sum())

    if meta.subtype in {"catégorielle", "binaire", "ordinale"}:
        target_n = _norm(target_value)
        eq_mask = s_clean[non_empty_mask].map(lambda x: _norm(x) == target_n)
    else:
        target_n = _norm(target_value)
        eq_mask = s_clean[non_empty_mask].map(lambda x: target_n in _norm(x))

    count = int(eq_mask.sum())
    pct = (100.0 * count / non_empty) if non_empty else 0.0
    unique_non_empty = int(s_clean[non_empty_mask].nunique())
    return {
        "target": target_value,
        "count": count,
        "non_empty": non_empty,
        "pct": pct,
        "unique_non_empty": unique_non_empty,
    }

def frequency_table(series: pd.Series, meta: ColMeta, top_k: int = 20) -> pd.DataFrame:
    s = series.astype(str).str.strip()
    s = s.replace("", np.nan).dropna()
    vc = s.value_counts().head(top_k).reset_index()
    vc.columns = ["valeur", "compte"]
    total = int(s.shape[0]) if s.shape[0] else 1
    vc["%"] = (vc["compte"] / total * 100.0).round(1)
    return vc

def unique_query(series: pd.Series, list_values: bool = False, top_k: int = 100) -> Tuple[int, Optional[List[str]]]:
    s = series.astype(str).str.strip()
    s = s.replace("", np.nan).dropna()
    nunique = int(s.nunique())
    if list_values:
        vals = list(pd.unique(s))[:top_k]
        return nunique, vals
    return nunique, None

def numeric_summary(series: pd.Series) -> Dict[str, float]:
    num = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(num.count()),
        "mean": float(num.mean()),
        "std": float(num.std()),
        "min": float(num.min()),
        "q25": float(num.quantile(0.25)),
        "median": float(num.median()),
        "q75": float(num.quantile(0.75)),
        "max": float(num.max()),
    }

def contains_phrase_count(series: pd.Series, phrase: str) -> Dict[str, Any]:
    s = series.astype(str).str.strip()
    non_empty = int(s.replace("", np.nan).dropna().shape[0])
    phrase_n = _norm(phrase)
    count = int(s.map(lambda x: phrase_n in _norm(x)).sum())
    pct = (100.0 * count / non_empty) if non_empty else 0.0
    return {"phrase": phrase, "count": count, "non_empty": non_empty, "pct": pct}

# --- Parsers de requêtes utilisateur (FR/EN minimalistes)
def _parse_topk(q: str, default: int = 20) -> int:
    m = re.search(r"\btop\s+(\d+)\b", q, flags=re.I)
    if m:
        try:
            v = int(m.group(1))
            return max(1, min(200, v))
        except:
            pass
    return default

def _is_freq_query(q: str) -> bool:
    qn = _norm(q)
    keys = ["répartition","repartition","fréquence","frequence","top","classement","distribution","part","parts","pourcentage par","pourcent par"]
    return any(k in qn for k in keys)

def _is_unique_count_query(q: str) -> bool:
    qn = _norm(q)
    return ("valeurs uniques" in qn) or ("unique values" in qn) or ("combien de valeurs uniques" in qn)

def _is_unique_list_query(q: str) -> bool:
    qn = _norm(q)
    return ("liste des valeurs uniques" in qn) or ("list unique" in qn) or ("quelles sont les valeurs uniques" in qn)

def _parse_contains(q: str) -> Optional[str]:
    # cherche "contient 'xxx'" ou 'contiennent "xxx"' etc.
    m = re.search(r"contien\w*\s+[\"'«]([^\"'»]+)[\"'»]", q, flags=re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"(contains|include|includes)\s+[\"']([^\"']+)[\"']", q, flags=re.I)
    if m:
        return m.group(2).strip()
    return None

def _is_percent_of_value_query(q: str) -> bool:
    qn = _norm(q)
    return ("pourcentage de" in qn) or ("percent of" in qn)

def _parse_numeric_stat_key(q: str) -> Optional[str]:
    qn = _norm(q)
    # Renvoie une clé: mean, median, min, max, std, sum
    if "moyenne" in qn or "mean" in qn: return "mean"
    if "médiane" in qn or "mediane" in qn or "median" in qn: return "median"
    if "minimum" in qn or re.search(r"\bmin\b", qn): return "min"
    if "maximum" in qn or re.search(r"\bmax\b", qn): return "max"
    if "écart-type" in qn or "ecart-type" in qn or "std" in qn or "standard deviation" in qn: return "std"
    if "somme" in qn or "sum" in qn: return "sum"
    if "quartiles" in qn or "percentiles" in qn: return "quartiles"
    return None

# ============================
# Corrélations & statistiques
# ============================
def eta_squared_num_vs_cat(x: pd.Series, cat: pd.Series) -> float:
    df = pd.DataFrame({"x": x, "g": cat}).dropna()
    if df.empty:
        return float("nan")
    grand_mean = df["x"].mean()
    ss_between = 0.0
    for _g, sub in df.groupby("g"):
        n = len(sub)
        if n == 0:
            continue
        ss_between += n * (sub["x"].mean() - grand_mean) ** 2
    ss_total = ((df["x"] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(ss_between / ss_total)

def cramers_v(a: pd.Series, b: pd.Series) -> float:
    df = pd.crosstab(a, b)
    n = df.values.sum()
    if n == 0:
        return float("nan")
    expected = np.outer(df.sum(axis=1), df.sum(axis=0)) / n
    chi2 = ((df - expected) ** 2 / expected)
    chi2 = chi2.replace([np.inf, -np.inf], 0).fillna(0).to_numpy().sum()
    r, k = df.shape
    return float(math.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0)

def compute_overall_correlations(df: pd.DataFrame, metas: Dict[str, ColMeta]) -> pd.DataFrame:
    cols = list(df.columns)
    res = []
    for i, c1 in enumerate(cols):
        for c2 in cols[i:]:
            m1, m2 = metas[c1], metas[c2]
            if c1 == c2:
                val = 1.0
            elif m1.is_numeric and m2.is_numeric:
                s = pd.concat([pd.to_numeric(df[c1], errors="coerce"), pd.to_numeric(df[c2], errors="coerce")], axis=1).dropna()
                val = float(s.corr(method="pearson").iloc[0, 1]) if len(s) > 1 else float("nan")
            elif m1.is_numeric and not m2.is_numeric:
                val = eta_squared_num_vs_cat(pd.to_numeric(df[c1], errors="coerce"), df[c2].astype(str))
            elif not m1.is_numeric and m2.is_numeric:
                val = eta_squared_num_vs_cat(pd.to_numeric(df[c2], errors="coerce"), df[c1].astype(str))
            else:
                val = cramers_v(df[c1].astype(str), df[c2].astype(str))
            res.append({"col1": c1, "col2": c2, "strength": val})
            if c1 != c2:
                res.append({"col1": c2, "col2": c1, "strength": val})
    out = pd.DataFrame(res)
    if not out.empty:
        out.loc[out["strength"].abs() > 1, "strength"] = np.nan
    return out

# ==================
# Streamlit UI logic
# ==================
logo_path = "EPITA_LOGO_FR.png"
st.set_page_config(
    page_title="Analyse des enquêtes",
    page_icon=logo_path if os.path.exists(logo_path) else "🧪",
    layout="wide"
)
st.markdown("""
<style>
.tag-chip{
  display:inline-block;
  padding:4px 8px;
  margin:2px 6px 2px 0;
  background:#eef2f7;
  color:#1f2937;
  border:1px solid #d1d5db;
  border-radius:14px;
  font-size:12px;
  line-height:18px;
  white-space:nowrap;
}
.soft-sep{ margin:10px 0 16px 0; border-top:1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

header_cols = st.columns([1, 6])
with header_cols[0]:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
with header_cols[1]:
    st.title("Analyse des enquêtes")
    st.caption("Mini-synthèse IA • Chatbot (analyses descriptives + RAG) • Storytelling • Analyse verbatims")

with st.sidebar:
    st.markdown("### ⚙️ Paramètres")
    embed_model = st.selectbox("Modèle d'embedding", ["text-embedding-3-small", "text-embedding-3-large"], index=0)
    gen_model = st.selectbox("Modèle de génération", ["gpt-5", "gpt-5-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"], index=0)
    st.markdown("---")
    st.markdown("**Clé OpenAI** (ou via st.secrets/ENV)")
    _use_custom_key = st.checkbox("Saisir une clé OpenAI manuellement", value=False)
    if _use_custom_key:
        _manual_key = st.text_input("OPENAI_API_KEY", type="password", value="")
        if _manual_key:
            os.environ["OPENAI_API_KEY"] = _manual_key

uploaded = st.file_uploader("Déposez un fichier Excel (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Chargez un fichier pour commencer.")
    st.stop()

# Lecture Excel
xl = pd.ExcelFile(uploaded)
with st.sidebar:
    sheet = st.selectbox("Feuille Excel", xl.sheet_names, index=0)
raw_df = xl.parse(sheet)
if raw_df.empty:
    st.warning("La feuille est vide.")
    st.stop()

# --- Suggestion colonnes à ignorer (avant tout traitement)
with st.expander("🧹 Colonnes à ignorer pour l'analyse (IDs, noms, e-mails...)", expanded=True):
    sug = suggest_ignored_columns(raw_df)
    st.write("**Suggestion** (modifiable) :")
    sel = st.multiselect("Choisissez les colonnes à exclure",
                         options=list(raw_df.columns),
                         default=sug,
                         key=f"exclude_select_{sheet}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Appliquer exclusions", key=f"apply_ex_{sheet}"):
            st.session_state[f"excluded_{sheet}"] = list(sel)
    with c2:
        if st.button("♻️ Réinitialiser", key=f"reset_ex_{sheet}"):
            st.session_state[f"excluded_{sheet}"] = []

excluded_cols = st.session_state.get(f"excluded_{sheet}", [])
if excluded_cols:
    st.success(f"Colonnes exclues : {', '.join(excluded_cols)}")

# Nettoyage léger + exclusions
df = raw_df.copy()
if excluded_cols:
    df = df.drop(columns=[c for c in excluded_cols if c in df.columns], errors="ignore")
empty_cols = [c for c in df.columns if df[c].dropna().astype(str).str.strip().eq("").all()]
if empty_cols:
    df = df.drop(columns=empty_cols)

if df.empty:
    st.warning("Après exclusions, il ne reste plus de colonnes à analyser.")
    st.stop()

# Détection des métadonnées
metas: Dict[str, ColMeta] = {c: detect_column_meta(df[c], c) for c in df.columns}

# Sélecteur
col = st.selectbox("Choisissez une question (colonne)", list(df.columns), index=0)
meta = metas[col]

TAB1, TAB2, TAB3, TAB4 = st.tabs([
    "🧠 Mini-synthèse IA",
    "🤖 Chatbot d’analyse",
    "📚 Storytelling global",
    "📝 Analyse des verbatims"
])

# =====================
# TAB 1 — MINI-SYNTHÈSE
# =====================
with TAB1:
    st.subheader(f"Mini-synthèse — {col}")
    st.caption(f"Type détecté : **{meta.supertype} / {meta.subtype}** — non nuls: {meta.n_nonnull} / {meta.n} — uniques: {meta.n_unique} ({meta.unique_ratio:.0%})")
    series = df[col]
    client = ensure_openai()
    if client is None:
        st.info("Pour la synthèse IA, renseignez une clé OpenAI valide.")
    else:
        if st.button("⚡ Générer une synthèse rapide de la colonne"):
            if meta.subtype == "texte":
                sample = series.dropna().astype(str).tolist()[:500]
                texts_df = pd.DataFrame({"text": sample})
                with st.spinner("Création des embeddings & RAG..."):
                    X = embed_texts(client, texts_df["text"].tolist(), model=embed_model)
                    nn = get_nn_index(X)
                    retrieved = retrieve("Quels sont les thèmes dominants et recommandations ?", client, texts_df, X, nn, top_k=min(200, len(texts_df)))
                    context = build_context(col, retrieved)
            else:
                context = f"Données pour '{col}' — type {meta.supertype}/{meta.subtype}.\n"
                if meta.is_numeric:
                    num = pd.to_numeric(series, errors="coerce")
                    context += f"Stats: mean={num.mean():.2f}, std={num.std():.2f}, median={num.median():.2f}, min={num.min()}, max={num.max()}\n"
                else:
                    vc = series.astype(str).value_counts().head(20)
                    context += "Top catégories:\n" + "\n".join([f"- {k}: {v}" for k, v in vc.items()])

            result = ask_llm_structured(client, gen_model, context, "Fais une synthèse exploitable en bullets.")
            st.session_state[f"synth_{sheet}_{col}"] = result  # persiste

        # Affichage persistant
        result = st.session_state.get(f"synth_{sheet}_{col}")
        if isinstance(result, dict):
            # Thèmes
            if isinstance(result.get("themes"), list) and len(result["themes"]) > 0:
                themes_df = pd.DataFrame(result["themes"])
                if "label" not in themes_df.columns:
                    themes_df["label"] = [f"Thème {i+1}" for i in range(len(themes_df))]
                share_vals = pd.to_numeric(themes_df.get("share", 0), errors="coerce").fillna(0.0)
                if share_vals.max() <= 1.0 + 1e-6:
                    share_vals = share_vals * 100.0
                themes_df["share_pct"] = share_vals

                st.markdown("**Thèmes (part estimée %)**")
                chart_themes = (
                    alt.Chart(themes_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", axis=alt.Axis(title="Thèmes", labelAngle=-35), sort='-y'),
                        y=alt.Y("share_pct:Q", title="Part estimée (%)"),
                        tooltip=["label", alt.Tooltip("share_pct:Q", format=".1f")]
                    )
                    .properties(width="container", height=320)
                )
                st.altair_chart(chart_themes, use_container_width=True)
            else:
                st.info("Aucun thème structuré retourné par le modèle.")

            # Sentiment estimé — camembert (masqué si total=0)
            pos = neu = neg = 0.0
            if isinstance(result.get("sentiment"), dict):
                sent = result["sentiment"]
                pos = float(sent.get("positif", 0) or 0)
                neu = float(sent.get("neutre", 0) or 0)
                neg = float(sent.get("négatif", sent.get("negatif", 0) or 0) or 0)
                if max(pos, neu, neg) <= 1.0 + 1e-6:
                    pos, neu, neg = pos*100.0, neu*100.0, neg*100.0
            total_sent = pos + neu + neg
            if total_sent > 0:
                st.markdown("**Sentiment estimé (%)**")
                sent_df = pd.DataFrame({"sentiment": ["positif","neutre","négatif"], "val": [pos, neu, neg]})
                pie = (
                    alt.Chart(sent_df)
                    .mark_arc(innerRadius=0)
                    .encode(theta=alt.Theta(field="val", type="quantitative"),
                            color=alt.Color(field="sentiment", type="nominal"),
                            tooltip=["sentiment", alt.Tooltip("val:Q", format=".1f")])
                    .properties(height=320, width=380)
                )
                st.altair_chart(pie, use_container_width=False)
            else:
                st.caption("_Sentiment non fourni (ou 0/0/0) — camembert masqué._")

            # Word Cloud (texte non catégoriel)
            if meta.subtype == "texte":
                st.markdown("**Word cloud**")
                if _HAS_WC:
                    wc_texts = series.dropna().astype(str).tolist()
                    wc_text = " ".join(wc_texts)
                    stop = set(WC_STOPWORDS) | FR_STOPWORDS | {"les","des","aux","avec","pour","une","un","de","la","le","et"}
                    wc = WordCloud(width=1200, height=350, background_color="white", stopwords=stop)
                    img = wc.generate(wc_text).to_image()
                    st.image(img, use_container_width=True)
                else:
                    st.info("Module `wordcloud` non installé. Faites : `pip install wordcloud`.")

            # Documentation
            with st.expander("📄 Documentation de la synthèse"):
                if result.get("insights"):
                    st.markdown("**Insights clés**")
                    st.markdown("\n".join([f"- {i}" for i in result["insights"]]))
                if result.get("actions"):
                    st.markdown("**Actions prioritaires**")
                    st.markdown("\n".join([f"- ✅ {a}" for a in result["actions"]]))
                if result.get("quotes"):
                    st.markdown("**Verbatims représentatifs (anonymisés)**")
                    for q in result["quotes"]:
                        st.markdown(f"> {strip_pii(str(q))}")

            with st.expander("🧪 JSON brut de la synthèse"):
                st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")
        else:
            st.info("Cliquez sur « ⚡ Générer une synthèse rapide » pour lancer l’analyse.")

# =====================
# TAB 2 — CHATBOT (RAG + Statistiques déterministes)
# =====================
with TAB2:
    st.subheader(f"Chatbot d'analyse — {col}")
    st.caption("Le chatbot combine des **analyses descriptives déterministes** (comptages, répartition, uniques, stats numériques...) et le **RAG/LLM** pour les questions qualitatives.")

    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {}

    series_full = df[col]
    corpus_df = pd.DataFrame({"text": series_full.astype(str).fillna("").str.strip()})
    corpus_df = corpus_df[corpus_df["text"].ne("")].reset_index(drop=True)

    client = ensure_openai()
    if client is None:
        st.error("Clé OpenAI manquante/invalide : le chatbot nécessite une clé valide.")
    else:
        # Embeddings pour le RAG (sur TOUTE la colonne)
        cache_key_emb = f"emb_{sheet}_{col}"
        cache_key_nn  = f"nn_{sheet}_{col}"
        if cache_key_emb not in st.session_state:
            with st.spinner("Calcul des embeddings de la colonne..."):
                X = embed_texts(client, corpus_df["text"].tolist(), model=embed_model)
                st.session_state[cache_key_emb] = X
                st.session_state[cache_key_nn]  = get_nn_index(X)
        X  = st.session_state[cache_key_emb]
        nn = st.session_state[cache_key_nn]

        chat_key = f"chat_{sheet}_{col}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for role, content in st.session_state[chat_key]:
            with st.chat_message(role):
                # Les tableaux/dataframes précédemment postés sont convertis en markdown
                st.markdown(content if isinstance(content, str) else str(content))

        user_msg = st.chat_input("Votre question d'analyse…")
        if user_msg:
            st.session_state[chat_key].append(("user", user_msg))
            with st.chat_message("user"):
                st.markdown(user_msg)

            answered = False

            # A) Comptage d’une valeur citée (ex: "Combien ont répondu Cycle préparatoire ?")
            if not answered:
                target_val = _best_value_from_query(user_msg, series_full)
                if target_val and re.search(r"\b(combien|nombre|nb|nbr|how\s+many|count|total|pourcentage)\b", _norm(user_msg)):
                    stats = count_value_in_column(series_full, meta, target_val)
                    if "pourcentage" in _norm(user_msg) or "percent" in _norm(user_msg):
                        answer = (
                            f"**« {col} » — Pourcentage pour \"{stats['target']}\"**\n\n"
                            f"- {stats['pct']:.1f}% ({stats['count']} / {stats['non_empty']} réponses non vides).\n"
                            f"- Valeurs uniques (non vides) : {stats['unique_non_empty']}."
                        )
                    else:
                        answer = (
                            f"**« {col} » — Comptage pour \"{stats['target']}\"**\n\n"
                            f"- Occurrences : **{stats['count']}** sur **{stats['non_empty']}** réponses non vides "
                            f"({stats['pct']:.1f}%).\n"
                            f"- Valeurs uniques (non vides) : {stats['unique_non_empty']}."
                        )
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state[chat_key].append(("assistant", answer))
                    answered = True

            # B) Comptage global
            if not answered and is_count_query(user_msg):
                cnt = column_counts(series_full)
                answer = (
                    f"**Comptage pour « {col} »**\n\n"
                    f"- Réponses non vides : **{cnt['non_empty']}**\n"
                    f"- Lignes totales (y compris vides) : {cnt['total_rows']}\n"
                    f"- Valeurs manquantes/vides : {cnt['empty_or_blank']}\n"
                    f"- Valeurs uniques (sur réponses non vides) : {cnt['unique_non_empty']}\n"
                )
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state[chat_key].append(("assistant", answer))
                answered = True

            # C) Répartition / Top-K
            if not answered and _is_freq_query(user_msg):
                top_k = _parse_topk(user_msg, default=20)
                table = frequency_table(series_full, meta, top_k=top_k)
                with st.chat_message("assistant"):
                    st.markdown(f"**Répartition (Top {top_k}) — {col}**")
                    st.dataframe(table)
                    # Chart (tri décroissant)
                    chart = (
                        alt.Chart(table)
                        .mark_bar()
                        .encode(
                            x=alt.X("valeur:N", axis=alt.Axis(labelAngle=-35), sort='-y'),
                            y=alt.Y("compte:Q", title="Comptes"),
                            tooltip=["valeur","compte", alt.Tooltip("%:Q", format=".1f")]
                        ).properties(height=320, width="container")
                    )
                    st.altair_chart(chart, use_container_width=True)
                st.session_state[chat_key].append(("assistant", table.to_markdown(index=False)))
                answered = True

            # D) Uniques (nombre ou liste)
            if not answered and (_is_unique_count_query(user_msg) or _is_unique_list_query(user_msg)):
                list_values = _is_unique_list_query(user_msg)
                nunique, vals = unique_query(series_full, list_values=list_values, top_k=100)
                if list_values and vals is not None:
                    md = f"**Valeurs uniques ({min(len(vals), 100)} premières sur {nunique}) — {col}**\n\n- " + "\n- ".join(map(str, vals))
                else:
                    md = f"**Nombre de valeurs uniques (non vides) — {col} : {nunique}**"
                with st.chat_message("assistant"):
                    st.markdown(md)
                st.session_state[chat_key].append(("assistant", md))
                answered = True

            # E) "Contiennent « … »"
            if not answered:
                phrase = _parse_contains(user_msg)
                if phrase:
                    stats = contains_phrase_count(series_full, phrase)
                    md = (
                        f"**Occurrences contenant « {stats['phrase']} » — {col}**\n\n"
                        f"- {stats['count']} / {stats['non_empty']} non vides "
                        f"({stats['pct']:.1f}%)."
                    )
                    with st.chat_message("assistant"):
                        st.markdown(md)
                    st.session_state[chat_key].append(("assistant", md))
                    answered = True

            # F) Stats numériques
            if not answered and meta.is_numeric:
                key = _parse_numeric_stat_key(user_msg)
                num = pd.to_numeric(series_full, errors="coerce")
                if key is not None:
                    if key == "sum":
                        val = float(num.sum())
                        md = f"**Somme — {col} : {val:.4f}**"
                    elif key == "mean":
                        md = f"**Moyenne — {col} : {float(num.mean()):.4f}**"
                    elif key == "median":
                        md = f"**Médiane — {col} : {float(num.median()):.4f}**"
                    elif key == "min":
                        md = f"**Minimum — {col} : {float(num.min())}**"
                    elif key == "max":
                        md = f"**Maximum — {col} : {float(num.max())}**"
                    elif key == "std":
                        md = f"**Écart-type — {col} : {float(num.std()):.4f}**"
                    elif key == "quartiles":
                        md = (
                            f"**Quartiles — {col}**\n\n"
                            f"- Q25 : {float(num.quantile(0.25)):.4f}\n"
                            f"- Médiane : {float(num.median()):.4f}\n"
                            f"- Q75 : {float(num.quantile(0.75)):.4f}\n"
                        )
                    else:
                        md = ""
                    if md:
                        with st.chat_message("assistant"):
                            st.markdown(md)
                        st.session_state[chat_key].append(("assistant", md))
                        answered = True
                else:
                    # Stats complètes si l'utilisateur demande "statistiques"
                    if "stat" in _norm(user_msg):
                        stats = numeric_summary(series_full)
                        with st.chat_message("assistant"):
                            st.markdown("**Statistiques descriptives**")
                            st.json(stats)
                        st.session_state[chat_key].append(("assistant", json.dumps(stats, ensure_ascii=False)))
                        answered = True

            # G) Sinon : RAG (qualitatif)
            if not answered:
                with st.spinner("Recherche d'extraits pertinents…"):
                    retrieved = retrieve(user_msg, client, corpus_df, X, nn, top_k=len(corpus_df))
                context = build_context(col, retrieved)
                with st.chat_message("assistant"):
                    answer = ask_llm_chat(client, gen_model, context, user_msg)
                    st.markdown(answer)
                st.session_state[chat_key].append(("assistant", answer))
                with st.expander("Extraits utilisés"):
                    st.dataframe(retrieved[["rank", "similarity", "text"]])

# ==========================
# TAB 3 — STORYTELLING
# ==========================
with TAB3:
    st.subheader("Storytelling global (corrélations & recommandations)")
    st.caption("Génère un résumé global et des recommandations. Téléchargeable en HTML.")

    client = ensure_openai()
    if client is None:
        st.info("Pour générer le storytelling, renseignez une clé OpenAI valide.")
    else:
        if st.button("📝 Générer le storytelling IA"):
            with st.spinner("Préparation du contexte…"):
                parts = []
                for c in df.columns:
                    m = metas[c]
                    if m.is_numeric:
                        num = pd.to_numeric(df[c], errors="coerce")
                        desc = f"{c} — {m.supertype}/{m.subtype} — mean={num.mean():.2f}, sd={num.std():.2f}, median={num.median():.2f}, min={num.min()}, max={num.max()}"
                    elif m.subtype in {"catégorielle","binaire","ordinale"}:
                        vc = df[c].astype(str).value_counts().head(10)
                        desc = f"{c} — {m.supertype}/{m.subtype} — top: " + ", ".join([f"{k} ({v})" for k, v in vc.items()])
                    else:
                        sample = (
                            df[c].astype(str).dropna().map(lambda x: x.strip()).loc[lambda s: s.str.len() > 0].head(50).tolist()
                        )
                        short = "; ".join([strip_pii(x[:160]) for x in sample])
                        desc = f"{c} — texte — extraits: {short}"
                    parts.append(desc)

                corrs = compute_overall_correlations(df, metas)
                strong = corrs.copy()
                strong["abs"] = strong["strength"].abs()
                strong = strong[(strong["col1"] != strong["col2"]) & strong["abs"].ge(0.25)]
                strong_lines = ", ".join([f"{r.col1}~{r.col2}={r.strength:.2f}" for r in strong.itertuples()][:30])

                # Blocs dynamiques
                data_summary = ("\n- " + "\n- ".join(parts)) if parts else ""
                corr_block = (f"\nCorrélations fortes (pré-calculées) : {strong_lines}\n" if strong_lines else "")

                # Nouveau prompt complet
                story_context = (
                    "OBJECTIF \n"
                    "Rédiger une analyse globale et des conclusions opérationnelles à partir des résultats d’une enquête, dans une perspective marketing. indique toujours dans l'analyse globale le nombre de réponses.\n\n"
                    "DONNÉES DISPONIBLES\n"
                    "Colonnes et résumés :" + data_summary + "\n"
                    + corr_block +
                    "CONTRAINTES \n\n"
                    "S’appuyer uniquement sur les données :\n\n"
                    "Ne jamais inventer de chiffres ou d’informations absentes de l’enquête.\n\n"
                    "Si certaines données sont insuffisantes pour conclure, l’indiquer explicitement.\n\n"
                    "Analyser de manière exhaustive l’ensemble des réponses.\n\n"
                    "Proposer des hypothèses lorsque c’est pertinent.\n\n"
                    "Rédiger en français, de manière claire, concise et structurée.\n\n"
                    "Rendre les chiffres comparables et lisibles :\n\n"
                    "Présenter les résultats sous forme de pourcentages et non seulement en valeurs absolues.\n\n"
                    "Arrondir les chiffres pour plus de clarté (ex. 47,3% → 47%).\n\n"
                    "Utiliser des tableaux ou puces pour résumer les données clés.\n\n"
                    "Citer les verbatims avec prudence :\n\n"
                    "Utiliser les verbatims uniquement pour illustrer une tendance, pas pour généraliser un cas isolé.\n\n"
                    "Rester neutre dans le ton :\n\n"
                    "Pas de surinterprétation ni de parti pris.\n\n"
                    "Toujours distinguer ce qui relève des faits observés vs. des hypothèses.\n\n"
                    "Valider la robustesse des corrélations :\n\n"
                    "Éviter les corrélations artificielles ou triviales.\n\n"
                    "Indiquer quand une corrélation est faible ou non significative.\n\n"
                    "Respecter une logique marketing :\n\n"
                    "Relier chaque observation à son impact business (comportement consommateur, potentiel de marché, leviers marketing).\n\n"
                    "Proposer des insights opérationnels et pas uniquement descriptifs.\n\n"
                    "Limiter le volume mais maximiser la clarté :\n\n"
                    "Préférer des phrases courtes, structurées et synthétiques.\n\n"
                    "Toujours terminer chaque grande section par une mini-conclusion (ce que ça veut dire en pratique).\n\n"
                    "Assurer la cohérence interne \n\n"
                    "Vérifier que les enseignements, corrélations, personas et recommandations s’alignent et ne se contredisent pas.\n\n"
                    "STRUCTURE ATTENDUE \n\n"
                    "1) Vue d’ensemble \n\n"
                    "Fournir une synthèse générale des résultats de l’enquête.\n\n"
                    "2) Enseignements clés \n\n"
                    "Présenter en plusieurs sous-parties : \n\n"
                    "Enseignement concis : Résumé rapide des enseignements principaux.\n\n"
                    "Enseignement détaillé : Analyse structurée avec données chiffrées, exemples et verbatims pertinents.\n\n"
                    "Top 3 enseignements à retenir : Les trois points majeurs, illustrés par des citations de répondants.\n\n"
                    "Enseignement sur les réponses ouvertes : Identifier les tendances et points d’intérêt dans les questions ouvertes.\n\n"
                    "Enseignement marketing : Mettre en avant les implications concrètes pour le marketing.\n\n"
                    "3) Corrélations \n\n"
                    "Analyser les relations entre variables : \n\n"
                    "Corrélations fortes : Identifier les liens les plus marqués (positifs ou négatifs).\n\n"
                    "Profils des répondants : Décrire les segments types (ex. : “Les répondants qui déclarent X expriment aussi Y et Z”).\n\n"
                    "Niveau d’importance : Classer les corrélations (fortes, moyennes, faibles) en précisant leur pertinence.\n\n"
                    "Présentation synthétique : Fournir une liste claire, accessible et orientée vers l’interprétation, en évitant les corrélations triviales.\n\n"
                    "4) Analyse par population \n\n"
                    "Comparer les résultats en fonction des caractéristiques socio-démographiques : \n\n"
                    "Hommes vs femmes.\n\n"
                    "Différences par âge.\n\n"
                    "Différences par zone géographique.\n"
                    "Mettre en lumière les écarts significatifs.\n\n"
                    "5) Top des personas \n\n"
                    "Proposer une segmentation en personas : \n\n"
                    "4 à 5 personas représentatifs des répondants.\n\n"
                    "Identifier 3 profils principaux liés au comportement vis-à-vis du sujet étudié.\n\n"
                    "Détailler chaque persona selon les critères socio-démographiques.\n\n"
                    "Préciser les critères de segmentation les plus pertinents d’après les résultats.\n\n"
                    "6) Recommandations marketing \n\n"
                    "Formuler des recommandations actionnables et priorisées : \n\n"
                    "Identifier les cibles marketing prioritaires.\n\n"
                    "Dresser une liste de recommandations concrètes classées par ordre de priorité.\n\n"
                    "Relier chaque recommandation aux enseignements et segments identifiés.\n\n"
                    "7) Limites & pistes de suivi \n\n"
                    "Mettre en évidence les limites de l’enquête (biais, échantillon, formulation des questions, etc.).\n\n"
                    "Proposer des pistes d’amélioration (ajouter des questions, mieux cibler certains points).\n"
                )

            with st.spinner("Génération du storytelling…"):
                messages = [
                    {"role": "system", "content": "Tu es un expert en études. Style clair, professionnel et concis."},
                    {"role": "user", "content": story_context},
                ]
                story_md = _chat_complete(client, gen_model, messages, temperature=0.2)

            st.markdown(story_md)

            # HTML download
            if _HAS_MD_CONV:
                body_html = md.markdown(
                    story_md,
                    extensions=["extra", "tables", "sane_lists", "toc", "nl2br"]
                )
            else:
                body_html = f"<pre>{story_md}</pre>"
            full_html = _wrap_html(body_html, title=f"Storytelling — {sheet}")
            b_html = io.BytesIO(full_html.encode("utf-8"))
            st.download_button(
                "⬇️ Télécharger le storytelling (HTML)",
                data=b_html,
                file_name="storytelling.html",
                mime="text/html"
            )

# ==========================
# TAB 4 — ANALYSE VERBATIMS (adjacent au Storytelling)
# ==========================
# --- Extraction multi-labels par verbatim (1 à 3) + progression/arrêt
def extract_multi_labels_progress(
    client,
    model: str,
    texts: List[str],
    batch_size: int,
    stop_flag_key: str,
    progress_placeholder,
    status_placeholder,
    with_sentiment: bool = False,
    fast_mode: bool = False,
    max_labels_per_text: int = 3,
) -> Tuple[List[List[str]], Optional[List[str]]]:
    total = len(texts)
    all_labels: List[List[str]] = []
    all_sents: List[str] = []

    if total == 0:
        progress_placeholder.progress(1.0)
        status_placeholder.info("Aucun verbatim.")
        return all_labels, (all_sents if with_sentiment else None)

    batch_size = max(1, int(batch_size))
    prog = progress_placeholder.progress(0.0)
    status_placeholder.write("Initialisation...")

    # Mode rapide (heuristique)
    if fast_mode:
        def heuristic_labels(t: str) -> List[str]:
            toks = re.findall(r"[A-Za-zÀ-ÿ']+", t)
            toks = [w for w in toks if _norm(w) not in FR_STOPWORDS]
            uniq = []
            seen = set()
            for w in toks:
                n = _norm(w)
                if n and n not in seen:
                    seen.add(n)
                    uniq.append(w.capitalize())
                if len(uniq) >= max_labels_per_text:
                    break
            return uniq if uniq else ["Autre"]

        for i, t in enumerate(texts):
            all_labels.append(heuristic_labels(t))
            if with_sentiment:
                all_sents.append("neutre")
            pct = int(round((i+1) * 100.0 / total))
            prog.progress(pct / 100.0)
        status_placeholder.success(f"Terminé (mode rapide) : {total}/{total} (100%)")
        return all_labels, (all_sents if with_sentiment else None)

    # Mode LLM
    sys = (
        "Tu es analyste. Pour CHAQUE verbatim, propose 1 à 3 étiquettes THÉMATIQUES concises (1–3 mots), "
        "dérivées UNIQUEMENT du texte, pas génériques. Les étiquettes doivent être informatives et non redondantes. "
        "Retourne UNIQUEMENT un JSON valide: une LISTE d'objets (un objet par verbatim, même ordre). "
        "Chaque objet DOIT contenir la clé 'labels' (liste de chaînes)."
    )
    if with_sentiment:
        sys += " Ajoute aussi 'sentiment' = 'positif' | 'neutre' | 'négatif'."

    example_json = (
        '[{"labels":["Communication","Management","Support"],"sentiment":"négatif"}]'
        if with_sentiment else
        '[{"labels":["Communication","Management"]}]'
    )

    prompt_tpl = (
        "Verbatims (un par ligne, à traiter dans le même ordre):\n<<CHUNK>>\n\n"
        f"Réponds avec un JSON, ex: {example_json}"
    )

    done = 0
    for i in range(0, total, batch_size):
        if st.session_state.get(stop_flag_key, False):
            status_placeholder.info(f"Arrêt demandé. Interruption après le lot courant ({done}/{total}).")
            break

        chunk_texts = [strip_pii(t) for t in texts[i:i+batch_size]]
        user = prompt_tpl.replace("<<CHUNK>>", "\n".join(f"- {t}" for t in chunk_texts))
        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]

        try:
            out = _chat_complete(client, model, msgs, temperature=0.0)
            data, _ = _try_parse_json(out)
        except Exception:
            data = None

        if not isinstance(data, list):
            # Fallback: heuristique
            data = [] 
            for t in chunk_texts:
                toks = re.findall(r"[A-Za-zÀ-ÿ']+", t)
                toks = [w for w in toks if _norm(w) not in FR_STOPWORDS]
                uniq, seen = [], set()
                for w in toks:
                    n = _norm(w)
                    if n and n not in seen:
                        seen.add(n)
                        uniq.append(w.capitalize())
                    if len(uniq) >= max_labels_per_text:
                        break
                data.append({"labels": uniq if uniq else ["Autre"]})

        # Normalisation des retours
        for obj in data:
            labels: List[str] = []
            if isinstance(obj, dict) and "labels" in obj:
                raw = obj.get("labels")
                if isinstance(raw, list):
                    labels = [str(x).strip() for x in raw if str(x).strip()]
                elif isinstance(raw, str):
                    labels = [raw.strip()]
            elif isinstance(obj, list):
                labels = [str(x).strip() for x in obj if str(x).strip()]
            elif isinstance(obj, str):
                labels = [obj.strip()]
            if not labels:
                labels = ["Autre"]
            all_labels.append(labels[:max_labels_per_text])

            if with_sentiment:
                s = "neutre"
                if isinstance(obj, dict) and "sentiment" in obj:
                    sv = str(obj.get("sentiment","neutre")).lower()
                    s = "positif" if "positif" in sv else ("négatif" if ("négatif" in sv or "negatif" in sv) else "neutre")
                all_sents.append(s)

        done = min(i + batch_size, total)
        pct = int(round(done * 100.0 / total))
        progress_placeholder.progress(pct / 100.0)
        status_placeholder.write(f"Traitement : {done}/{total} ({pct}%)")

    if done >= total:
        status_placeholder.success(f"Terminé : {done}/{total} (100%)")
        progress_placeholder.progress(1.0)

    return all_labels, (all_sents if with_sentiment else None)

with TAB4:
    st.subheader(f"Analyse des verbatims — {col}")

    # NEW: permettre de forcer l'analyse même si la détection n'est pas "texte"
    is_textlike = metas[col].subtype == "texte"
    force_anyway = False
    if not is_textlike:
        st.info("Cet onglet est disponible uniquement pour les colonnes **texte** (non catégorielles).")
        force_anyway = st.checkbox(
            "🔓 Forcer l’analyse des verbatims sur cette colonne (conversion en texte)",
            value=False,
            key=f"vbml_force_{sheet}_{col}"
        )

    if not (is_textlike or force_anyway):
        # on sort proprement si ni texte ni forcé
        st.stop()

    client = ensure_openai()
    if client is None:
        st.error("Clé OpenAI manquante/invalide : l'analyse des verbatims nécessite une clé valide.")
    else:
        import html as _html

        # -- Préparation des textes : (FORCÉ OU TEXTE) -> on cast en str et on filtre les vides
        texts_series = (
            df[col]
            .astype(object)
            .where(pd.notna(df[col]), None)
            .dropna()
            .astype(str)
            .str.strip()
        )
        texts_series = texts_series[texts_series.ne("")]
        texts = texts_series.reset_index(drop=True).tolist()
        orig_idx = texts_series.index.to_series().reset_index(drop=True)

        # -- Clés / état
        cache_base = f"{sheet}_{col}"
        key_df    = f"vbml_df_{cache_base}"
        key_page  = f"vbml_page_{cache_base}"
        key_stop_flag = f"vbml_stop_flag_{cache_base}"   # FLAG logique (pas une clé de widget)
        key_flabels   = f"vbml_filter_labels_{cache_base}"
        key_fsent     = f"vbml_filter_sent_{cache_base}"

        if key_stop_flag not in st.session_state:
            st.session_state[key_stop_flag] = False

        # -- Contrôles
        st.markdown("**Contrôles**")
        # FIX: ccom3 -> c3
        c1, c2, c3, c4, c5 = st.columns([1,1,2,3,2])
        with c1:
            batch_size = st.number_input("Taille de lot", min_value=10, max_value=200, value=40, step=10)
        with c2:
            if st.button("🛑 Arrêter", key=f"vbml_stop_btn_{cache_base}"):
                st.session_state[key_stop_flag] = True
        with c3:
            if st.button("♻️ Réinitialiser arrêt", key=f"vbml_reset_btn_{cache_base}"):
                st.session_state[key_stop_flag] = False
        with c4:
            with_sentiment = st.checkbox("Activer l'analyse du sentiment (optionnel)", value=False, key=f"vbml_sent_on_{cache_base}")
        with c5:
            show_sentiment = st.checkbox("Afficher le sentiment (si disponible)", value=True, key=f"vbml_sent_show_{cache_base}")

        fast_mode = st.checkbox("Mode rapide (mots-clés, sans LLM)", value=False, key=f"vbml_fast_{cache_base}")

        progress_ph = st.empty()
        status_ph = st.empty()

        # -- Bouton unique pour générer
        if st.button("🔎 Générer les étiquettes par verbatim", key=f"vbml_generate_{cache_base}"):
            # reset du flag d'arrêt
            st.session_state[key_stop_flag] = False

            labels_list, sentiments = extract_multi_labels_progress(
                client=client,
                model=gen_model,
                texts=texts,
                batch_size=int(batch_size),
                stop_flag_key=key_stop_flag,
                progress_placeholder=progress_ph,
                status_placeholder=status_ph,
                with_sentiment=with_sentiment,
                fast_mode=fast_mode,
                max_labels_per_text=3,
            )

            # -- Harmonisation des longueurs
            n_texts  = len(texts)
            n_labels = len(labels_list)
            n_sents  = len(sentiments) if sentiments is not None else n_texts
            n = min(n_texts, n_labels, n_sents)

            texts_aligned      = texts[:n]
            labels_aligned     = labels_list[:n]
            sentiments_aligned = sentiments[:n] if sentiments is not None else None

            def _norm_labels(x):
                if isinstance(x, list):
                    cleaned = [str(v).strip() for v in x if str(v).strip()]
                    return cleaned if cleaned else ["Autre"]
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    return ["Autre"]
                s = str(x).strip()
                return [s] if s else ["Autre"]

            labels_aligned = [_norm_labels(x) for x in labels_aligned]

            # -- DataFrame final
            vb_df = pd.DataFrame({
                "text": texts_aligned,
                "labels": labels_aligned
            })
            if sentiments_aligned is not None:
                vb_df["sentiment"] = (
                    pd.Series(sentiments_aligned, index=vb_df.index)
                    .astype(str).str.lower().replace({"negatif": "négatif"})
                )

            # -- Persistance + reset filtres/pagination
            st.session_state[key_df] = vb_df
            st.session_state[key_page] = 1
            st.session_state[key_flabels] = []
            st.session_state[key_fsent]   = []

            # -- Export TOUT
            export_all_df = _prep_export_frame(vb_df, orig_idx)
            ext_all, bytes_all, mime_all = _to_bytes_auto(export_all_df)
            st.download_button(
                label=f"⬇️ Exporter TOUTES les réponses (verbatims + labels + sentiment) en {ext_all.upper()}",
                data=bytes_all,
                file_name=f"verbatims_{sheet}_{col}.{ext_all}",
                mime=mime_all,
                key=f"vbml_export_all_{cache_base}"
            )

        # === AFFICHAGE PERSISTANT (stats + filtres + pagination)
        if key_df in st.session_state:
            vb_df: pd.DataFrame = st.session_state[key_df]

            # Filtres
            all_labels = sorted({str(lab) for labs in vb_df["labels"] for lab in (labs if isinstance(labs, list) else [labs])})
            flabels = st.multiselect("Filtrer par labels", options=all_labels,
                                     default=st.session_state.get(key_flabels, []),
                                     key=key_flabels)
            fsents: List[str] = []
            if "sentiment" in vb_df.columns:
                fsents = st.multiselect("Filtrer par sentiments",
                                        options=["positif", "neutre", "négatif"],
                                        default=st.session_state.get(key_fsent, []),
                                        key=key_fsent)

            def _row_match(row) -> bool:
                ok_label = True
                if flabels:
                    rlabels = row["labels"] if isinstance(row["labels"], list) else [row["labels"]]
                    ok_label = any(str(l) in flabels for l in rlabels)
                ok_sent = True
                if fsents and "sentiment" in vb_df.columns:
                    ok_sent = str(row.get("sentiment", "")).lower() in fsents
                return ok_label and ok_sent

            vb_show = vb_df[vb_df.apply(_row_match, axis=1)].reset_index(drop=True)

            export_filtered_df = _prep_export_frame(vb_show, orig_idx) if not vb_show.empty else pd.DataFrame(columns=["row_id", "reponse", "labels", "sentiment"])
            ext_f, bytes_f, mime_f = _to_bytes_auto(export_filtered_df)
            st.download_button(
                label=f"⬇️ Exporter les RÉSULTATS FILTRÉS en {ext_f.upper()}",
                data=bytes_f,
                file_name=f"verbatims_filtres_{sheet}_{col}.{ext_f}",
                mime=mime_f,
                key=f"vbml_export_filtered_{cache_base}"
            )

            flat = []
            for labs in vb_show["labels"]:
                if isinstance(labs, list):
            # Stats labels (tri décroissant)
                    flat.extend([str(x) for x in labs])
                else:
                    flat.append(str(labs))
            if flat:
                counts = pd.Series(flat).value_counts().reset_index()
                counts.columns = ["label", "count"]
                st.markdown("**Répartition des labels (toutes réponses)**")
                chart = (
                    alt.Chart(counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("label:N", axis=alt.Axis(labelAngle=-35), sort='-y'),
                        y=alt.Y("count:Q", title="Comptes"),
                        tooltip=["label", "count"]
                    ).properties(height=320, width="container")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Aucun label pour les filtres en cours.")
            # Stats sentiments en camembert
            if "sentiment" in vb_show.columns:
                s_counts = vb_show["sentiment"].value_counts().reindex(["positif", "neutre", "négatif"]).fillna(0)
                total_s = int(s_counts.sum())
                if total_s > 0:
                    st.markdown("**Répartition des sentiments (camembert)**")
                    s_df = s_counts.reset_index()
                    s_df.columns = ["sentiment", "val"]
                    pie = (
                        alt.Chart(s_df)
                        .mark_arc(innerRadius=0)
                        .encode(theta=alt.Theta("val:Q"),
                                color=alt.Color("sentiment:N"),
                                tooltip=["sentiment", "val"])
                        .properties(height=300, width=360)
                    )
                    st.altair_chart(pie, use_container_width=False)
            # Pagination 20 / page
            per_page = 20
            total = len(vb_show)
            total_pages = max(1, (total + per_page - 1) // per_page)
            if key_page not in st.session_state:
                st.session_state[key_page] = 1
            st.session_state[key_page] = min(st.session_state[key_page], total_pages)
            def go_prev():
                st.session_state[key_page] = max(1, st.session_state[key_page] - 1)
            def go_next():
                st.session_state[key_page] = min(total_pages, st.session_state[key_page] + 1)
            col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
            with col_nav1:
                st.button("⬅️ Précédent", on_click=go_prev, key=f"vbml_prev_{cache_base}")
            with col_nav3:
                st.button("Suivant ➡️", on_click=go_next, key=f"vbml_next_{cache_base}")
            with col_nav2:
                st.write(
                    f"Page {st.session_state[key_page]} / {total_pages} — "
                    f"Affichage {(st.session_state[key_page]-1)*per_page+1}–{min(st.session_state[key_page]*per_page, total)} sur {total}"
                )
            start = (st.session_state[key_page]-1)*per_page
            end = min(start + per_page, total)
            for i in range(start, end):
                row = vb_show.iloc[i]
                labs = row["labels"] if isinstance(row["labels"], list) else [row["labels"]]
                # Smile + texte
                prefix = ""
                if "sentiment" in vb_show.columns and row.get("sentiment") is not None:
                    s = str(row.get("sentiment", "neutre")).lower()
                    if "positif" in s:
                        prefix = f"{SMILEYS['positif']} "
                    elif ("négatif" in s) or ("negatif" in s):
                        prefix = f"{SMILEYS['négatif']} "
                    else:
                        prefix = f"{SMILEYS['neutre']} "
                st.markdown(f"**{i+1}.** {prefix}{_html.escape(str(row['text']))}")
                chips = " ".join([f"<span class='tag-chip'>{_html.escape(str(l))}</span>" for l in labs])
                st.markdown(chips, unsafe_allow_html=True)
                st.markdown("<div class='soft-sep'></div>", unsafe_allow_html=True)
            else:
                st.info("Cliquez sur « 🔎 Générer les étiquettes par verbatim » pour lancer l'analyse.")
