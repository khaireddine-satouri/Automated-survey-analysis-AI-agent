# Streamlit — Survey Analysis
Summary + Chatbot + Storytelling + Verbatim Analysis

## Features

### Chatbot
- Deterministic descriptive analyses:
  - Counts
  - Top-K distribution
  - Unique values
  - Search "contain ..."
  - Percentage of a value
  - Numerical statistics
- RAG / LLM for qualitative questions

### Main functionalities
- Exact count: "How many answered … <value>" across the entire column
- Automatic suggestion of columns to ignore (IDs, names, emails...) with apply / reset option
- Verbatim analysis:
  - Descending sort of labels
  - Filters by labels / sentiments
  - Mood indicators
  - Sentiment pie chart
  - Pagination (20 responses per page)
  - Persistent charts
- Mini-summary:
  - Wordcloud for text columns
  - Sentiment pie chart (hidden if no results)

## Requirements

Install the required dependencies:

```bash
pip install streamlit pandas numpy scikit-learn unidecode openai openpyxl altair wordcloud markdown
