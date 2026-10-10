"""Local searchable technical notes: Unicode character TF-IDF, with source IDs."""
import json
from importlib.resources import files

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def search_knowledge(query: str, limit: int = 3) -> list[dict]:
    if not query.strip():
        raise ValueError('Search query cannot be empty')
    if not 1 <= limit <= 5:
        raise ValueError('Search limit must be between 1 and 5')
    notes = json.loads(files('navdiag').joinpath('resources/knowledge.json').read_text(encoding='utf-8'))
    texts = [note['title'] + ' ' + note['content'] + ' ' + ' '.join(note['keywords']) for note in notes]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    selected = np.argsort(-scores, kind='stable')[:limit]
    return [{**notes[index], 'relevance_score': float(scores[index])} for index in selected if scores[index] > 0]
