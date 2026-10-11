"""Local lexical/semantic retrieval over sourced, allowlisted project passages."""
import json
import os
import re
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
from importlib.resources import files

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from navdiag import semantic


def normalize_query(query: str) -> str:
    """Correct close model-name typos for retrieval; preserve the user's actual question."""
    names = ('transformer', 'lstm', 'gru', 'mlp', 'cnn', 'randomforest')
    def replace(match):
        token = match.group().lower()
        if token in names:
            return token
        # Short acronyms are too ambiguous to fuzzy-match safely.
        candidates = get_close_matches(token, [name for name in names if len(name) >= 6], n=1, cutoff=0.84)
        return candidates[0] if candidates else match.group()
    return re.sub(r'[A-Za-z]+', replace, query)


def expand_retrieval_query(query: str) -> str:
    """Small, inspectable GNSS vocabulary expansion, not a semantic embedding model."""
    normalized = normalize_query(query)
    aliases = (
        (('后面的观测', '之后的观测', '未来数据', '未来观测', '未来信息', 'look ahead'), ' 因果 时间窗口 未来 泄漏'),
        (('信号强度', '强信号', '高载噪比', '弱信号'), ' C/N0 载噪比 反射 NLOS'),
        (('信号质量', 'receiver signal quality'), ' C/N0 载噪比 噪声'),
        (('楼挡', '被挡住', '直达信号', '直达路径'), ' LOS NLOS 遮挡 多径'),
        (('交换信息', '其他卫星', '跨卫星'), ' 跨卫星 注意力 LSTM Transformer'),
        (('双向', 'bilstm'), ' 历史窗口 未来 泄漏'),
        (('迁移', '新地方', '新城市', '跨城市', '跨区域', '泛化'), ' 泛化 域偏移 同域 跨区域'),
        (('地图标签', '人工标', 'unknown', '未标注', '覆盖率'), ' UrbanNav 静态建筑 参考标签 unknown 覆盖率'),
        (('一趟车', '打散', '相邻时间', '相邻的时间', '相邻的两个时间'), ' 完整采集 行程 分组 训练 测试 泄漏'),
        (('坏信号', '坏卫星'), ' NLOS Precision Recall 误报 漏报'),
        (('楼房地图', '参考类别'), ' 静态建筑 标签 unknown 覆盖率'),
    )
    for terms, expansion in aliases:
        if any(term in normalized.lower() for term in terms):
            normalized += expansion
    if any(term in normalized for term in ('城市', '外地', '地区', '道路', '区域')) and any(
            term in normalized for term in ('推广', '代表', '使用', '适用')):
        normalized += ' 跨城市 泛化 域偏移'
    return normalized


def _tokens(text):
    lower = normalize_query(text).lower().replace('c/n0', 'cn0')
    english_stop = {'a', 'an', 'the', 'i', 'we', 'you', 'do', 'does', 'is', 'are', 'was', 'to',
                    'in', 'on', 'of', 'for', 'and', 'or', 'how', 'what', 'why', 'can', 'it', 'this'}
    tokens = [word for word in re.findall(r'[a-z][a-z0-9_]*', lower) if word not in english_stop]
    stop = {'为什么', '是否', '什么', '如何', '哪些', '解释', '我们', '这个', '可以', '目前',
            '比较', '说明', '问题', '得到', '已经', '本身'}
    for phrase in re.findall(r'[\u4e00-\u9fff]+', lower):
        tokens.extend(phrase[index:index + 2] for index in range(len(phrase) - 1)
                      if phrase[index:index + 2] not in stop)
    return tokens


@lru_cache(maxsize=1)
def _index():
    resources = files('navdiag').joinpath('resources')
    notes = json.loads(resources.joinpath('knowledge.json').read_text(encoding='utf-8'))
    document_path = resources.joinpath('rag_documents.json')
    documents = json.loads(document_path.read_text(encoding='utf-8'))['chunks'] if document_path.is_file() else []
    records = [{**note, 'document_id': note['id'], 'heading': note['title']} for note in notes] + documents
    texts = [record['title'] + ' ' + record.get('heading', '') + ' ' + record['content'] + ' ' +
             ' '.join(record['keywords']) for record in records]
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    counts = [Counter(_tokens(text + ' ' + records[i]['title'] + ' ' + ' '.join(records[i]['keywords'])))
              for i, text in enumerate(texts)]
    lengths = np.array([sum(count.values()) for count in counts], dtype=float)
    document_frequency = Counter(token for count in counts for token in count)
    return records, vectorizer, matrix, counts, lengths, document_frequency


def retrieve_knowledge(query: str, limit: int = 3, mode: str | None = None) -> dict:
    if not query.strip():
        raise ValueError('Search query cannot be empty')
    if not 1 <= limit <= 5:
        raise ValueError('Search limit must be between 1 and 5')
    mode = mode or os.environ.get('NAVDIAG_RETRIEVAL_MODE', 'auto')
    if mode not in ('auto', 'sparse', 'hybrid'):
        raise ValueError('Retrieval mode must be auto, sparse, or hybrid')
    records, vectorizer, matrix, counts, lengths, df = _index()
    normalized = expand_retrieval_query(query)
    # Chinese encoders can under-rank Latin acronyms. Keep exact technical terms
    # eligible through the independent lexical channel, not generic language.
    technical_anchor = bool(re.search(
        r'(?<![a-z])(gnss|gps|nlos|los|hdop|cn0|c/n0|gru|lstm|cnn|mlp|imu|kalman|dataset|synthetic)(?![a-z])'
        r'|载噪比|历元|泄漏|伪距|卫星|遮挡|泛化|阈值', normalize_query(query).lower()))
    char_scores = (matrix @ vectorizer.transform([normalized]).T).toarray().ravel()
    bm25 = np.zeros(len(records))
    for token in set(_tokens(normalized)):
        frequency = np.array([count[token] for count in counts], dtype=float)
        idf = np.log(1 + (len(records) - df[token] + 0.5) / (df[token] + 0.5))
        bm25 += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * lengths / lengths.mean()))
    raw_bm25 = bm25.copy()
    bm25 = bm25 / bm25.max() if bm25.max() else bm25
    scores = 0.55 * char_scores + 0.45 * bm25
    # Short curated notes are useful anchors, without excluding more specific document passages.
    scores += np.array([0.03 if record['id'].startswith('K') and char_scores[i] > 0.08 else 0
                        for i, record in enumerate(records)])
    semantic_scores, fallback = None, None
    if mode != 'sparse':
        try:
            semantic_scores = semantic.similarity(normalized, records)
            if semantic_scores is None:
                fallback = 'local_encoder_not_installed'
        except (ImportError, OSError, ValueError, RuntimeError, KeyError):
            fallback = 'local_encoder_unavailable'
    method = 'bm25_char_tfidf_v2'
    if semantic_scores is not None:
        method = 'bge_bm25_char_v2'
        # Core weights/gates use development queries. Exact-term eligibility was
        # restored after checking the existing agent regression suite.
        scores = 0.55 * semantic_scores + 0.45 * scores
    selected, per_document, token_sets = [], Counter(), []
    for index in np.argsort(-scores, kind='stable'):
        # BM25/max is a ranking score, not confidence: a lone accidental term
        # must not become relevant just because it is the maximum in this corpus.
        lexical_relevant = char_scores[index] >= 0.065 and raw_bm25[index] >= 2.0
        anchored_lexical = technical_anchor and char_scores[index] >= 0.035 and raw_bm25[index] >= 2.0
        lexical_relevant |= anchored_lexical
        if semantic_scores is None and not lexical_relevant:
            continue
        if semantic_scores is not None and semantic_scores[index] < 0.46 and not anchored_lexical:
            continue
        record = records[index]
        if per_document[record['document_id']] >= 2:
            continue
        terms = set(_tokens(record['content']))
        if any(len(terms & previous) / max(1, len(terms | previous)) > 0.8 for previous in token_sets):
            continue
        selected.append({**record, 'relevance_score': float(scores[index]),
                         'retrieval_method': method,
                         'search_query': normalized,
                         'retrieval_scores': {'bm25_normalized': float(bm25[index]),
                                              'bm25_raw': float(raw_bm25[index]),
                                              'char_tfidf': float(char_scores[index]),
                                              'semantic_cosine': float(semantic_scores[index])
                                                  if semantic_scores is not None else None}})
        per_document[record['document_id']] += 1
        token_sets.append(terms)
        if len(selected) == limit:
            break
    return {'query': query, 'retrieval_query': normalized, 'retrieval_method': method,
            'requested_mode': mode, 'fallback_reason': fallback,
            'embedding_model': semantic.MODEL_ID if semantic_scores is not None else None,
            'score_notice': 'Ranking similarities, not calibrated probabilities or answer confidence.',
            'notes': selected}


def search_knowledge(query: str, limit: int = 3, mode: str | None = None) -> list[dict]:
    return retrieve_knowledge(query, limit, mode)['notes']
