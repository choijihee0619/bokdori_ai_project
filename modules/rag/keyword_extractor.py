import re
import json
import os
import logging
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Java 없이 동작하도록 soynlp 기반 토크나이저 사용
try:
    from soynlp.tokenizer import LTokenizer
    word_scores = {
        '복도리': 10.0, 'AI': 9.0, '비서': 8.5,
        '노인': 8.0, '감정': 7.5, '대화': 7.0
    }
    tokenizer = LTokenizer(scores=word_scores)
    use_tokenizer = True
    logger.info("LTokenizer(soynlp) 로드 완료")
except ImportError:
    tokenizer = None
    use_tokenizer = False
    logger.warning("soynlp가 설치되지 않아 토크나이저를 사용할 수 없습니다.")

class KeywordExtractor:
    """텍스트에서 중요 키워드를 추출하는 클래스"""
    
    def __init__(self, stopwords_file=None):
        self.stopwords = self._load_stopwords(stopwords_file)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )

        self.use_tokenizer = use_tokenizer
        self.tokenizer = tokenizer

        logger.info("KeywordExtractor 초기화 완료")

    def _load_stopwords(self, stopwords_file=None):
        default_stopwords = [
            "이", "그", "저", "것", "수", "를", "은", "는", "이", "가", "으로", "에서",
            "하고", "하는", "하다", "한", "것", "들", "그것", "그리고", "또는", "그런",
            "이런", "저런", "하지만", "입니다", "있습니다"
        ]
        if stopwords_file and os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    custom_stopwords = [line.strip() for line in f if line.strip()]
                logger.info(f"불용어 파일 로드 완료: {stopwords_file}, {len(custom_stopwords)}개")
                return default_stopwords + custom_stopwords
            except Exception as e:
                logger.error(f"불용어 파일 로드 실패: {e}")
        return default_stopwords

    def preprocess_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_with_tfidf(self, texts, top_n=10):
        if not texts or not isinstance(texts, list):
            return []
        try:
            preprocessed = [self.preprocess_text(t) for t in texts]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_sum = np.sum(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(tfidf_sum)[-top_n:][::-1]
            keywords = [(feature_names[i], tfidf_sum[i]) for i in top_indices]
            return keywords
        except Exception as e:
            logger.error(f"TF-IDF 키워드 추출 실패: {e}")
            return []

    def extract_with_tokenizer(self, text, top_n=10):
        if not self.use_tokenizer or not text:
            return []
        tokens = self.tokenizer.tokenize(self.preprocess_text(text))
        counts = Counter(t for t in tokens if t not in self.stopwords and len(t) > 1)
        return counts.most_common(top_n)

    def extract_keywords(self, text_or_texts, method="tfidf", top_n=10):
        if isinstance(text_or_texts, str):
            text_or_texts = [text_or_texts]
        if method == "tfidf":
            return self.extract_with_tfidf(text_or_texts, top_n)
        elif method == "tokenizer" and self.use_tokenizer:
            combined = " ".join(text_or_texts)
            return self.extract_with_tokenizer(combined, top_n=top_n)
        else:
            return self.extract_with_tfidf(text_or_texts, top_n)

    def extract_from_conversation(self, conversation_history, top_n=10):
        user_msgs = [conversation_history[i] for i in range(0, len(conversation_history), 2)]
        if not user_msgs:
            return []
        tfidf_keywords = self.extract_with_tfidf(user_msgs, top_n)
        tokenizer_keywords = []
        if self.use_tokenizer:
            combined_text = " ".join(user_msgs)
            tokenizer_keywords = self.extract_with_tokenizer(combined_text, top_n)

        all_keywords = {}
        for word, weight in tfidf_keywords:
            all_keywords[word] = weight
        for word, count in tokenizer_keywords:
            if word in all_keywords:
                all_keywords[word] *= (1 + min(count / 10, 1.0))
            else:
                all_keywords[word] = count

        sorted_kws = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_kws[:top_n]