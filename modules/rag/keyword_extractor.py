import re
import json
import os
import logging
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """텍스트에서 중요 키워드를 추출하는 클래스"""
    
    def __init__(self, stopwords_file=None):
        """
        KeywordExtractor 초기화
        
        Args:
            stopwords_file (str, optional): 불용어 파일 경로
        """
        # 기본 한국어 불용어 목록
        self.stopwords = self._load_stopwords(stopwords_file)
        
        # TF-IDF 벡터라이저 준비
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stopwords,
            ngram_range=(1, 2),  # 단일어 및 바이그램 지원
            min_df=1,
            max_df=0.9
        )
        
        # 명사 추출기 (konlpy 또는 다른 한국어 형태소 분석기 필요)
        self.use_pos_tagger = False
        try:
            from konlpy.tag import Okt
            self.okt = Okt()
            self.use_pos_tagger = True
            logger.info("Okt(konlpy) 형태소 분석기 로드 완료")
        except ImportError:
            logger.warning("konlpy 라이브러리가 설치되지 않아 형태소 분석을 사용할 수 없습니다.")
        
        logger.info("KeywordExtractor 초기화 완료")
    
    def _load_stopwords(self, stopwords_file=None):
        """
        불용어 목록 로드
        
        Args:
            stopwords_file (str, optional): 불용어 파일 경로
            
        Returns:
            list: 불용어 목록
        """
        # 기본 한국어 불용어
        default_stopwords = [
            "이", "그", "저", "것", "수", "를", "은", "는", "이", "가", "으로", "에서",
            "하고", "하는", "하다", "한", "것", "들", "그것", "และ", "ของ", "แล้ว",
            "그리고", "또는", "그런", "이런", "저런", "하지만", "입니다", "있습니다"
        ]
        
        # 파일에서 불용어 로드
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
        """
        텍스트 전처리
        
        Args:
            text (str): 전처리할 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        if not text:
            return ""
        
        # 소문자 변환
        text = text.lower()
        
        # 특수문자 제거 (한글, 영문, 숫자만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_with_tfidf(self, texts, top_n=10):
        """
        TF-IDF를 사용하여 키워드 추출
        
        Args:
            texts (list): 텍스트 목록
            top_n (int): 추출할 키워드 수
            
        Returns:
            list: 추출된 키워드 목록
        """
        if not texts or not isinstance(texts, list):
            return []
        
        try:
            # 텍스트 전처리
            preprocessed_texts = [self.preprocess_text(text) for text in texts]
            
            # TF-IDF 행렬 생성
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_texts)
            
            # 특성명 목록
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # 문서별 가중치 합산 (평균)
            tfidf_sum = np.sum(tfidf_matrix.toarray(), axis=0)
            
            # 상위 키워드 인덱스
            top_indices = np.argsort(tfidf_sum)[-top_n:][::-1]
            
            # 키워드 및 가중치
            keywords = [(feature_names[i], tfidf_sum[i]) for i in top_indices]
            
            return keywords
        
        except Exception as e:
            logger.error(f"TF-IDF 키워드 추출 실패: {e}")
            return []
    
    def extract_with_pos_tagging(self, text, pos_types=None, top_n=10):
        """
        형태소 분석기를 사용하여 키워드 추출
        
        Args:
            text (str): 대상 텍스트
            pos_types (list, optional): 추출할 품사 유형 목록
            top_n (int): 추출할 키워드 수
            
        Returns:
            list: 추출된 키워드 목록
        """
        if not text or not self.use_pos_tagger:
            return []
        
        if pos_types is None:
            pos_types = ['Noun']  # 기본적으로 명사만 추출
        
        try:
            # 형태소 분석
            pos_tagged = self.okt.pos(text)
            
            # 지정된 품사만 필터링
            filtered_words = [word for word, pos in pos_tagged 
                            if pos in pos_types and word not in self.stopwords and len(word) > 1]
            
            # 빈도 카운트
            word_counts = Counter(filtered_words)
            
            # 상위 키워드
            top_keywords = word_counts.most_common(top_n)
            
            return top_keywords
        
        except Exception as e:
            logger.error(f"형태소 분석 키워드 추출 실패: {e}")
            return []
    
    def extract_keywords(self, text_or_texts, method="tfidf", top_n=10):
        """
        텍스트에서 키워드 추출
        
        Args:
            text_or_texts (str or list): 대상 텍스트 또는 텍스트 목록
            method (str): 추출 방법 ("tfidf" 또는 "pos")
            top_n (int): 추출할 키워드 수
            
        Returns:
            list: 추출된 키워드 목록
        """
        if isinstance(text_or_texts, str):
            text_or_texts = [text_or_texts]
        
        if method == "tfidf":
            return self.extract_with_tfidf(text_or_texts, top_n)
        elif method == "pos" and self.use_pos_tagger:
            # 여러 텍스트를 하나로 합침
            combined_text = " ".join(text_or_texts)
            return self.extract_with_pos_tagging(combined_text, top_n=top_n)
        else:
            # 기본값 또는 형태소 분석기가 없는 경우 TF-IDF 사용
            return self.extract_with_tfidf(text_or_texts, top_n)
    
    def extract_from_conversation(self, conversation_history, top_n=10):
        """
        대화 내역에서 키워드 추출
        
        Args:
            conversation_history (list): 대화 내역 [사용자, AI, 사용자, ...]
            top_n (int): 추출할 키워드 수
            
        Returns:
            list: 추출된 키워드 목록
        """
        # 사용자 메시지만 추출 (짝수 인덱스)
        user_messages = [conversation_history[i] for i in range(0, len(conversation_history), 2)
                        if i < len(conversation_history)]
        
        if not user_messages:
            return []
        
        # TF-IDF 사용 추출
        keywords_tfidf = self.extract_with_tfidf(user_messages, top_n=top_n)
        
        # 형태소 분석 사용 가능하면 추가로 추출
        keywords_pos = []
        if self.use_pos_tagger:
            combined_text = " ".join(user_messages)
            keywords_pos = self.extract_with_pos_tagging(combined_text, top_n=top_n)
        
        # 결과 병합 및 중복 제거
        all_keywords = {}
        
        for keyword, weight in keywords_tfidf:
            all_keywords[keyword] = weight
        
        for keyword, count in keywords_pos:
            # 이미 있으면 가중치 더하기 (최대 2배)
            if keyword in all_keywords:
                all_keywords[keyword] *= (1 + min(count/10, 1.0))
            else:
                all_keywords[keyword] = count
        
        # 가중치 기준 정렬
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 키워드 반환
        return sorted_keywords[:top_n]