import os
import json
import re
import logging
from datetime import datetime, timedelta
import numpy as np

# 로컬 모듈 임포트
from modules.utils.helpers import load_config

logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """대화 내용에서 감정을 분석하는 클래스"""
    
    def __init__(self, config=None):
        """
        EmotionAnalyzer 초기화
        
        Args:
            config (dict, optional): 설정 정보. 기본값은 None
        """
        self.config = config or load_config()
        
        # 감정 패턴 로드
        self.patterns = self._load_emotion_patterns()
        
        # 감정 분류
        self.emotion_categories = {
            "positive": ["기쁨", "행복", "만족", "흥미", "기대", "사랑"],
            "negative": ["슬픔", "분노", "불안", "공포", "우울", "절망", "실망"],
            "neutral": ["평온", "무관심", "집중", "고요"]
        }
        
        logger.info("EmotionAnalyzer 초기화 완료")
    
    def _load_emotion_patterns(self):
        """
        감정 패턴 파일 로드
        
        Returns:
            dict: 감정 패턴 정보
        """
        patterns_path = "config/emotion_patterns.json"
        
        # 기본 패턴 설정
        default_patterns = {
            "emotions": {
                "기쁨": ["좋아", "기쁘", "행복", "웃", "신나", "즐겁", "재밌"],
                "슬픔": ["슬퍼", "우울", "눈물", "울", "속상", "마음이 아프", "고통스럽"],
                "분노": ["화나", "짜증", "열받", "분노", "화가 나", "짜증나", "미치겠"],
                "불안": ["걱정", "불안", "초조", "두렵", "무섭", "떨려", "긴장"],
                "우울": ["우울", "의미 없", "공허", "허무", "살기 싫", "절망", "희망이 없"],
                "평온": ["평온", "고요", "침착", "차분", "편안", "안정"]
            },
            "intensity_modifiers": {
                "high": ["매우", "정말", "너무", "엄청", "굉장히"],
                "low": ["조금", "약간", "살짝", "다소"]
            },
            "negation_words": ["아니", "않", "없", "말", "못"]
        }
        
        try:
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                
                logger.info(f"감정 패턴 로드 완료: {patterns_path}")
                return patterns
            else:
                logger.warning(f"감정 패턴 파일을 찾을 수 없음: {patterns_path}. 기본 패턴을 사용합니다.")
                # 기본 패턴 저장
                self._save_default_patterns(default_patterns, patterns_path)
                return default_patterns
        
        except Exception as e:
            logger.error(f"감정 패턴 파일 로드 실패: {e}. 기본 패턴을 사용합니다.")
            return default_patterns
    
    def _save_default_patterns(self, patterns, file_path):
        """기본 패턴 저장"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(patterns, f, ensure_ascii=False, indent=2)
            logger.info(f"기본 감정 패턴 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"기본 감정 패턴 저장 실패: {e}")
    
    def analyze_text(self, text):
        """
        텍스트에서 감정 분석
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감정 분석 결과
        """
        if not text or len(text) < 3:
            return {
                "dominant_emotion": "unknown",
                "emotion_scores": {},
                "emotion_category": "unknown",
                "confidence": 0.0
            }
        
        # 감정 점수 초기화
        emotion_scores = {emotion: 0 for emotion in self.patterns["emotions"].keys()}
        
        # 각 감정에 대한 키워드 검사
        for emotion, keywords in self.patterns["emotions"].items():
            for keyword in keywords:
                # 정규 표현식으로 키워드 검색 (단어 경계 고려)
                matches = re.finditer(r'\b' + keyword + r'\w*\b', text.lower())
                
                for match in matches:
                    # 기본 점수
                    score = 1.0
                    
                    # 맥락 확인 (주변 텍스트)
                    context_start = max(0, match.start() - 20)
                    context_end = min(len(text), match.end() + 20)
                    context = text[context_start:context_end].lower()
                    
                    # 강도 수정자 확인
                    for high_mod in self.patterns["intensity_modifiers"]["high"]:
                        if high_mod in context:
                            score *= 1.5
                            break
                    
                    for low_mod in self.patterns["intensity_modifiers"]["low"]:
                        if low_mod in context:
                            score *= 0.7
                            break
                    
                    # 부정어 확인
                    for neg in self.patterns["negation_words"]:
                        if neg in context:
                            score *= -0.5  # 부정이면 감정 반전 (약화)
                            break
                    
                    # 점수 누적
                    emotion_scores[emotion] += score
        
        # 최종 점수 계산 및 정규화
        total_score = sum(abs(score) for score in emotion_scores.values())
        
        if total_score > 0:
            # 점수 정규화 (0-1 사이)
            normalized_scores = {e: abs(s)/total_score for e, s in emotion_scores.items()}
            
            # 가장 강한 감정 찾기
            dominant_emotion = max(normalized_scores, key=normalized_scores.get)
            dominant_score = normalized_scores[dominant_emotion]
        else:
            normalized_scores = {e: 0 for e in emotion_scores}
            dominant_emotion = "neutral"
            dominant_score = 0.0
        
        # 감정 카테고리 결정
        if dominant_emotion in self.emotion_categories["positive"]:
            category = "positive"
        elif dominant_emotion in self.emotion_categories["negative"]:
            category = "negative"
        else:
            category = "neutral"
        
        # 신뢰도 계산 (주요 감정과 다음 감정의 점수 차이)
        scores_list = sorted(normalized_scores.values(), reverse=True)
        confidence = dominant_score
        if len(scores_list) > 1 and scores_list[1] > 0:
            confidence = (scores_list[0] - scores_list[1]) / scores_list[0]
        
        # 주요 키워드 추출
        keywords = []
        for emotion, keywords_list in self.patterns["emotions"].items():
            for keyword in keywords_list:
                if re.search(r'\b' + keyword + r'\w*\b', text.lower()):
                    keywords.append(keyword)
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_category": category,
            "emotion_scores": normalized_scores,
            "confidence": min(confidence, 1.0),  # 0-1 사이로 제한
            "keywords": list(set(keywords))[:5]  # 중복 제거 및 상위 5개만
        }
    
    def analyze_conversation(self, conversation_history):
        """
        대화 내역 전체 분석
        
        Args:
            conversation_history (list): 대화 내역 목록 [사용자 발화, AI 응답, ...]
            
        Returns:
            dict: 대화 감정 분석 결과
        """
        # 사용자 발화만 추출 (짝수 인덱스)
        user_messages = [conversation_history[i] for i in range(0, len(conversation_history), 2) if i < len(conversation_history)]
        
        if not user_messages:
            return {
                "overall_emotion": "neutral",
                "emotion_trend": "stable",
                "emotion_distribution": {"neutral": 1.0},
                "emotion_scores_by_message": []
            }
        
        # 각 메시지별 감정 분석
        emotion_results = []
        for message in user_messages:
            result = self.analyze_text(message)
            emotion_results.append(result)
        
        # 전체 감정 분포 계산
        emotion_counts = {}
        for result in emotion_results:
            category = result["emotion_category"]
            emotion_counts[category] = emotion_counts.get(category, 0) + 1
        
        # 정규화된 분포
        total_messages = len(emotion_results)
        emotion_distribution = {category: count/total_messages for category, count in emotion_counts.items()}
        
        # 전반적인 감정 결정 (가장 많은 카테고리)
        overall_emotion = max(emotion_distribution, key=emotion_distribution.get) if emotion_distribution else "neutral"
        
        # 감정 변화 추세 분석
        if len(emotion_results) > 2:
            # 감정 카테고리를 숫자로 변환 (긍정:1, 중립:0, 부정:-1)
            emotion_values = []
            for result in emotion_results:
                if result["emotion_category"] == "positive":
                    emotion_values.append(1)
                elif result["emotion_category"] == "negative":
                    emotion_values.append(-1)
                else:
                    emotion_values.append(0)
            
            # 단순 선형 회귀로 추세 계산
            x = np.array(range(len(emotion_values)))
            y = np.array(emotion_values)
            
            if len(x) > 1:  # 최소 2개 이상의 포인트 필요
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0.1:
                    trend = "improving"  # 감정이 좋아지는 추세
                elif slope < -0.1:
                    trend = "worsening"  # 감정이 나빠지는 추세
                else:
                    trend = "stable"     # 안정적인 감정 상태
            else:
                trend = "stable"
        else:
            trend = "stable"  # 대화가 짧으면 안정적으로 간주
        
        return {
            "overall_emotion": overall_emotion,
            "emotion_trend": trend,
            "emotion_distribution": emotion_distribution,
            "emotion_scores_by_message": emotion_results
        }
    
    def get_emotion_keywords(self, text, top_n=5):
        """
        텍스트에서 감정 관련 키워드 추출
        
        Args:
            text (str): 분석할 텍스트
            top_n (int): 반환할 키워드 수
            
        Returns:
            list: 추출된 감정 키워드
        """
        result = self.analyze_text(text)
        return result.get("keywords", [])[:top_n]