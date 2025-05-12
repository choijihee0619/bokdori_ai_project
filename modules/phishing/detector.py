import json
import re
import os
import logging
from dotenv import load_dotenv

# 로컬 모듈 임포트
from modules.langchain.chains import ChainManager

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

class PhishingDetector:
    """보이스피싱 감지 클래스"""
    
    def __init__(self, config=None):
        """
        PhishingDetector 초기화
        
        Args:
            config (dict, optional): 구성 정보. 기본값은 None
        """
        self.config = config or {}
        
        # 보이스피싱 패턴 로드
        self.patterns = self._load_patterns()
        
        # 임계값 설정
        phishing_config = self.config.get("phishing_detection", {})
        self.threshold = phishing_config.get("threshold", 0.7)
        
        # 보이스피싱 감지용 LLM 체인 생성
        self.chain_manager = ChainManager(self.config)
        
        logger.info(f"PhishingDetector 초기화 완료: threshold={self.threshold}, {len(self.patterns['high_risk'])} 고위험 패턴 로드")
    
    def _load_patterns(self):
        """
        보이스피싱 패턴 파일 로드
        
        Returns:
            dict: 보이스피싱 패턴 정보
        """
        patterns_path = "config/phishing_patterns.json"
        
        # 기본 패턴 설정
        default_patterns = {
            "high_risk": ["계좌번호", "보안코드", "인증번호", "비밀번호", "OTP"],
            "medium_risk": ["송금", "이체", "금융사고", "검찰", "경찰", "금감원"],
            "low_risk": ["급한", "긴급", "빨리", "지금 당장"]
        }
        
        try:
            if os.path.exists(patterns_path):
                with open(patterns_path, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                
                # 기본 키가 있는지 확인
                if all(k in patterns for k in ["high_risk", "medium_risk", "low_risk"]):
                    logger.info(f"보이스피싱 패턴 로드 완료: {patterns_path}")
                    return patterns
                else:
                    logger.warning(f"불완전한 패턴 파일: {patterns_path}. 기본 패턴을 사용합니다.")
                    return default_patterns
            else:
                logger.warning(f"패턴 파일을 찾을 수 없음: {patterns_path}. 기본 패턴을 사용합니다.")
                return default_patterns
        
        except Exception as e:
            logger.error(f"패턴 파일 로드 실패: {e}. 기본 패턴을 사용합니다.")
            return default_patterns
    
    def detect_with_patterns(self, text):
        """
        규칙 기반 보이스피싱 감지
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감지 결과
        """
        # 타입 검증 추가
        if not isinstance(text, str):
            try:
                text = str(text)
                logger.warning(f"문자열이 아닌 입력을 문자열로 변환: {type(text).__name__} -> str")
            except Exception as e:
                logger.error(f"텍스트 변환 실패: {e}")
                return {"risk_level": "unknown", "score": 0, "keywords": [], "explanation": "유효하지 않은 텍스트 형식입니다."}
        
        if not text or len(text) < 5:
            return {"risk_level": "unknown", "score": 0, "keywords": [], "explanation": "텍스트가 너무 짧습니다."}
        
        # 소문자 변환 및 특수문자 제거
        normalized_text = text.lower()
        
        # 감지된 키워드
        detected_keywords = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }
        
        # 키워드 검색
        for level, keywords in self.patterns.items():
            for keyword in keywords:
                # 키워드 타입 검증 추가
                if not isinstance(keyword, str):
                    continue
                
                if keyword.lower() in normalized_text:
                    detected_keywords[level].append(keyword)
        
        # 점수 계산
        score = (
            len(detected_keywords["high_risk"]) * 0.5 +
            len(detected_keywords["medium_risk"]) * 0.3 +
            len(detected_keywords["low_risk"]) * 0.1
        )
        
        # 정규화된 점수 (0-1 사이)
        normalized_score = min(score, 1.0)
        
        # 위험 수준 결정
        if normalized_score >= 0.7:
            risk_level = "high"
        elif normalized_score >= 0.4:
            risk_level = "medium"
        elif normalized_score > 0:
            risk_level = "low"
        else:
            risk_level = "safe"
        
        # 키워드 합치기
        all_keywords = detected_keywords["high_risk"] + detected_keywords["medium_risk"] + detected_keywords["low_risk"]
        
        # 설명 생성
        if risk_level == "high":
            explanation = "다수의 고위험 보이스피싱 징후가 감지되었습니다. 주의하세요!"
        elif risk_level == "medium":
            explanation = "일부 보이스피싱 관련 용어가 감지되었습니다. 의심스러운 부분이 있습니다."
        elif risk_level == "low":
            explanation = "약간의 의심스러운 표현이 포함되어 있습니다. 주의가 필요할 수 있습니다."
        else:
            explanation = "보이스피싱 징후가 감지되지 않았습니다."
        
        return {
            "risk_level": risk_level,
            "score": normalized_score,
            "keywords": all_keywords,
            "explanation": explanation
        }
    
    def detect_with_llm(self, text):
        """
        LLM 기반 보이스피싱 감지
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감지 결과
        """
        # 타입 검증 추가
        if not isinstance(text, str):
            try:
                text = str(text)
                logger.warning(f"문자열이 아닌 입력을 문자열로 변환: {type(text).__name__} -> str")
            except Exception as e:
                logger.error(f"텍스트 변환 실패: {e}")
                return {"risk_level": "unknown", "score": 0, "keywords": [], "explanation": "유효하지 않은 텍스트 형식입니다."}
        
        if not text or len(text) < 10:
            return {"risk_level": "unknown", "score": 0, "keywords": [], "explanation": "텍스트가 너무 짧습니다."}
        
        try:
            # 보이스피싱 감지 체인 가져오기
            chain = self.chain_manager.get_phishing_detection_chain()
            
            # 텍스트 분석
            result = chain.run(text=text)
            
            # 결과 파싱
            parsed = self._parse_llm_result(result)
            
            return parsed
        
        except Exception as e:
            logger.error(f"LLM 기반 보이스피싱 감지 실패: {e}")
            
            # 실패 시 규칙 기반 분석으로 대체
            return self.detect_with_patterns(text)
    
    def _parse_llm_result(self, result):
        """
        LLM 결과 파싱
        
        Args:
            result (str): LLM 결과 문자열
            
        Returns:
            dict: 구조화된 결과
        """
        # 타입 검증 추가
        if not isinstance(result, str):
            try:
                result = str(result)
                logger.warning(f"문자열이 아닌 LLM 결과를 문자열로 변환: {type(result).__name__} -> str")
            except Exception as e:
                logger.error(f"LLM 결과 변환 실패: {e}")
                return {
                    "risk_level": "unknown",
                    "score": 0,
                    "keywords": [],
                    "explanation": "분석 결과를 해석할 수 없습니다."
                }
        
        try:
            # 기본값 설정
            parsed = {
                "risk_level": "unknown",
                "score": 0,
                "keywords": [],
                "explanation": "분석 결과를 해석할 수 없습니다."
            }
            
            # 확률 추출 시도
            probability_match = re.search(r'확률[:\s]*([0-9.]+)', result)
            if probability_match:
                try:
                    parsed["score"] = float(probability_match.group(1))
                except (ValueError, TypeError):
                    parsed["score"] = 0
            
            # 위험 수준 추출 시도
            level_match = re.search(r'위험[^:]*[:\s]*(안전|주의|경고|위험)', result)
            if level_match:
                level = level_match.group(1)
                if level == "안전":
                    parsed["risk_level"] = "safe"
                elif level == "주의":
                    parsed["risk_level"] = "low"
                elif level == "경고":
                    parsed["risk_level"] = "medium"
                elif level == "위험":
                    parsed["risk_level"] = "high"
            
            # 키워드 추출 시도
            keywords_match = re.search(r'키워드[^:]*[:\s]*(.*?)(?:\n|$)', result)
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                parsed["keywords"] = [k.strip() for k in keywords_str.split(',') if k.strip()]
            
            # 설명 추출 시도
            explanation_match = re.search(r'설명[^:]*[:\s]*(.*?)(?:\n|대응|\Z)', result, re.DOTALL)
            if explanation_match:
                parsed["explanation"] = explanation_match.group(1).strip()
            
            return parsed
        
        except Exception as e:
            logger.error(f"LLM 결과 파싱 실패: {e}")
            return {
                "risk_level": "unknown",
                "score": 0,
                "keywords": [],
                "explanation": "분석 결과를 해석할 수 없습니다."
            }
    
    def detect(self, text):
        """
        텍스트에서 보이스피싱 감지 (패턴 기반 + LLM 기반)
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 감지 결과
        """
        # 타입 검증 추가
        if not isinstance(text, str):
            try:
                text = str(text)
                logger.warning(f"문자열이 아닌 입력을 문자열로 변환: {type(text).__name__} -> str")
            except Exception as e:
                logger.error(f"텍스트 변환 실패: {e}")
                return {
                    "is_phishing": False,
                    "risk_level": "unknown",
                    "score": 0,
                    "keywords": [],
                    "explanation": "유효하지 않은 텍스트 형식입니다.",
                    "method": "none"
                }
        
        # 규칙 기반 분석
        pattern_result = self.detect_with_patterns(text)
        
        # 규칙 기반 분석 점수가 낮은 경우 LLM 분석 스킵
        if pattern_result["score"] < 0.2:
            return {
                "is_phishing": False,
                "risk_level": pattern_result["risk_level"],
                "score": pattern_result["score"],
                "keywords": pattern_result["keywords"],
                "explanation": pattern_result["explanation"],
                "method": "pattern"
            }
        
        # LLM 기반 분석
        llm_result = self.detect_with_llm(text)
        
        # 결합된 결과
        combined_score = max(pattern_result["score"], llm_result["score"])
        
        # 키워드 타입 안전성 확보
        pattern_keywords = pattern_result.get("keywords", []) or []
        llm_keywords = llm_result.get("keywords", []) or []
        
        # 리스트가 아닌 경우 빈 리스트로 처리
        if not isinstance(pattern_keywords, list):
            pattern_keywords = []
        if not isinstance(llm_keywords, list):
            llm_keywords = []
        
        # 키워드 합치기 (중복 제거)
        combined_keywords = []
        for keyword in pattern_keywords + llm_keywords:
            if isinstance(keyword, str) and keyword not in combined_keywords:
                combined_keywords.append(keyword)
        
        # 위험 수준 결정
        if combined_score >= self.threshold:
            risk_level = "high"
            is_phishing = True
        elif combined_score >= self.threshold * 0.7:
            risk_level = "medium"
            is_phishing = True
        elif combined_score >= self.threshold * 0.3:
            risk_level = "low"
            is_phishing = False
        else:
            risk_level = "safe"
            is_phishing = False
        
        # 설명 선택 (LLM 설명 우선)
        llm_explanation = llm_result.get("explanation", "")
        pattern_explanation = pattern_result.get("explanation", "")
        
        if not isinstance(llm_explanation, str):
            llm_explanation = ""
        if not isinstance(pattern_explanation, str):
            pattern_explanation = ""
        
        explanation = llm_explanation if llm_explanation and llm_explanation != "분석 결과를 해석할 수 없습니다." else pattern_explanation
        
        return {
            "is_phishing": is_phishing,
            "risk_level": risk_level,
            "score": combined_score,
            "keywords": combined_keywords,
            "explanation": explanation,
            "method": "combined"
        }