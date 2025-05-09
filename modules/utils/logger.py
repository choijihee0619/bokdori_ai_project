import os
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import time

class Logger:
    """애플리케이션 로깅 관리 클래스"""
    
    def __init__(self, log_level=None):
        """
        Logger 초기화
        
        Args:
            log_level (str, optional): 로그 레벨. 기본값은 환경 변수에서 가져옴
        """
        # 로그 디렉토리 생성
        self.log_dir = "logs"
        
        # 서브 디렉토리 확인
        for subdir in ["conversations", "emotions", "phishing", "alerts"]:
            os.makedirs(os.path.join(self.log_dir, subdir), exist_ok=True)
        
        # 로그 레벨 설정
        log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
        
        # 로그 레벨 매핑
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        
        # 로그 설정
        logging.basicConfig(
            level=level_map.get(log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # 콘솔 출력
                logging.StreamHandler(),
                # 파일 출력 (일반)
                RotatingFileHandler(
                    os.path.join(self.log_dir, "bokdori.log"),
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                ),
                # 파일 출력 (오류)
RotatingFileHandler(
                    os.path.join(self.log_dir, "error.log"),
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    level=logging.ERROR
                )
            ]
        )
        
        self.logger = logging.getLogger("bokdori")
        self.logger.info("로거 초기화 완료")
    
    def log_conversation(self, user_input, ai_response, metadata=None):
        """
        대화 내용 로깅
        
        Args:
            user_input (str): 사용자 입력
            ai_response (str): AI 응답
            metadata (dict, optional): 추가 메타데이터
            
        Returns:
            bool: 성공 여부
        """
        # 로그 디렉토리 확인
        conversation_dir = os.path.join(self.log_dir, "conversations")
        
        # 날짜별 파일명
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(conversation_dir, f"{date_str}_conversation_log.json")
        
        # 로그 엔트리 생성
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": int(time.time()),
            "user_input": user_input,
            "ai_response": ai_response
        }
        
        # 메타데이터 추가
        if metadata:
            log_entry["metadata"] = metadata
        
        try:
            # 파일에 추가
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.logger.debug(f"대화 로깅 완료: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"대화 로깅 실패: {e}")
            return False
    
    def log_emotion(self, text, emotion_result):
        """
        감정 분석 결과 로깅
        
        Args:
            text (str): 분석된 텍스트
            emotion_result (dict): 감정 분석 결과
            
        Returns:
            bool: 성공 여부
        """
        # 로그 디렉토리 확인
        emotion_dir = os.path.join(self.log_dir, "emotions")
        
        # 날짜별 파일명
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(emotion_dir, f"{date_str}_emotion_log.json")
        
        # 로그 엔트리 생성
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": int(time.time()),
            "text": text,
            "dominant_emotion": emotion_result.get("dominant_emotion"),
            "emotion_category": emotion_result.get("emotion_category"),
            "confidence": emotion_result.get("confidence"),
            "keywords": emotion_result.get("keywords", [])
        }
        
        try:
            # 기존 로그 파일 확인
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        try:
                            logs = json.loads(content)
                            if not isinstance(logs, list):
                                logs = [logs]
                        except json.JSONDecodeError:
                            # 한 줄씩 JSON 파싱 시도
                            logs = []
                            for line in content.strip().split('\n'):
                                try:
                                    log = json.loads(line)
                                    logs.append(log)
                                except json.JSONDecodeError:
                                    pass
                    else:
                        logs = []
            else:
                logs = []
            
            # 새 로그 추가
            logs.append(log_entry)
            
            # 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"감정 로깅 완료: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"감정 로깅 실패: {e}")
            
            # 오류 발생 시 단순 추가 방식으로 시도
            try:
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                return True
            except Exception as e2:
                self.logger.error(f"감정 로깅 재시도 실패: {e2}")
                return False
    
    def log_phishing_detection(self, text, result):
        """
        보이스피싱 감지 결과 로깅
        
        Args:
            text (str): 분석된 텍스트
            result (dict): 감지 결과
            
        Returns:
            bool: 성공 여부
        """
        # 로그 디렉토리 확인
        phishing_dir = os.path.join(self.log_dir, "phishing")
        
        # 날짜별 파일명
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(phishing_dir, f"{date_str}_phishing_log.json")
        
        # 로그 엔트리 생성
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "unix_timestamp": int(time.time()),
            "text": text,
            "result": result
        }
        
        try:
            # 파일에 추가
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            # 위험도가 높은 경우 추가 로깅
            if result.get("is_phishing", False) or result.get("risk_level") in ["high", "medium"]:
                self.logger.warning(f"보이스피싱 의심 감지: score={result.get('score')}, level={result.get('risk_level')}")
            
            self.logger.debug(f"보이스피싱 감지 로깅 완료: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"보이스피싱 감지 로깅 실패: {e}")
            return False
    
    def log_weekly_summary(self, log_type, summary_data):
        """
        주간 요약 로깅
        
        Args:
            log_type (str): 로그 유형
            summary_data (dict): 요약 데이터
            
        Returns:
            bool: 성공 여부
        """
        # 로그 디렉토리 확인
        reports_dir = "data/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # 파일명 생성
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = os.path.join(reports_dir, f"weekly_{log_type}_summary_{date_str}.json")
        
        try:
            # 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"주간 {log_type} 요약 저장 완료: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"주간 {log_type} 요약 저장 실패: {e}")
            return False