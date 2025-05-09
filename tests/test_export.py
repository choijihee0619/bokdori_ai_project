import os
import sys
from pathlib import Path

# 상위 경로를 모듈 검색 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import json

# 로컬 모듈 임포트
from modules.export.csv_exporter import LogExporter

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_logs():
    """테스트용 로그 생성"""
    print("\n=== 테스트 로그 생성 ===")
    
    # 날짜 설정
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    # 로그 디렉토리 생성
    log_dirs = {
        "conversations": "logs/conversations",
        "emotions": "logs/emotions",
        "phishing": "logs/phishing"
    }
    
    for dir_name, dir_path in log_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    # 테스트 로그 데이터
    test_logs = {
        "conversations": [
            {
                "timestamp": today.isoformat(),
                "user_input": "안녕하세요, 오늘 날씨 어때요?",
                "ai_response": "안녕하세요! 오늘은 맑고 화창한 날씨입니다.",
                "metadata": {
                    "processing_time": 0.45,
                    "use_rag": False
                }
            },
            {
                "timestamp": yesterday.isoformat(),
                "user_input": "복도리 AI 비서란 무엇인가요?",
                "ai_response": "복도리 AI 비서는 LLM, RAG, LangChain 기술을 활용한 지능형 대화 시스템입니다.",
                "metadata": {
                    "processing_time": 0.62,
                    "use_rag": True
                }
            }
        ],
        "emotions": [
            {
                "timestamp": today.isoformat(),
                "text": "오늘은 정말 행복한 하루였어요. 좋은 소식을 들었거든요!",
                "dominant_emotion": "기쁨",
                "emotion_category": "positive",
                "confidence": 0.85,
                "keywords": ["행복", "좋은", "소식"]
            },
            {
                "timestamp": yesterday.isoformat(),
                "text": "조금 피곤하네요. 그래도 괜찮아요.",
                "dominant_emotion": "평온",
                "emotion_category": "neutral",
                "confidence": 0.70,
                "keywords": ["피곤", "괜찮"]
            }
        ],
        "phishing": [
            {
                "timestamp": today.isoformat(),
                "text": "제 계좌번호를 알려드릴까요?",
                "result": {
                    "is_phishing": True,
                    "risk_level": "medium",
                    "score": 0.65,
                    "keywords": ["계좌번호"],
                    "explanation": "계좌번호 관련 발언은 주의가 필요합니다."
                }
            },
            {
                "timestamp": yesterday.isoformat(),
                "text": "오늘 약속 장소가 어디였죠?",
                "result": {
                    "is_phishing": False,
                    "risk_level": "safe",
                    "score": 0.05,
                    "keywords": [],
                    "explanation": "보이스피싱 징후가 감지되지 않았습니다."
                }
            }
        ]
    }
    
    # 로그 파일 저장
    for log_type, logs in test_logs.items():
        today_file = os.path.join(log_dirs[log_type], f"{today.strftime('%Y-%m-%d')}_{log_type[:-1]}_log.json")
        yesterday_file = os.path.join(log_dirs[log_type], f"{yesterday.strftime('%Y-%m-%d')}_{log_type[:-1]}_log.json")
        
        # 오늘 로그
        with open(today_file, 'w', encoding='utf-8') as f:
            json.dump([logs[0]], f, ensure_ascii=False, indent=2)
        
        # 어제 로그
        with open(yesterday_file, 'w', encoding='utf-8') as f:
            json.dump([logs[1]], f, ensure_ascii=False, indent=2)
        
        print(f"{log_type} 테스트 로그 생성 완료: {today_file}, {yesterday_file}")
    
    return True

def test_log_exporter():
    """로그 내보내기 테스트"""
    print("\n=== 로그 내보내기 테스트 ===")
    
    # 로그 내보내기 초기화
    exporter = LogExporter()
    
    # 날짜 설정
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    today_str = today.strftime('%Y-%m-%d')
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    # 로그 로드 테스트
    for log_type in ["emotions", "conversations", "phishing"]:
        logs = exporter.load_logs(log_type, yesterday_str, today_str)
        print(f"{log_type} 로그 로드: {len(logs)}개")
    
# CSV 내보내기 테스트
    csv_file = exporter.export_to_csv("emotions", yesterday_str, today_str)
    print(f"CSV 내보내기 결과: {csv_file}")
    
    # JSON 내보내기 테스트
    json_file = exporter.export_to_json("phishing", yesterday_str, today_str)
    print(f"JSON 내보내기 결과: {json_file}")
    
    # 대화 보고서 생성 테스트
    report_file = exporter.generate_conversation_report(yesterday_str, today_str)
    print(f"대화 보고서 생성 결과: {report_file}")
    
    # 내보내기 파일 확인
    for file_path in [csv_file, json_file, report_file]:
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  파일 확인: {file_path}, 크기: {file_size} 바이트")
    
    return True

def main():
    """메인 테스트 함수"""
    print("데이터 내보내기 시스템 테스트 시작")
    
    # 테스트 로그 생성
    create_test_logs()
    
    # 테스트 실행
    test_results = {
        "로그 내보내기": test_log_exporter()
    }
    
    # 결과 출력
    print("\n=== 테스트 결과 요약 ===")
    for name, result in test_results.items():
        print(f"{name}: {'성공' if result else '실패'}")
    
    # 종합 결과
    all_passed = all(test_results.values())
    print(f"\n전체 테스트 결과: {'성공' if all_passed else '실패'}")

if __name__ == "__main__":
    main()