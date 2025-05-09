import os
import sys
from pathlib import Path

# 상위 경로를 모듈 검색 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import logging

# 로컬 모듈 임포트
from modules.emotion.analyzer import EmotionAnalyzer
from modules.emotion.trend_monitor import EmotionTrendMonitor
from modules.emotion.alert_manager import AlertManager
from modules.utils.helpers import load_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_emotion_analyzer():
    """감정 분석기 테스트"""
    print("\n=== 감정 분석기 테스트 ===")
    
    analyzer = EmotionAnalyzer()
    
    # 다양한 감정 텍스트 테스트
    test_texts = [
        "오늘은 정말 행복한 하루였어요. 좋은 소식을 들었거든요!",
        "너무 슬프고 우울해요. 마음이 아파서 울고 싶어요.",
        "정말 화가 나고 짜증나요. 이런 일이 있다니 믿기지 않아요.",
        "걱정이 너무 많아서 잠을 잘 수가 없어요. 불안해요.",
        "그냥 평범한 하루였어요. 특별한 일은 없었어요."
    ]
    
    for text in test_texts:
        print(f"\n텍스트: '{text}'")
        result = analyzer.analyze_text(text)
        print(f"주요 감정: {result['dominant_emotion']}")
        print(f"감정 카테고리: {result['emotion_category']}")
        print(f"감정 점수: {result['emotion_scores']}")
        print(f"신뢰도: {result['confidence']:.2f}")
        print(f"키워드: {', '.join(result.get('keywords', []))}")
    
    # 대화 분석 테스트
    conversation = [
        "안녕하세요! 오늘 기분이 어떠세요?",  # AI
        "안녕하세요. 오늘은 좋은 일이 있어서 기분이 좋아요.",  # 사용자
        "다행이네요! 어떤 좋은 일이 있었나요?",  # AI
        "승진 소식을 들었어요. 정말 기쁘고 행복해요.",  # 사용자
        "축하드립니다! 정말 기쁜 소식이네요.",  # AI
        "감사합니다. 열심히 노력한 보람이 있네요."  # 사용자
    ]
    
    print("\n대화 분석 테스트:")
    conversation_result = analyzer.analyze_conversation(conversation)
    print(f"전체 감정: {conversation_result['overall_emotion']}")
    print(f"감정 추세: {conversation_result['emotion_trend']}")
    print(f"감정 분포: {conversation_result['emotion_distribution']}")
    
    return True

def test_trend_monitor():
    """감정 추세 모니터링 테스트"""
    print("\n=== 감정 추세 모니터링 테스트 ===")
    
    # 테스트를 위한 감정 로그 생성
    log_dir = "logs/emotions"
    os.makedirs(log_dir, exist_ok=True)
    
    # 오늘 날짜로 테스트 로그 생성
    from datetime import datetime, timedelta
    import json
    
    today = datetime.now()
    file_path = os.path.join(log_dir, f"{today.strftime('%Y-%m-%d')}_emotion_log.json")
    
    # 테스트 로그 데이터
    test_logs = [
        {
            "timestamp": (today - timedelta(hours=6)).isoformat(),
            "dominant_emotion": "기쁨",
            "emotion_category": "positive",
            "confidence": 0.8,
            "keywords": ["행복", "기쁘", "좋아"]
        },
        {
            "timestamp": (today - timedelta(hours=3)).isoformat(),
            "dominant_emotion": "만족",
            "emotion_category": "positive",
            "confidence": 0.7,
            "keywords": ["만족", "좋아", "적당"]
        }
    ]
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(test_logs, f, ensure_ascii=False, indent=2)
    
    # 트렌드 모니터 초기화
    monitor = EmotionTrendMonitor()
    
    # 로그 로드 테스트
    logs = monitor.load_emotion_logs(days=1)
    print(f"로드된 로그 수: {len(logs)}")
    
    # 일별 감정 통계 테스트
    daily_stats = monitor.calculate_daily_emotions(logs)
    print("일별 감정 통계:")
    for date, stats in daily_stats.items():
        print(f"  {date}: 긍정={stats['positive_ratio']:.2f}, 부정={stats['negative_ratio']:.2f}, 중립={stats['neutral_ratio']:.2f}")
    
    # 우울증 위험 감지 테스트
    depression_risk = monitor.detect_depression_risk(days=1)
    print(f"우울증 위험 감지: {depression_risk}")
    
    # 주간 보고서 생성 테스트
    report = monitor.generate_weekly_report()
    print("주간 보고서 생성:")
    print(f"  기간: {report['period']['start']} ~ {report['period']['end']}")
    print(f"  주요 감정: {report['overall_stats']['dominant_emotion']}")
    print(f"  감정 비율: 긍정={report['overall_stats']['positive_ratio']:.2f}, 부정={report['overall_stats']['negative_ratio']:.2f}")
    print(f"  우울증 위험: {report['depression_risk']}")
    
    return True

def test_alert_manager():
    """알림 관리자 테스트"""
    print("\n=== 알림 관리자 테스트 ===")
    
    # 알림 관리자 초기화
    alert_manager = AlertManager()
    
    # 모든 알림 확인
    alerts = alert_manager.check_all_alerts()
    
    print(f"생성된 알림 수: {len(alerts)}")
    for alert in alerts:
        print(f"  유형: {alert['type']}")
        print(f"  심각도: {alert['severity']}")
        print(f"  메시지: {alert['message']}")
    
    return True

def main():
    """메인 테스트 함수"""
    print("감정 분석 시스템 테스트 시작")
    
    # 테스트 실행
    test_results = {
        "감정 분석기": test_emotion_analyzer(),
        "감정 추세 모니터": test_trend_monitor(),
        "알림 관리자": test_alert_manager()
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