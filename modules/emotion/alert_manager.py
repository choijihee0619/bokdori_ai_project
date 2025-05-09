import os
import json
import logging
from datetime import datetime, timedelta

# 로컬 모듈 임포트
from modules.emotion.trend_monitor import EmotionTrendMonitor

logger = logging.getLogger(__name__)

class AlertManager:
    """감정 기반 알림을 관리하는 클래스"""
    
    def __init__(self, alerts_dir="logs/alerts"):
        """
        AlertManager 초기화
        
        Args:
            alerts_dir (str): 알림 로그 디렉토리
        """
        self.alerts_dir = alerts_dir
        self.emotion_monitor = EmotionTrendMonitor()
        
        # 알림 로그 디렉토리 생성
        os.makedirs(alerts_dir, exist_ok=True)
        
        logger.info(f"AlertManager 초기화 완료: {alerts_dir}")
    
    def check_depression_alert(self):
        """
        우울감 지속 알림 확인
        
        Returns:
            dict: 알림 정보 (필요 시)
        """
        # 우울증 위험 감지
        is_at_risk = self.emotion_monitor.detect_depression_risk(days=7, threshold=0.6)
        
        if not is_at_risk:
            logger.debug("우울증 위험이 감지되지 않았습니다.")
            return None
        
        # 최근 알림 확인 (중복 방지)
        last_alert = self._get_last_alert("depression")
        
        # 최근 3일 이내에 알림을 보낸 경우 중복 방지
        if last_alert and (datetime.now() - datetime.fromisoformat(last_alert["timestamp"])).days < 3:
            logger.info(f"최근에 우울증 알림이 이미 전송됨: {last_alert['timestamp']}")
            return None
        
        # 알림 생성
        alert = {
            "type": "depression",
            "timestamp": datetime.now().isoformat(),
            "severity": "warning",
            "message": "지난 7일 동안 지속적인 부정적 감정이 감지되었습니다. 사용자의 상태를 확인해 주세요.",
            "details": {
                "detection_threshold": 0.6,
                "detection_period": 7
            }
        }
        
        # 알림 저장
        self._save_alert(alert)
        
        logger.warning("우울증 위험 알림이 생성되었습니다.")
        return alert
    
    def check_emotion_change_alert(self):
        """
        감정 급변 알림 확인
        
        Returns:
            dict: 알림 정보 (필요 시)
        """
        # 최근 3일 로그 로드
        logs = self.emotion_monitor.load_emotion_logs(days=3)
        
        if len(logs) < 2:
            # 로그가 충분하지 않으면 패스
            return None
        
        # 로그를 시간순으로 정렬
        logs.sort(key=lambda x: x.get('timestamp', ''))
        
        # 최근 두 개의 로그 비교
        latest = logs[-1]
        previous = logs[-2]
        
        # 감정 카테고리 확인
        latest_category = latest.get("emotion_category", "neutral")
        previous_category = previous.get("emotion_category", "neutral")
        
        # 긍정 -> 부정 또는 부정 -> 긍정으로 급변한 경우
        significant_change = (
            (latest_category == "negative" and previous_category == "positive") or
            (latest_category == "positive" and previous_category == "negative")
        )
        
        if not significant_change:
            return None
        
        # 최근 알림 확인 (중복 방지)
        last_alert = self._get_last_alert("emotion_change")
        
        # 최근 1일 이내에 알림을 보낸 경우 중복 방지
        if last_alert and (datetime.now() - datetime.fromisoformat(last_alert["timestamp"])).days < 1:
            return None
        
        # 알림 생성
        change_type = f"{previous_category}_to_{latest_category}"
        message = f"사용자의 감정 상태가 {previous_category}에서 {latest_category}로 급격히 변화했습니다."
        
        alert = {
            "type": "emotion_change",
            "timestamp": datetime.now().isoformat(),
            "severity": "info",
            "message": message,
            "details": {
                "change_type": change_type,
                "from_emotion": previous_category,
                "to_emotion": latest_category,
                "from_timestamp": previous.get("timestamp"),
                "to_timestamp": latest.get("timestamp")
            }
        }
        
        # 알림 저장
        self._save_alert(alert)
        
        logger.info(f"감정 변화 알림이 생성되었습니다: {change_type}")
        return alert
    
    def _get_last_alert(self, alert_type):
        """
        특정 유형의 최근 알림 조회
        
        Args:
            alert_type (str): 알림 유형
            
        Returns:
            dict: 최근 알림 정보 또는 None
        """
        # 최근 일주일 알림 로그 확인
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        alerts = []
        
        # 날짜별 파일 검사
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            log_file = os.path.join(self.alerts_dir, f"{date_str}_alerts.json")
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        day_alerts = json.load(f)
                    
                    # 리스트가 아니면 리스트로 변환
                    if not isinstance(day_alerts, list):
                        day_alerts = [day_alerts]
                    
                    # 특정 유형의 알림만 필터링
                    filtered_alerts = [a for a in day_alerts if a.get("type") == alert_type]
                    alerts.extend(filtered_alerts)
                except Exception as e:
                    logger.error(f"알림 로그 파일 로드 실패: {log_file}, {e}")
            
            current_date += timedelta(days=1)
        
        # 시간순 정렬
        alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # 가장 최근 알림 반환
        return alerts[0] if alerts else None
    
    def _save_alert(self, alert):
        """
        알림 저장
        
        Args:
            alert (dict): 알림 정보
            
        Returns:
            bool: 성공 여부
        """
        # 날짜 추출
        try:
            timestamp = datetime.fromisoformat(alert["timestamp"].replace('Z', '+00:00'))
            date_str = timestamp.strftime('%Y-%m-%d')
        except ValueError:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 파일 경로
        log_file = os.path.join(self.alerts_dir, f"{date_str}_alerts.json")
        
        try:
            # 기존 파일 확인
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    try:
                        alerts = json.load(f)
                        # 리스트가 아니면 리스트로 변환
                        if not isinstance(alerts, list):
                            alerts = [alerts]
                    except json.JSONDecodeError:
                        alerts = []
            else:
                alerts = []
            
            # 알림 추가
            alerts.append(alert)
            
            # 저장
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(alerts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"알림 저장 완료: {log_file}")
            return True
        
        except Exception as e:
            logger.error(f"알림 저장 실패: {e}")
            return False
    
    def check_all_alerts(self):
        """
        모든 알림 유형 확인
        
        Returns:
            list: 생성된 알림 목록
        """
        alerts = []
        
        # 우울감 지속 알림
        depression_alert = self.check_depression_alert()
        if depression_alert:
            alerts.append(depression_alert)
        
        # 감정 급변 알림
        emotion_change_alert = self.check_emotion_change_alert()
        if emotion_change_alert:
            alerts.append(emotion_change_alert)
        
        return alerts