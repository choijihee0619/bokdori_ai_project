import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class EmotionTrendMonitor:
    """감정 추세를 모니터링하는 클래스"""
    
    def __init__(self, logs_dir="logs/emotions"):
        """
        EmotionTrendMonitor 초기화
        
        Args:
            logs_dir (str): 감정 로그 디렉토리
        """
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)
        logger.info(f"EmotionTrendMonitor 초기화 완료: {logs_dir}")
    
    def load_emotion_logs(self, days=7):
        """
        최근 감정 로그 로드
        
        Args:
            days (int): 로드할 일수
            
        Returns:
            list: 감정 로그 리스트
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logs = []
        
        # 날짜별 파일 검사
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            log_file = os.path.join(self.logs_dir, f"{date_str}_emotion_log.json")
            
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        day_logs = json.load(f)
                    
                    # 리스트가 아니면 리스트로 변환
                    if not isinstance(day_logs, list):
                        day_logs = [day_logs]
                    
                    logs.extend(day_logs)
                except Exception as e:
                    logger.error(f"로그 파일 로드 실패: {log_file}, {e}")
            
            current_date += timedelta(days=1)
        
        logs.sort(key=lambda x: x.get('timestamp', ''))
        return logs
    
    def calculate_daily_emotions(self, logs):
        """
        일별 감정 통계 계산
        
        Args:
            logs (list): 감정 로그 리스트
            
        Returns:
            dict: 일별 감정 통계
        """
        if not logs:
            return {}
        
        daily_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        
        for log in logs:
            # 타임스탬프가 없으면 건너뛰기
            if "timestamp" not in log:
                continue
            
            try:
                # ISO 형식 타임스탬프 파싱
                timestamp = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
                date_str = timestamp.strftime('%Y-%m-%d')
                
                # 감정 카테고리 집계
                category = log.get("emotion_category", "neutral")
                daily_stats[date_str][category] += 1
                daily_stats[date_str]["total"] += 1
            except Exception as e:
                logger.error(f"로그 처리 중 오류: {e}, 로그: {log}")
        
        # 비율 계산
        result = {}
        for date, stats in daily_stats.items():
            total = stats["total"]
            if total > 0:
                result[date] = {
                    "positive_ratio": stats["positive"] / total,
                    "negative_ratio": stats["negative"] / total,
                    "neutral_ratio": stats["neutral"] / total,
                    "dominant_emotion": max(["positive", "negative", "neutral"], 
                                        key=lambda x: stats[x])
                }
        
        return result
    
    def detect_depression_risk(self, days=7, threshold=0.6):
        """
        우울증 위험 감지 (지속적인 부정 감정)
        
        Args:
            days (int): 확인할 일수
            threshold (float): 부정 감정 비율 임계값
            
        Returns:
            bool: 위험 감지 여부
        """
        logs = self.load_emotion_logs(days)
        daily_stats = self.calculate_daily_emotions(logs)
        
        # 최근 N일 동안의 데이터만 분석
        dates = sorted(daily_stats.keys())[-days:]
        
        if len(dates) < days:
            # 데이터가 충분하지 않으면 위험 없음으로 간주
            return False
        
        # 부정 감정이 임계값을 넘는 날 수
        high_negative_days = 0
        
        for date in dates:
            if date in daily_stats and daily_stats[date]["negative_ratio"] >= threshold:
                high_negative_days += 1
        
        # 대부분의 날(70% 이상)에서 부정 감정이 높으면 위험으로 간주
        return high_negative_days >= (days * 0.7)
    
    def generate_weekly_report(self):
        """
        주간 감정 보고서 생성
        
        Returns:
            dict: 주간 보고서 데이터
        """
        # 최근 7일 로그 로드
        logs = self.load_emotion_logs(7)
        daily_stats = self.calculate_daily_emotions(logs)
        
        # 감정 키워드 빈도 분석
        keyword_counts = defaultdict(int)
        for log in logs:
            for keyword in log.get("keywords", []):
                keyword_counts[keyword] += 1
        
        # 상위 키워드
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 주간 통계 계산
        positive_ratio = sum(stats["positive_ratio"] for stats in daily_stats.values()) / len(daily_stats) if daily_stats else 0
        negative_ratio = sum(stats["negative_ratio"] for stats in daily_stats.values()) / len(daily_stats) if daily_stats else 0
        neutral_ratio = sum(stats["neutral_ratio"] for stats in daily_stats.values()) / len(daily_stats) if daily_stats else 0
        
        # 보고서 생성
        report = {
            "period": {
                "start": (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                "end": datetime.now().strftime('%Y-%m-%d')
            },
            "generated_at": datetime.now().isoformat(),
            "overall_stats": {
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "neutral_ratio": neutral_ratio,
                "dominant_emotion": max(["positive", "negative", "neutral"], 
                                    key=lambda x: locals()[f"{x}_ratio"])
            },
            "daily_stats": daily_stats,
            "top_keywords": dict(top_keywords),
            "depression_risk": self.detect_depression_risk()
        }
        
        return report
    
    def save_weekly_report(self, report=None):
        """
        주간 보고서 저장
        
        Args:
            report (dict, optional): 저장할 보고서. 없으면 생성
            
        Returns:
            str: 저장된 파일 경로
        """
        if report is None:
            report = self.generate_weekly_report()
        
        # 저장 디렉토리 생성
        reports_dir = "data/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # 파일명 생성 (주 마지막 날짜 기준)
        end_date = report["period"]["end"]
        file_path = os.path.join(reports_dir, f"weekly_emotion_report_{end_date}.json")
        
        # 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"주간 감정 보고서 저장 완료: {file_path}")
        return file_path