import os
import json
import csv
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class LogExporter:
    """로그 데이터를 다양한 형식으로 내보내는 클래스"""
    
    def __init__(self, base_logs_dir="logs", export_dir="data/exports"):
        """
        LogExporter 초기화
        
        Args:
            base_logs_dir (str): 로그 기본 디렉토리
            export_dir (str): 내보내기 파일 저장 디렉토리
        """
        self.base_logs_dir = base_logs_dir
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
        logger.info(f"LogExporter 초기화 완료: {base_logs_dir} -> {export_dir}")
    
    def load_logs(self, log_type, start_date, end_date):
        """
        특정 기간의 로그 로드
        
        Args:
            log_type (str): 로그 유형 (emotions, conversations, phishing)
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            
        Returns:
            list: 로그 데이터
        """
        # 로그 디렉토리 설정
        log_dir = os.path.join(self.base_logs_dir, log_type)
        if not os.path.exists(log_dir):
            logger.warning(f"로그 디렉토리가 없습니다: {log_dir}")
            return []
        
        # 날짜 파싱
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"날짜 형식 오류: {start_date} ~ {end_date}, 형식은 YYYY-MM-DD여야 합니다.")
            return []
        
        # 로그 수집
        logs = []
        current = start
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            log_file = os.path.join(log_dir, f"{date_str}_{log_type[:-1]}_log.json")  # 단수형으로 변환
            
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
            
            current += timedelta(days=1)
        
        return logs
    
    def export_to_csv(self, log_type, start_date, end_date, output_file=None):
        """
        로그를 CSV 파일로 내보내기
        
        Args:
            log_type (str): 로그 유형 (emotions, conversations, phishing)
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            output_file (str, optional): 출력 파일 경로
            
        Returns:
            str: 내보낸 파일 경로
        """
        # 로그 로드
        logs = self.load_logs(log_type, start_date, end_date)
        
        if not logs:
            logger.warning(f"내보낼 로그가 없습니다: {log_type}, {start_date} ~ {end_date}")
            return None
        
        # 출력 파일 경로 설정
        if output_file is None:
            output_file = os.path.join(
                self.export_dir, 
                f"{log_type}_{start_date}_to_{end_date}.csv"
            )
        
        try:
            # DataFrame으로 변환
            df = pd.json_normalize(logs)
            
            # 타임스탬프를 날짜/시간으로 변환 (있는 경우)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # CSV로 저장
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"CSV 내보내기 완료: {output_file}, {len(logs)}개 로그")
            return output_file
        
        except Exception as e:
            logger.error(f"CSV 내보내기 실패: {e}")
            return None
    
    def export_to_json(self, log_type, start_date, end_date, output_file=None):
        """
        로그를 JSON 파일로 내보내기
        
        Args:
            log_type (str): 로그 유형 (emotions, conversations, phishing)
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            output_file (str, optional): 출력 파일 경로
            
        Returns:
            str: 내보낸 파일 경로
        """
        # 로그 로드
        logs = self.load_logs(log_type, start_date, end_date)
        
        if not logs:
            logger.warning(f"내보낼 로그가 없습니다: {log_type}, {start_date} ~ {end_date}")
            return None
        
        # 출력 파일 경로 설정
        if output_file is None:
            output_file = os.path.join(
                self.export_dir, 
                f"{log_type}_{start_date}_to_{end_date}.json"
            )
        
        try:
            # 메타데이터 추가
            export_data = {
                "export_info": {
                    "log_type": log_type,
                    "period": {
                        "start_date": start_date,
                        "end_date": end_date
                    },
                    "exported_at": datetime.now().isoformat(),
                    "total_logs": len(logs)
                },
                "logs": logs
            }
            
            # JSON으로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON 내보내기 완료: {output_file}, {len(logs)}개 로그")
            return output_file
        
        except Exception as e:
            logger.error(f"JSON 내보내기 실패: {e}")
            return None
    
    def generate_conversation_report(self, start_date, end_date, output_file=None):
        """
        대화 보고서 생성
        
        Args:
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            output_file (str, optional): 출력 파일 경로
            
        Returns:
            str: 보고서 파일 경로
        """
        # 대화 로그 로드
        logs = self.load_logs("conversations", start_date, end_date)
        
        if not logs:
            logger.warning(f"보고서를 생성할 대화 로그가 없습니다: {start_date} ~ {end_date}")
            return None
        
        # 출력 파일 경로 설정
        if output_file is None:
            output_file = os.path.join(
                self.export_dir, 
                f"conversation_report_{start_date}_to_{end_date}.json"
            )
        
        try:
            # 대화 통계 계산
            daily_counts = {}
            keywords = {}
            topics = {}
            
            for log in logs:
                # 날짜 추출
                timestamp = log.get("timestamp", "")
                if timestamp:
                    try:
                        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                        daily_counts[date] = daily_counts.get(date, 0) + 1
                    except ValueError:
                        pass
                
                # 키워드 집계
                for keyword in log.get("keywords", []):
                    keywords[keyword] = keywords.get(keyword, 0) + 1
                
                # 주제 집계
                for topic in log.get("topics", []):
                    topics[topic] = topics.get(topic, 0) + 1
            
            # 보고서 생성
            report = {
                "period": {
                    "start_date": start_date,
                    "end_date": end_date
                },
                "generated_at": datetime.now().isoformat(),
                "total_conversations": len(logs),
                "daily_conversation_counts": daily_counts,
                "top_keywords": dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:20]),
                "top_topics": dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
            # JSON으로 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"대화 보고서 생성 완료: {output_file}")
            return output_file
        
        except Exception as e:
            logger.error(f"대화 보고서 생성 실패: {e}")
            return None