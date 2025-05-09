import os
import json
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def load_config():
    """
    설정 파일 로드
    
    Returns:
        dict: 설정 정보
    """
    config_path = "config/config.json"
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"설정 파일 로드 완료: {config_path}")
            return config
        else:
            logger.warning(f"설정 파일을 찾을 수 없음: {config_path}. 기본 설정을 사용합니다.")
            return {}
    
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}. 기본 설정을 사용합니다.")
        return {}

def save_config(config):
    """
    설정 파일 저장
    
    Args:
        config (dict): 저장할 설정 정보
        
    Returns:
        bool: 성공 여부
    """
    config_path = "config/config.json"
    
    try:
        # 디렉토리 확인
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 파일에 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"설정 파일 저장 완료: {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"설정 파일 저장 실패: {e}")
        return False

def create_backup(original_file):
    """
    파일 백업 생성
    
    Args:
        original_file (str): 백업할 파일 경로
        
    Returns:
        str: 백업 파일 경로 또는 None (실패 시)
    """
    if not os.path.exists(original_file):
        logger.error(f"백업할 파일이 없음: {original_file}")
        return None
    
    try:
        # 백업 디렉토리 확인
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        
        # 파일명 분리
        filename = os.path.basename(original_file)
        base, ext = os.path.splitext(filename)
        
        # 백업 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{base}_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # 파일 복사
        with open(original_file, 'rb') as src:
            with open(backup_path, 'wb') as dst:
                dst.write(src.read())
        
        logger.info(f"백업 생성 완료: {backup_path}")
        return backup_path
    
    except Exception as e:
        logger.error(f"백업 생성 실패: {e}")
        return None

def format_time(seconds):
    """
    초 단위 시간을 사람이 읽기 좋은 형식으로 변환
    
    Args:
        seconds (float): 초 단위 시간
        
    Returns:
        str: 형식화된 시간 문자열
    """
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}시간"