import os
import logging
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

def get_embedding_model(config=None):
    """
    구성에 따라 적절한 임베딩 모델 반환
    
    Args:
        config (dict, optional): 구성 정보. 기본값은 None
    
    Returns:
        Embeddings: 임베딩 모델 객체
    """
    embedding_config = {}
    provider = "openai"
    model_name = "text-embedding-ada-002"
    
    # 구성 파일이 제공된 경우 설정 로드
    if config and "embedding" in config:
        embedding_config = config["embedding"]
        provider = embedding_config.get("provider", provider)
        model_name = embedding_config.get("model_name", model_name)
    
    logger.info(f"임베딩 모델 로드 중: provider={provider}, model={model_name}")
    
    # API 키 환경 변수 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("환경 변수에서 OPENAI_API_KEY를 찾을 수 없습니다.")
    else:
        key_type = "project" if api_key.startswith("sk-proj-") else "standard"
        logger.info(f"API 키 유형: {key_type}")
    
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("langchain_openai 패키지 사용 중")
            return OpenAIEmbeddings(
                model=model_name
                # API 키는 환경 변수에서 자동으로 로드
            )
        except Exception as e:
            logger.error(f"OpenAI 임베딩 모델 초기화 실패: {e}")
            raise
    
    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)
        except Exception as e:
            logger.error(f"HuggingFace 임베딩 모델 초기화 실패: {e}")
            raise
    
    else:
        logger.warning(f"지원되지 않는 임베딩 제공자: {provider}. OpenAI 임베딩을 기본값으로 사용합니다.")
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_name)
        except Exception as e:
            logger.error(f"기본 임베딩 모델 초기화 실패: {e}")
            raise