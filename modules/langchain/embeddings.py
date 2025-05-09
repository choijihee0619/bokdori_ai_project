from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
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
    
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI 임베딩을 사용하려면 OPENAI_API_KEY 환경 변수가 필요합니다.")
        
        return OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
    
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    
    else:
        logger.warning(f"지원되지 않는 임베딩 제공자: {provider}. OpenAI 임베딩을 기본값으로 사용합니다.")
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )