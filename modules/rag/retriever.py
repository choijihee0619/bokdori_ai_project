from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import logging

# 로컬 모듈 임포트
from modules.rag.chroma_client import ChromaManager

logger = logging.getLogger(__name__)

def get_retriever(config=None, llm=None):
    """
    문서 검색기 생성
    
    Args:
        config (dict, optional): 구성 정보. 기본값은 None
        llm (BaseLLM, optional): 압축에 사용할 LLM. 기본값은 None
        
    Returns:
        Retriever: 문서 검색기
    """
    logger.info("검색기 생성 중...")
    
    # RAG 설정
    rag_config = {}
    if config and "rag" in config:
        rag_config = config["rag"]
    
    top_k = rag_config.get("top_k", 3)
    use_compression = rag_config.get("use_compression", False)
    
    # Chroma DB 로드
    chroma_manager = ChromaManager(rag_config)
    db = chroma_manager.get_or_create_db()
    
    # 기본 검색기
    base_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    # LLM 컨텍스트 압축 사용 (선택적)
    if use_compression and llm:
        logger.info("LLM 컨텍스트 압축 활성화")
        
        compressor = LLMChainExtractor.from_llm(llm)
        
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return retriever
    
    logger.info(f"기본 검색기 생성 완료 (top_k={top_k})")
    return base_retriever