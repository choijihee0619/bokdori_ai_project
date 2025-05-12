from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import logging
import json

# 로컬 모듈 임포트
from modules.langchain.prompts import get_conversation_prompt, get_rag_prompt, get_phishing_detection_prompt
from modules.llm.openai_client import OpenAIClient

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

def load_config():
    """설정 파일 로드"""
    config_path = "config/config.json"
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"설정 파일 로드 실패: {e}. 기본 설정을 사용합니다.")
        return {}

class ChainManager:
    """LangChain 체인 관리 클래스"""
    
    def __init__(self, config=None):
        """
        ChainManager 초기화
        
        Args:
            config (dict, optional): 구성 정보. 기본값은 None
        """
        self.config = config or load_config()
        
        # LLM 설정
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model_name", "gpt-3.5-turbo")
        temperature = llm_config.get("temperature", 0.7)
        
        # API 키 로드 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("환경 변수에서 OPENAI_API_KEY를 찾을 수 없습니다.")
        else:
            key_prefix = api_key[:10] if len(api_key) > 10 else "***"
            logger.info(f"API 키 형식 확인: {key_prefix}...")
        
        # LLM 초기화 - 최신 패키지 먼저 시도
        try:
            # 최신 langchain_openai 패키지 사용
            from langchain_openai import ChatOpenAI
            logger.info("langchain_openai 패키지 사용 중")
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature
                # API 키는 환경변수에서 자동으로 로드
            )
        except Exception as e:
            logger.error(f"LLM 초기화 오류: {e}")
            raise ValueError(f"LLM 초기화 실패: {e}")
        
        # 대화 메모리 초기화
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        except Exception as e:
            logger.error(f"메모리 초기화 오류: {e}")
            raise
        
        logger.info(f"ChainManager 초기화 완료: model={model_name}, temp={temperature}")
    
    def get_conversation_chain(self):
        """
        기본 대화 체인 생성
        
        Returns:
            LLMChain: 대화 체인
        """
        logger.debug("대화 체인 생성")
        
        prompt = get_conversation_prompt()
        
        try:
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                memory=self.memory,
                verbose=self.config.get("debug", False)
            )
            return chain
        except Exception as e:
            logger.error(f"대화 체인 생성 실패: {e}")
            raise
    
    def get_rag_chain(self, retriever):
        """
        RAG 체인 생성
        
        Args:
            retriever: 문서 검색기
            
        Returns:
            ConversationalRetrievalChain: RAG 체인
        """
        logger.debug("RAG 체인 생성")
        
        if not retriever:
            raise ValueError("RAG 체인을 생성하려면 retriever가 필요합니다.")
        
        try:
            # 최신 LangChain에 맞게 RAG 체인 구성
            from langchain.chains import create_retrieval_chain
            from langchain.chains.combine_documents import create_stuff_documents_chain
            
            # 문서 체인 생성
            prompt = get_rag_prompt()  # RAG용 프롬프트 가져오기
            document_chain = create_stuff_documents_chain(self.llm, prompt)
            
            # 검색 체인 생성
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            return retrieval_chain
        except Exception as e:
            logger.error(f"RAG 체인 생성 실패: {e}")
            raise
    
    def get_phishing_detection_chain(self):
        """
        보이스피싱 감지 체인 생성
        
        Returns:
            LLMChain: 보이스피싱 감지 체인
        """
        logger.debug("보이스피싱 감지 체인 생성")
        
        prompt = get_phishing_detection_prompt()
        
        try:
            chain = LLMChain(
                llm=self.llm,
                prompt=prompt,
                verbose=self.config.get("debug", False)
            )
            return chain
        except Exception as e:
            logger.error(f"보이스피싱 감지 체인 생성 실패: {e}")
            raise
    
    def clear_memory(self):
        """대화 기록 초기화"""
        logger.debug("대화 기록 초기화")
        try:
            self.memory.clear()
        except Exception as e:
            logger.error(f"메모리 초기화 오류: {e}")