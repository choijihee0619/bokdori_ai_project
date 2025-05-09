from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
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
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 대화 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        logger.info(f"ChainManager 초기화 완료: model={model_name}, temp={temperature}")
    
    def get_conversation_chain(self):
        """
        기본 대화 체인 생성
        
        Returns:
            LLMChain: 대화 체인
        """
        logger.debug("대화 체인 생성")
        
        prompt = get_conversation_prompt()
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=self.config.get("debug", False)
        )
        
        return chain
    
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
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=self.config.get("debug", False)
        )
        
        return chain
    
    def get_phishing_detection_chain(self):
        """
        보이스피싱 감지 체인 생성
        
        Returns:
            LLMChain: 보이스피싱 감지 체인
        """
        logger.debug("보이스피싱 감지 체인 생성")
        
        prompt = get_phishing_detection_prompt()
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=self.config.get("debug", False)
        )
        
        return chain
    
    def clear_memory(self):
        """대화 기록 초기화"""
        logger.debug("대화 기록 초기화")
        self.memory.clear()