import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
import time

# 모듈 임포트
from modules.llm.openai_client import OpenAIClient
from modules.langchain.chains import ChainManager
from modules.phishing.detector import PhishingDetector
from modules.rag.chroma_client import ChromaManager
from modules.rag.document_loader import load_document, split_documents
from modules.rag.retriever import get_retriever
from modules.utils.helpers import load_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BokdoriDemo:
    """복도리 AI 비서 데모 클래스"""
    
    def __init__(self):
        """초기화"""
        print("복도리 AI 비서 초기화 중...")
        
        # 설정 로드
        self.config = load_config()
        
        # 컴포넌트 초기화
        self.llm_client = OpenAIClient()
        self.chain_manager = ChainManager(self.config)
        self.chroma_manager = ChromaManager(self.config.get("rag", {}))
        self.phishing_detector = PhishingDetector(self.config)
        
        # RAG 검색기 초기화
        try:
            self.retriever = get_retriever(self.config)
            print("RAG 검색기 초기화 완료")
        except Exception as e:
            print(f"RAG 검색기 초기화 실패: {e}")
            self.retriever = None
        
        print("복도리 AI 비서 초기화 완료")
    
    def add_test_document(self):
        """테스트 문서 추가"""
        print("\n=== 테스트 문서 추가 ===")
        
        # 테스트 문서 디렉토리 생성
        import os
        os.makedirs("data/documents/test", exist_ok=True)
        
        # 테스트 문서 생성
        test_file = "data/documents/test/about_bokdori.txt"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("복도리 AI 비서는 OpenAI와 LangChain을 활용한 대화형 AI 시스템입니다.\n")
            f.write("사용자 질문에 응답하고, 문서 기반 정보를 제공하며, 보이스피싱 위험을 감지합니다.\n")
            f.write("주요 기능으로는 대화 처리, RAG(Retrieval Augmented Generation), 보이스피싱 감지가 있습니다.\n")
            f.write("LLM으로는 OpenAI의 GPT 모델을 사용하며, 벡터 데이터베이스로는 Chroma를 활용합니다.\n")
        
        print(f"테스트 문서 생성: {test_file}")
        
        # 문서 로드 및 분할
        documents = load_document(test_file)
        chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)
        
        # Chroma DB에 추가
        print(f"문서 청크 {len(chunks)}개를 Chroma DB에 추가합니다...")
        db = self.chroma_manager.add_documents(chunks)
        
        # 검색기 갱신
        self.retriever = get_retriever(self.config)
        
        print("테스트 문서 추가 완료")
    
    def process_message(self, user_input, use_rag=True):
        """사용자 메시지 처리"""
        if not user_input or len(user_input.strip()) == 0:
            return "메시지가 비어있습니다. 질문이나 대화를 입력해주세요."
        
        start_time = time.time()
        print(f"\n사용자: {user_input}")
        
        # 보이스피싱 감지
        phishing_result = self.phishing_detector.detect_with_patterns(user_input)
        
        # 위험도가 높은 보이스피싱 감지 시
        if phishing_result["risk_level"] in ["high", "medium"]:
            warning = f"⚠️ 주의: 이 대화에서 보이스피싱 의심 징후가 감지되었습니다!\n\n"
            warning += f"위험 수준: {phishing_result['risk_level']}\n"
            warning += f"감지된 키워드: {', '.join(phishing_result['keywords'])}\n\n"
            warning += f"{phishing_result['explanation']}"
            
            print(f"복도리: {warning}")
            return warning
        
        try:
            # RAG 또는 일반 대화 처리
            if use_rag and self.retriever:
                # RAG 체인으로 처리
                chain = self.chain_manager.get_rag_chain(self.retriever)
                result = chain({"question": user_input})
                
                response = result.get("answer", "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")
            else:
                # 일반 대화 체인으로 처리
                chain = self.chain_manager.get_conversation_chain()
                result = chain({"question": user_input})
                
                response = result.get("text", "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")
            
            elapsed_time = time.time() - start_time
            print(f"복도리: {response}")
            print(f"처리 시간: {elapsed_time:.2f}초")
            
            return response
        
        except Exception as e:
            print(f"메시지 처리 중 오류 발생: {e}")
            return f"죄송합니다. 메시지를 처리하는 중에 오류가 발생했습니다: {e}"

def interactive_demo():
    """대화형 데모 실행"""
    print("="*50)
    print("복도리 AI 비서 데모")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("="*50)
    
    # 복도리 데모 초기화
    bokdori = BokdoriDemo()
    
    # 테스트 문서 추가
    bokdori.add_test_document()
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n사용자 > ")
            
            # 종료 명령 확인
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("복도리 AI 비서 데모를 종료합니다.")
                break
            
            # 메시지 처리
            bokdori.process_message(user_input)
        
        except KeyboardInterrupt:
            print("\n복도리 AI 비서 데모를 종료합니다.")
            break
        
        except Exception as e:
            print(f"오류 발생: {e}")

if __name__ == "__main__":
    interactive_demo()