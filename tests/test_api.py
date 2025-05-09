import sys
from pathlib import Path

# 상위 경로를 모듈 검색 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import logging
import time

from modules.llm.openai_client import OpenAIClient
from modules.langchain.chains import ChainManager
from modules.langchain.prompts import get_conversation_prompt, get_rag_prompt
from modules.phishing.detector import PhishingDetector
from modules.utils.helpers import load_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_openai_client():
    """OpenAI 클라이언트 테스트"""
    print("\n=== OpenAI 클라이언트 테스트 ===")
    
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAIClient()
        
        # 간단한 대화 생성 테스트
        prompt = "안녕하세요, 당신은 누구인가요? 한 문장으로 대답해주세요."
        start_time = time.time()
        response = client.generate_text(prompt, max_tokens=50)
        elapsed_time = time.time() - start_time
        
        print(f"응답: {response}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        
        # 임베딩 테스트
        text = "이것은 임베딩 테스트를 위한 텍스트입니다."
        embedding = client.create_embedding(text)
        
        print(f"임베딩 차원: {len(embedding)}")
        print(f"임베딩 샘플: {embedding[:5]}...")
        
        return True
    except Exception as e:
        print(f"OpenAI 클라이언트 테스트 실패: {e}")
        return False

def test_chain_manager():
    """ChainManager 테스트"""
    print("\n=== ChainManager 테스트 ===")
    
    try:
        config = load_config()
        manager = ChainManager(config)
        
        # 대화 체인 테스트
        conversation_chain = manager.get_conversation_chain()
        
        # 질문 테스트
        question = "인공지능이란 무엇인가요? 간단히 설명해주세요."
        start_time = time.time()
        result = conversation_chain({"question": question})
        elapsed_time = time.time() - start_time
        
        print(f"질문: {question}")
        print(f"응답: {result['text']}")
        print(f"응답 시간: {elapsed_time:.2f}초")
        
        # 메모리 테스트
        follow_up = "그것의 주요 응용 분야는 무엇인가요?"
        result2 = conversation_chain({"question": follow_up})
        
        print(f"후속 질문: {follow_up}")
        print(f"응답: {result2['text']}")
        
        return True
    except Exception as e:
        print(f"ChainManager 테스트 실패: {e}")
        return False

def test_phishing_detector():
    """보이스피싱 감지 테스트"""
    print("\n=== 보이스피싱 감지 테스트 ===")
    
    try:
        config = load_config()
        detector = PhishingDetector(config)
        
        # 정상 텍스트 테스트
        normal_text = "내일 날씨가 어떨까요? 오후에 약속이 있어서요."
        result1 = detector.detect_with_patterns(normal_text)
        
        print(f"정상 텍스트: '{normal_text}'")
        print(f"분석 결과: {result1}")
        
        # 의심 텍스트 테스트
        suspicious_text = "지금 당장 OTP 번호를 알려주세요. 계좌에 문제가 생겼습니다."
        result2 = detector.detect_with_patterns(suspicious_text)
        
        print(f"의심 텍스트: '{suspicious_text}'")
        print(f"분석 결과: {result2}")
        
        return True
    except Exception as e:
        print(f"보이스피싱 감지 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("LLM/LangChain API 테스트 시작")
    
    # 테스트 실행
    test_results = {
        "OpenAI 클라이언트": test_openai_client(),
        "ChainManager": test_chain_manager(),
        "보이스피싱 감지": test_phishing_detector()
    }
    
    # 결과 출력
    print("\n=== 테스트 결과 요약 ===")
    for name, result in test_results.items():
        print(f"{name}: {'성공' if result else '실패'}")
    
    # 종합 결과
    all_passed = all(test_results.values())
    print(f"\n전체 테스트 결과: {'성공' if all_passed else '실패'}")

if __name__ == "__main__":
    main()