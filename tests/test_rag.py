import os
import sys
from pathlib import Path

# 상위 경로를 모듈 검색 경로에 추가
sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import logging
from modules.rag.chroma_client import ChromaManager
from modules.rag.document_loader import load_document, split_documents
from modules.rag.retriever import get_retriever
from modules.utils.helpers import load_config

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_document_loading():
    """문서 로드 테스트"""
    print("\n=== 문서 로드 테스트 ===")
    
    # 테스트 문서 생성
    os.makedirs("data/documents/test", exist_ok=True)
    test_file = "data/documents/test/test_document.txt"
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("이것은 RAG 시스템 테스트를 위한 문서입니다.\n")
        f.write("복도리 AI 비서는 사용자의 질문에 정확하게 답변하기 위해 문서 기반 검색을 활용합니다.\n")
        f.write("LangChain과 Chroma를 사용하여 효율적인 벡터 검색을 구현했습니다.")
    
    # 문서 로드
    documents = load_document(test_file)
    print(f"로드된 문서: {len(documents)} 개")
    print(f"문서 내용 샘플: {documents[0].page_content[:50]}...")
    
    # 문서 분할
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)
    print(f"분할된 청크: {len(chunks)} 개")
    for i, chunk in enumerate(chunks):
        print(f"청크 {i+1}: {chunk.page_content[:30]}...")
    
    return len(chunks) > 0

def test_chroma_db():
    """Chroma 벡터 저장소 테스트"""
    print("\n=== Chroma 벡터 저장소 테스트 ===")
    
    config = load_config()
    
    # ChromaManager 초기화
    chroma_manager = ChromaManager(config.get("rag", {}))
    
    # 테스트 문서 생성 및 로드
    test_file = "data/documents/test/test_document.txt"
    documents = load_document(test_file)
    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)
    
    # Chroma DB에 문서 추가
    db = chroma_manager.add_documents(chunks)
    count = db._collection.count() if hasattr(db, '_collection') else 0
    print(f"Chroma DB 문서 수: {count}")
    
    # 간단한 쿼리 테스트
    results = chroma_manager.search_documents("복도리 AI 비서는 무엇을 사용하나요?", k=2)
    print(f"검색 결과: {len(results)} 개")
    for i, (doc, score) in enumerate(results):
        print(f"결과 {i+1} (유사도: {score:.4f}): {doc.page_content[:50]}...")
    
    return len(results) > 0

def test_rag_retriever():
    """RAG 검색기 테스트"""
    print("\n=== RAG 검색기 테스트 ===")
    
    config = load_config()
    
    # 검색기 가져오기
    retriever = get_retriever(config)
    
    # 쿼리 테스트
    query = "복도리 AI 비서는 어떤 기술을 사용하나요?"
    results = retriever.get_relevant_documents(query)
    
    print(f"검색 결과: {len(results)} 개")
    for i, doc in enumerate(results):
        print(f"결과 {i+1}: {doc.page_content[:50]}...")
        print(f"   메타데이터: {doc.metadata}")
    
    return len(results) > 0

def main():
    """메인 테스트 함수"""
    print("RAG 시스템 테스트 시작")
    
    # 테스트 실행
    test_results = {
        "문서 로드": test_document_loading(),
        "Chroma DB": test_chroma_db(),
        "RAG 검색기": test_rag_retriever()
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