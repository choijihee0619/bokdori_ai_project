from langchain_community.vectorstores import Chroma
import os
import logging
from dotenv import load_dotenv
import shutil

# 로컬 모듈 임포트
from modules.langchain.embeddings import get_embedding_model

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

class ChromaManager:
    """Chroma 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, config=None):
        """
        ChromaManager 초기화
        
        Args:
            config (dict, optional): 구성 정보. 기본값은 None
        """
        self.config = config or {}
        
        # Chroma 설정
        self.persist_directory = config.get("chroma_persist_directory", 
                                        os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/embeddings"))
        self.collection_name = config.get("chroma_collection_name", 
                                    os.getenv("CHROMA_COLLECTION_NAME", "bokdori_knowledge"))
        
        # 임베딩 모델 로드
        self.embedding_model = get_embedding_model(config)
        
        # 저장 디렉토리 생성
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"ChromaManager 초기화 완료: collection={self.collection_name}, directory={self.persist_directory}")
    
    def get_or_create_db(self):
        """
        기존 Chroma DB를 불러오거나 새로 생성
        
        Returns:
            Chroma: Chroma 벡터 데이터베이스
        """
        try:
            # 기존 DB 로드 시도
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            # 컬렉션이 비어있는지 확인
            if db._collection.count() == 0:
                logger.info("빈 Chroma 컬렉션입니다. 문서를 추가해주세요.")
            else:
                logger.info(f"기존 Chroma DB 로드 완료: {db._collection.count()} 문서")
            
            return db
        
        except Exception as e:
            logger.error(f"Chroma DB 로드 실패: {e}. 새 DB를 생성합니다.")
            
            # 디렉토리 재생성
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 새 DB 생성
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            
            return db
    
    def add_documents(self, documents):
        """
        Chroma DB에 문서 추가
        
        Args:
            documents (list): Document 객체 리스트
            
        Returns:
            Chroma: 업데이트된 Chroma DB
        """
        if not documents:
            logger.warning("추가할 문서가 없습니다.")
            return self.get_or_create_db()
        
        logger.info(f"Chroma DB에 {len(documents)} 문서 추가 중...")
        
        try:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            # 변경사항 저장
            db.persist()
            
            logger.info(f"문서 추가 완료: 총 {db._collection.count()} 문서")
            return db
        
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return self.get_or_create_db()
    
    def search_documents(self, query, k=3):
        """
        쿼리와 관련된 문서 검색
        
        Args:
            query (str): 검색 쿼리
            k (int, optional): 반환할 문서 수. 기본값은 3
            
        Returns:
            list: 검색 결과와 메타데이터
        """
        logger.debug(f"문서 검색: '{query}', k={k}")
        
        db = self.get_or_create_db()
        
        try:
            results = db.similarity_search_with_relevance_scores(query, k=k)
            logger.debug(f"검색 결과: {len(results)} 문서")
            return results
        
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return []
    
    def clear_db(self):
        """
        Chroma DB 초기화 (모든 문서 삭제)
        
        Returns:
            bool: 성공 여부
        """
        logger.warning("Chroma DB 초기화 중...")
        
        try:
            db = self.get_or_create_db()
            db._collection.delete(where={})
            db.persist()
            
            logger.info("Chroma DB 초기화 완료")
            return True
        
        except Exception as e:
            logger.error(f"Chroma DB 초기화 실패: {e}")
            return False