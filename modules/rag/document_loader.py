from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.document_loaders import CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
import json

logger = logging.getLogger(__name__)

def load_document(file_path):
    """
    파일 경로로부터 문서 로드
    
    Args:
        file_path (str): 로드할 파일 경로
        
    Returns:
        list: Document 객체 리스트
    """
    logger.info(f"문서 로드 중: {file_path}")
    
    # 파일 확장자에 따라 적절한 로더 선택
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.csv':
            loader = CSVLoader(file_path)
        elif file_ext == '.json':
            loader = JSONLoader(
                file_path=file_path,
                jq_schema='.', 
                text_content=False
            )
        else:  # 기본: 텍스트 파일로 처리 (.txt, .md 등)
            loader = TextLoader(file_path, encoding='utf-8')
        
        documents = loader.load()
        logger.info(f"문서 로드 완료: {len(documents)} 문서")
        return documents
    
    except Exception as e:
        logger.error(f"문서 로드 실패: {e}")
        return []

def load_directory(directory_path, glob_pattern="**/*.*"):
    """
    디렉토리에서 모든 문서 로드
    
    Args:
        directory_path (str): 문서가 있는 디렉토리 경로
        glob_pattern (str, optional): 로드할 파일 패턴. 기본값은 "**/*.*"
        
    Returns:
        list: Document 객체 리스트
    """
    logger.info(f"디렉토리에서 문서 로드 중: {directory_path}, 패턴: {glob_pattern}")
    
    try:
        loader = DirectoryLoader(
            directory_path, 
            glob=glob_pattern,
            recursive=True
        )
        
        documents = loader.load()
        logger.info(f"디렉토리 로드 완료: {len(documents)} 문서")
        return documents
    
    except Exception as e:
        logger.error(f"디렉토리 로드 실패: {e}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    문서를 청크로 분할
    
    Args:
        documents (list): Document 객체 리스트
        chunk_size (int, optional): 청크 크기. 기본값은 1000
        chunk_overlap (int, optional): 청크 간 중복 크기. 기본값은 200
        
    Returns:
        list: 분할된 Document 객체 리스트
    """
    logger.info(f"문서 분할 중: {len(documents)} 문서, 청크 크기: {chunk_size}, 중복: {chunk_overlap}")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        logger.info(f"문서 분할 완료: {len(chunked_documents)} 청크")
        return chunked_documents
    
    except Exception as e:
        logger.error(f"문서 분할 실패: {e}")
        return documents  # 분할 실패 시 원본 문서 반환