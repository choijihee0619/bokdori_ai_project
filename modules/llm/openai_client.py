import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import logging

# .env 파일 로드
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIClient:
    """OpenAI API와 상호작용하는 클라이언트 클래스"""
    
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """
        OpenAI API 클라이언트 초기화
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. .env 파일이나 생성자에 제공해주세요.")
        
        # API 키 확인 로그
        key_type = "project" if self.api_key.startswith("sk-proj-") else "personal"
        logger.info(f"OpenAI API 키 유형: {key_type}")
        
        # 최신 OpenAI 클라이언트 사용
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"OpenAI 클라이언트 초기화 완료: 모델={self.model_name}")
    
    def generate_text(self, prompt, temperature=0.7, max_tokens=1000):
        """
        텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트
            temperature (float, optional): 생성 다양성 (0-1). 기본값은 0.7
            max_tokens (int, optional): 최대 토큰 수. 기본값은 1000
            
        Returns:
            str: 생성된 텍스트
        """
        try:
            logger.debug(f"텍스트 생성 요청: {prompt[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.debug("텍스트 생성 성공")
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"텍스트 생성 실패: {e}")
            return f"오류 발생: {e}"
    
    def generate_stream(self, prompt, temperature=0.7, max_tokens=1000):
        """
        스트리밍 방식으로 텍스트 생성 (생성형 제너레이터)
        
        Args:
            prompt (str): 입력 프롬프트
            temperature (float, optional): 생성 다양성 (0-1). 기본값은 0.7
            max_tokens (int, optional): 최대 토큰 수. 기본값은 1000
            
        Yields:
            str: 생성된 텍스트 청크
        """
        try:
            logger.debug(f"스트리밍 텍스트 생성 요청: {prompt[:50]}...")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in response:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            logger.debug("스트리밍 텍스트 생성 완료")
        
        except Exception as e:
            logger.error(f"스트리밍 텍스트 생성 실패: {e}")
            yield f"오류 발생: {e}"
    
    def create_embedding(self, text, model_name="text-embedding-ada-002"):
        """
        텍스트 임베딩 생성
        
        Args:
            text (str): 임베딩할 텍스트
            model_name (str, optional): 임베딩 모델명. 기본값은 'text-embedding-ada-002'
            
        Returns:
            list: 임베딩 벡터
        """
        try:
            logger.debug(f"임베딩 생성 요청: {text[:50]}...")
            
            response = self.client.embeddings.create(
                model=model_name,
                input=text
            )
            
            logger.debug("임베딩 생성 성공")
            return response.data[0].embedding
        
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            raise ValueError(f"임베딩 생성 중 오류 발생: {e}")