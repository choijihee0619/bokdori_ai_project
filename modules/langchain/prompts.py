from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import logging

logger = logging.getLogger(__name__)

def get_conversation_prompt():
    """
    기본 대화 프롬프트 템플릿 생성
    
    Returns:
        PromptTemplate: 대화 프롬프트 템플릿
    """
    logger.debug("기본 대화 프롬프트 템플릿 생성")
    
    template = """당신은 '복도리'라는 AI 비서입니다. 사용자에게 친절하고 도움이 되는 방식으로 응답해 주세요.
    
특히 다음 사항에 유의하세요:
- 보이스피싱과 같은 금융 사기를 감지하고 경고해야 합니다
- 계좌번호, 비밀번호, OTP 등의 민감한 금융 정보 요청에 주의해야 합니다
- 간결하고 자연스러운 대화체로 응답하세요
- 한국어로 대화합니다

이전 대화 기록:
{chat_history}

사용자: {input}
복도리: """
    
    return PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )

def get_rag_prompt():
    """
    RAG 시스템용 프롬프트 템플릿 생성
    
    Returns:
        ChatPromptTemplate: RAG 프롬프트 템플릿
    """
    logger.debug("RAG 프롬프트 템플릿 생성")
    
    system_template = """당신은 '복도리'라는 AI 비서입니다. 사용자에게 친절하고 도움이 되는 방식으로 응답해 주세요.

다음 정보를 참고하여 사용자의 질문에 답변하세요:

{context}

특징:
- 주어진 정보에 기반하여 정확하게 답변하세요
- 정보가 없는 경우 모른다고 솔직하게 말하세요
- 답변은 간결하고 이해하기 쉽게 작성하세요
- 한국어로 대화합니다"""
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    
    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt

def get_phishing_detection_prompt():
    """
    보이스피싱 감지용 프롬프트 템플릿 생성
    
    Returns:
        PromptTemplate: 보이스피싱 감지 프롬프트 템플릿
    """
    logger.debug("보이스피싱 감지 프롬프트 템플릿 생성")
    
    template = """다음 대화 내용에서 보이스피싱 사기 시도가 있는지 분석해주세요:

대화 내용: {text}

보이스피싱 의심 징후:
1. 금융기관, 경찰, 검찰 등을 사칭
2. 급하게 송금이나 이체를 요청
3. 개인정보(계좌번호, 비밀번호, OTP, 주민등록번호 등) 요구
4. 기존 대출 상환을 위한 신규 대출 유도
5. 정부지원금, 환급금 등을 빙자한 금전 요구

분석 결과를 다음 형식으로 제공해주세요:
- 보이스피싱 확률(0-1 사이 숫자): 의심 정도를 나타내는 확률값
- 위험 수준(안전/주의/경고/위험): 전반적인 위험도
- 의심 키워드: 발견된 의심 키워드 목록
- 설명: 왜 의심되는지 또는 안전한지에 대한 간략한 설명
- 대응 방법: 사용자에게 제공할 적절한 대응 방법"""
    
    return PromptTemplate(
        input_variables=["text"],
        template=template
    )