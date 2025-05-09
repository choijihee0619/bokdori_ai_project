# 복도리 AI 비서 프로젝트

'복도리 AI 비서'는 LLM, RAG, LangChain 기술을 활용한 지능형 대화 시스템입니다. 음성 인식을 통한 사용자 입력을 처리하고, 문서 기반 지식을 활용하여 정확한 응답을 제공하며, 보이스피싱 위험과 감정 변화를 감지하는 기능을 제공합니다.

## 주요 기능

- **대화 처리**: OpenAI LLM을 활용한 자연스러운 대화 처리
- **문서 기반 응답(RAG)**: 사용자 문서를 기반으로 정확한 정보 제공
- **보이스피싱 감지**: 사용자 입력에서 보이스피싱 의심 징후 감지 및 경고
- **감정 분석**: 사용자의 감정 상태 분석 및 지속적인 모니터링
- **감정 알림**: 우울감 지속 및 감정 급변 시 알림 생성
- **키워드 추출**: 대화에서 중요 키워드 추출 및 하이라이팅
- **데이터 내보내기**: 로그 및 분석 결과를 CSV/JSON으로 내보내기
- **음성 인식/합성**: (다른 팀원 개발) STT/TTS 기능을 통한 음성 인터페이스
- **데이터 저장**: (다른 팀원 개발) MongoDB를 활용한 대화 및 정보 저장

## 새로 추가된 기능

- **감정 분석 시스템**: 사용자 대화에서 감정 상태 분석
- **감정 추이 모니터링**: 장기적인 감정 변화 추적 및 위험 감지
- **대화 요약 및 분석**: 대화 내용 요약 및 중요 키워드 추출
- **데이터 내보내기 기능**: CSV/JSON 형식으로 로그 및 분석 결과 내보내기
- **주간 보고서 자동 생성**: 감정 추이 및 대화 패턴 분석 보고서
- **알림 관리 시스템**: 다양한 조건에 따른 알림 생성 및 관리

## 시스템 구조

bokdori_ai_project/
├── .env                 # 환경 변수 (API 키 등)
├── main.py              # 메인 실행 파일
├── config/              # 설정 파일
│   ├── config.json      # 기본 설정
│   ├── keywords.json    # 키워드 설정
│   ├── phishing_patterns.json # 보이스피싱 패턴
│   └── emotion_patterns.json  # 감정 분석용 패턴
├── modules/             # 모듈별 코드
│   ├── llm/             # LLM 관련
│   ├── langchain/       # LangChain 관련
│   ├── rag/             # RAG 관련
│   ├── phishing/        # 보이스피싱 감지
│   ├── emotion/         # 감정 분석
│   ├── export/          # 데이터 내보내기
│   └── utils/           # 유틸리티
├── logs/                # 로그 파일 저장
│   ├── conversations/   # 대화 로그
│   ├── emotions/        # 감정 로그
│   ├── phishing/        # 보이스피싱 로그
│   └── alerts/          # 알림 로그
├── data/                # 데이터 파일
│   ├── documents/       # 원본 문서
│   ├── embeddings/      # 벡터 임베딩 저장
│   ├── exports/         # 내보내기 파일
│   └── reports/         # 자동 생성 보고서

## 설치 및 실행

### 필수 요구사항

- Python 3.8 이상
- OpenAI API 키
- 충분한 저장 공간 (문서 및 임베딩 저장용)

### 설치 방법

1. 저장소 클론: 깃허브
2. 필요한 패키지 설치:pip install -r requirements.txt
3. 환경 변수 설정 (.env 파일 생성):
OPENAI_API_KEY=your_api_key_here
CHROMA_PERSIST_DIRECTORY=./data/embeddings
CHROMA_COLLECTION_NAME=bokdori_knowledge

### 실행 방법

1. 대화형 모드 실행:python main.py
2. 문서 추가:python main.py --mode add_documents --files path/to/file1.pdf path/to/file2.txt
또는 python main.py --mode add_documents --dir path/to/documents/

## 모듈 설명

### LLM 모듈
OpenAI API와 통신하여 텍스트 생성 및 임베딩을 처리합니다.

### LangChain 모듈
대화 체인, RAG 체인, 프롬프트 템플릿 등을 관리합니다.

### RAG 모듈
문서 로딩, 분할, 벡터 저장소 관리, 검색 등을 담당합니다.

### 보이스피싱 감지 모듈
텍스트에서 보이스피싱 의심 징후를 감지하고 위험도를 평가합니다.

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.