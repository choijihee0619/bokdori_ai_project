import os
import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 모듈 임포트
from modules.llm.openai_client import OpenAIClient
from modules.langchain.chains import ChainManager
from modules.rag.chroma_client import ChromaManager
from modules.rag.document_loader import load_document, load_directory, split_documents
from modules.rag.retriever import get_retriever
from modules.rag.keyword_extractor import KeywordExtractor
from modules.phishing.detector import PhishingDetector
from modules.emotion.analyzer import EmotionAnalyzer
from modules.emotion.trend_monitor import EmotionTrendMonitor
from modules.emotion.alert_manager import AlertManager
from modules.export.csv_exporter import LogExporter
from modules.utils.logger import Logger
from modules.utils.helpers import load_config, save_config, format_time

# .env 파일 로드
load_dotenv()

# 로거 초기화
app_logger = Logger(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

class BokdoriAI:
    """복도리 AI 비서 메인 클래스"""
    
    def __init__(self):
        """복도리 AI 비서 초기화"""
        logger.info("복도리 AI 비서 초기화 중...")
        
        # 설정 로드
        self.config = load_config()
        
        # 컴포넌트 초기화
        self.llm_client = OpenAIClient(
            model_name=self.config.get("llm", {}).get("model_name", "gpt-3.5-turbo")
        )
        self.chain_manager = ChainManager(self.config)
        self.chroma_manager = ChromaManager(self.config.get("rag", {}))
        self.phishing_detector = PhishingDetector(self.config)
        
        # 새로 추가된 컴포넌트 초기화
        self.emotion_analyzer = EmotionAnalyzer(self.config)
        self.emotion_monitor = EmotionTrendMonitor()
        self.alert_manager = AlertManager()
        self.keyword_extractor = KeywordExtractor()
        self.log_exporter = LogExporter()
        
        # RAG 검색기 초기화
        self.retriever = get_retriever(self.config)
        
        # 대화 기록 초기화
        self.conversation_history = []
        
        logger.info("복도리 AI 비서 초기화 완료")
    
    def process_message(self, user_input, use_rag=True):
        """
        사용자 메시지 처리
        
        Args:
            user_input (str): 사용자 입력
            use_rag (bool, optional): RAG 사용 여부. 기본값은 True
            
        Returns:
            str: AI 응답
        """
        if not user_input or len(user_input.strip()) == 0:
            return "메시지가 비어있습니다. 질문이나 대화를 입력해주세요."
        
        start_time = time.time()
        logger.info(f"사용자 메시지 처리 중: '{user_input[:50]}...'")
        
        # 대화 기록에 추가
        self.conversation_history.append(user_input)
        
        # 보이스피싱 감지
        phishing_result = self.phishing_detector.detect(user_input)
        app_logger.log_phishing_detection(user_input, phishing_result)
        
        # 감정 분석
        emotion_result = self.emotion_analyzer.analyze_text(user_input)
        app_logger.log_emotion(user_input, emotion_result)
        
        # 위험도가 높은 보이스피싱 감지 시
        if phishing_result.get("is_phishing", False):
            level = phishing_result.get("risk_level", "unknown")
            if level in ["high", "medium"]:
                warning = f"⚠️ 주의: 이 대화에서 보이스피싱 의심 징후가 감지되었습니다! ({phishing_result['score']:.2f}점)\n\n"
                warning += f"{phishing_result['explanation']}\n\n"
                warning += "개인정보나 금융정보를 제공하지 마시고, 의심스러운 요청은 해당 기관에 직접 문의하세요."
                
                logger.warning(f"보이스피싱 의심 감지: {phishing_result['score']:.2f}점, {level} 위험")
                
                # 로깅 및 응답
                processing_time = time.time() - start_time
                app_logger.log_conversation(
                    user_input, 
                    warning, 
                    {
                        "phishing_detected": True,
                        "risk_level": level,
                        "processing_time": processing_time
                    }
                )
                
                # 대화 기록에 응답 추가
                self.conversation_history.append(warning)
                
                return warning
        
        try:
            # RAG 또는 일반 대화 처리
            if use_rag and self.retriever:
                # RAG 체인으로 처리
                result = self.chain_manager.get_rag_chain(self.retriever)({
                    "question": user_input
                })
                
                ai_response = result.get("answer", "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")
                
                # 소스 문서 정보 추가 (선택적)
                source_docs = result.get("source_documents", [])
                if source_docs and self.config.get("show_sources", False):
                    ai_response += "\n\n출처:"
                    for i, doc in enumerate(source_docs[:3], 1):
                        source = doc.metadata.get("source", "알 수 없는 출처")
                        ai_response += f"\n{i}. {os.path.basename(source)}"
            else:
                # 일반 대화 체인으로 처리
                result = self.chain_manager.get_conversation_chain()({
                    "question": user_input
                })
                
                ai_response = result.get("text", "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다.")
            
            # 처리 시간 측정
            processing_time = time.time() - start_time
            logger.info(f"메시지 처리 완료: {format_time(processing_time)}")
            
            # 응답 감정 분석 (사용자의 감정에 맞춘 응답 조정 가능)
            emotion_category = emotion_result.get("emotion_category", "neutral")
            emotion_response = ""
            
            # 특정 감정에 대한 추가 응답 (필요 시)
            if emotion_category == "negative" and emotion_result.get("confidence", 0) > 0.7:
                emotion_response = "\n\n혹시 무슨 일이 있으신가요? 도움이 필요하시면 말씀해주세요."
            
            # 최종 응답 구성
            final_response = ai_response + emotion_response
            
            # 대화 및 감정 로깅
            app_logger.log_conversation(
                user_input,
                final_response,
                {
                    "processing_time": processing_time,
                    "use_rag": use_rag,
                    "emotion": emotion_result.get("dominant_emotion"),
                    "emotion_category": emotion_category
                }
            )
            
            # 대화 기록에 응답 추가
            self.conversation_history.append(final_response)
            
            # 알림 확인 (일정 주기로 실행, 여기서는 10번째 메시지마다)
            if len(self.conversation_history) % 10 == 0:
                self._check_alerts()
            
            return final_response
        
        except Exception as e:
            logger.error(f"메시지 처리 중 오류 발생: {e}")
            
            # 오류 응답
            error_response = "죄송합니다. 메시지를 처리하는 중에 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
            
            # 대화 로깅
            processing_time = time.time() - start_time
            app_logger.log_conversation(
                user_input, 
                error_response, 
                {
                    "error": str(e),
                    "processing_time": processing_time
                }
            )
            
            # 대화 기록에 응답 추가
            self.conversation_history.append(error_response)
            
            return error_response
    
    def _check_alerts(self):
        """알림 확인 및 처리"""
        alerts = self.alert_manager.check_all_alerts()
        
        for alert in alerts:
            logger.info(f"알림 발생: {alert['type']} - {alert['severity']}")
            
            # 여기서 실제 알림 전송 로직 구현 가능
            # (예: 보호자에게 이메일/SMS 전송 등)
    
    def add_documents(self, file_paths=None, directory_path=None):
        """
        RAG 시스템에 문서 추가
        
        Args:
            file_paths (list, optional): 파일 경로 목록. 기본값은 None
            directory_path (str, optional): 디렉토리 경로. 기본값은 None
            
        Returns:
            int: 추가된 문서 수
        """
        documents = []
        
        # 파일 로드
        if file_paths:
            for file_path in file_paths:
                if os.path.exists(file_path):
                    logger.info(f"파일 로드 중: {file_path}")
                    docs = load_document(file_path)
                    documents.extend(docs)
                else:
                    logger.warning(f"파일을 찾을 수 없음: {file_path}")
        
        # 디렉토리 로드
        if directory_path:
            if os.path.exists(directory_path) and os.path.isdir(directory_path):
                logger.info(f"디렉토리 로드 중: {directory_path}")
                docs = load_directory(directory_path)
                documents.extend(docs)
            else:
                logger.warning(f"디렉토리를 찾을 수 없음: {directory_path}")
        
        if not documents:
            logger.warning("추가할 문서가 없습니다.")
            return 0
        
        # 청크 크기 및 중복 설정
        rag_config = self.config.get("rag", {})
        chunk_size = rag_config.get("chunk_size", 1000)
        chunk_overlap = rag_config.get("chunk_overlap", 200)
        
        # 문서 분할
        logger.info(f"문서 분할 중: {len(documents)} 문서, 청크 크기: {chunk_size}, 중복: {chunk_overlap}")
        chunked_documents = split_documents(documents, chunk_size, chunk_overlap)
        
        # Chroma DB에 추가
        logger.info(f"Chroma DB에 문서 추가 중: {len(chunked_documents)} 청크")
        db = self.chroma_manager.add_documents(chunked_documents)
        
        # 검색기 갱신
        self.retriever = get_retriever(self.config)
        
        logger.info(f"문서 추가 완료: {len(chunked_documents)} 청크")
        return len(chunked_documents)
    
    def reset_conversation(self):
        """대화 기록 초기화"""
        logger.info("대화 기록 초기화")
        self.conversation_history = []
        self.chain_manager.clear_memory()
        return "대화 기록이 초기화되었습니다."
    
    def clear_knowledge_base(self):
        """지식 베이스 초기화"""
        logger.warning("지식 베이스 초기화")
        result = self.chroma_manager.clear_db()
        
        if result:
            # 검색기 갱신
            self.retriever = get_retriever(self.config)
            return "지식 베이스가 초기화되었습니다."
        else:
            return "지식 베이스 초기화 중 오류가 발생했습니다."
    
    def generate_weekly_reports(self):
        """주간 보고서 생성"""
        logger.info("주간 보고서 생성 중...")
        
        reports = {}
        
        # 감정 보고서
        try:
            emotion_report = self.emotion_monitor.generate_weekly_report()
            report_path = self.emotion_monitor.save_weekly_report(emotion_report)
            reports["emotion"] = report_path
            logger.info(f"감정 보고서 생성 완료: {report_path}")
        except Exception as e:
            logger.error(f"감정 보고서 생성 실패: {e}")
        
        # 대화 보고서
        try:
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            
            report_path = self.log_exporter.generate_conversation_report(
                week_ago.strftime('%Y-%m-%d'),
                today.strftime('%Y-%m-%d')
            )
            reports["conversation"] = report_path
            logger.info(f"대화 보고서 생성 완료: {report_path}")
        except Exception as e:
            logger.error(f"대화 보고서 생성 실패: {e}")
        
        return reports

    def export_logs(self, log_type, start_date, end_date, format="csv"):
        """
        로그 내보내기
        
        Args:
            log_type (str): 로그 유형 (emotions, conversations, phishing)
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            format (str): 내보내기 형식 (csv 또는 json)
            
        Returns:
            str: 내보낸 파일 경로
        """
        logger.info(f"{log_type} 로그 내보내기 중: {start_date} ~ {end_date}, 형식: {format}")
        
        if format.lower() == "csv":
            return self.log_exporter.export_to_csv(log_type, start_date, end_date)
        else:
            return self.log_exporter.export_to_json(log_type, start_date, end_date)


def interactive_mode():
    """대화형 모드 실행"""
    print("="*50)
    print("복도리 AI 비서 - 대화형 모드")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.")
    print("="*50)
    
    # 복도리 AI 초기화
    bokdori = BokdoriAI()
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n사용자 > ")
            
            # 종료 명령 확인
            if user_input.lower() in ["exit", "quit", "종료"]:
                print("복도리 AI 비서를 종료합니다.")
                break
            
            # 특수 명령 처리
            if user_input.lower() == "reset":
                print("복도리 > " + bokdori.reset_conversation())
                continue
            
            if user_input.lower() == "report":
                reports = bokdori.generate_weekly_reports()
                print("복도리 > 주간 보고서가 생성되었습니다:")
                for report_type, path in reports.items():
                    print(f"  - {report_type}: {path}")
                continue
            
            # 메시지 처리
            response = bokdori.process_message(user_input)
            
            # 응답 출력
            print("복도리 > " + response)
        
        except KeyboardInterrupt:
            print("\n복도리 AI 비서를 종료합니다.")
            break
        
        except Exception as e:
            logger.error(f"예상치 못한 오류 발생: {e}")
            print(f"복도리 > 오류가 발생했습니다: {e}")


def add_documents_mode(file_paths=None, directory_path=None):
    """문서 추가 모드 실행"""
    print("="*50)
    print("복도리 AI 비서 - 문서 추가 모드")
    print("="*50)
    
    # 복도리 AI 초기화
    bokdori = BokdoriAI()
    
    # 파일 경로 확인
    if not file_paths and not directory_path:
        print("추가할 파일 또는 디렉토리를 지정해주세요.")
        
        # 대화형으로 파일/디렉토리 경로 받기
        path_type = input("파일 추가(F) 또는 디렉토리 추가(D)? [F/D]: ").strip().upper()
        
        if path_type == "F":
            paths = input("추가할 파일 경로를 입력하세요 (여러 개는 쉼표로 구분): ").strip()
            file_paths = [p.strip() for p in paths.split(",") if p.strip()]
        elif path_type == "D":
            directory_path = input("추가할 디렉토리 경로를 입력하세요: ").strip()
        else:
            print("잘못된 입력입니다. 프로그램을 종료합니다.")
            return
    
    # 문서 추가
    start_time = time.time()
    count = bokdori.add_documents(file_paths, directory_path)
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    print(f"\n{count}개 문서 청크가 추가되었습니다.")
    print(f"처리 시간: {format_time(elapsed_time)}")


def export_logs_mode(log_type=None, start_date=None, end_date=None, format=None):
    """로그 내보내기 모드 실행"""
    print("="*50)
    print("복도리 AI 비서 - 로그 내보내기 모드")
    print("="*50)
    
    # 복도리 AI 초기화
    bokdori = BokdoriAI()
    
    # 매개변수 확인
    if not log_type:
        log_type = input("내보낼 로그 유형을 선택하세요 [emotions/conversations/phishing]: ").strip().lower()
        if log_type not in ["emotions", "conversations", "phishing"]:
            print("잘못된 로그 유형입니다. 프로그램을 종료합니다.")
            return
    
    if not start_date:
        start_date = input("시작일을 입력하세요 (YYYY-MM-DD): ").strip()
        # 날짜 형식 검증 필요
    
    if not end_date:
        end_date = input("종료일을 입력하세요 (YYYY-MM-DD): ").strip()
        # 날짜 형식 검증 필요
    
    if not format:
        format = input("내보내기 형식을 선택하세요 [csv/json]: ").strip().lower()
        if format not in ["csv", "json"]:
            print("잘못된 형식입니다. CSV로 설정합니다.")
            format = "csv"
    
    # 로그 내보내기
    file_path = bokdori.export_logs(log_type, start_date, end_date, format)
    
    if file_path:
        print(f"\n로그가 성공적으로 내보내졌습니다: {file_path}")
    else:
        print("\n로그 내보내기에 실패했습니다.")


def report_mode():
    """보고서 생성 모드 실행"""
    print("="*50)
    print("복도리 AI 비서 - 보고서 생성 모드")
    print("="*50)
    
    # 복도리 AI 초기화
    bokdori = BokdoriAI()
    
    # 보고서 생성
    reports = bokdori.generate_weekly_reports()
    
    # 결과 출력
    if reports:
        print("\n다음 보고서가 생성되었습니다:")
        for report_type, path in reports.items():
            print(f"  - {report_type}: {path}")
    else:
        print("\n보고서 생성에 실패했습니다.")


def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="복도리 AI 비서")
    
    # 모드 선택
    parser.add_argument("--mode", choices=["interactive", "add_documents", "export_logs", "report", "server"], 
                    default="interactive", help="실행 모드 (기본값: interactive)")
    
    # 문서 추가 관련 인자
    parser.add_argument("--files", nargs="+", help="추가할 파일 경로 목록")
    parser.add_argument("--dir", help="추가할 문서가 있는 디렉토리 경로")
    
    # 로그 내보내기 관련 인자
    parser.add_argument("--log-type", choices=["emotions", "conversations", "phishing"], 
                    help="내보낼 로그 유형")
    parser.add_argument("--start-date", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", 
                    help="내보내기 형식 (기본값: csv)")
    
    args = parser.parse_args()
    
    # 모드에 따라 실행
    if args.mode == "interactive":
        interactive_mode()
    elif args.mode == "add_documents":
        add_documents_mode(args.files, args.dir)
    elif args.mode == "export_logs":
        export_logs_mode(args.log_type, args.start_date, args.end_date, args.format)
    elif args.mode == "report":
        report_mode()
    elif args.mode == "server":
        print("서버 모드는 다른 팀원이 구현할 예정입니다.")
    else:
        print(f"알 수 없는 모드: {args.mode}")


if __name__ == "__main__":
    main()