# metadata_manager.py
from flask import current_app
from openai import OpenAI
from datetime import datetime
import json
from pytz import timezone
from dotenv import load_dotenv
import logging

load_dotenv()
client = OpenAI()
KST = timezone('Asia/Seoul')

logger = logging.getLogger(__name__)

class MetadataManager:
    def __init__(self, user_id: int, message_model=None):
        self.user_id = user_id
        self.Message = message_model
        self.client = OpenAI()
        
        # 이전 메타데이터 로드
        self.metadata = self.load_previous_metadata()

    def load_previous_metadata(self):
        """이전 대화의 메타데이터 로드"""
        try:
            if self.Message is None:
                return self.initialize_metadata()
                
            last_message = self.Message.query.filter_by(
                user_id=self.user_id
            ).order_by(
                self.Message.timestamp.desc()
            ).first()
            
            if last_message:
                metadata = last_message.get_metadata()
                return metadata or self.initialize_metadata()
            
            return self.initialize_metadata()
            
        except Exception as e:
            print(f"이전 메타데이터 로드 중 오류: {str(e)}")
            return self.initialize_metadata()

    def initialize_metadata(self):
        """새로운 메타데이터 초기화"""
        return {
            'user_id': self.user_id,
            'topics': [],
            'entities': [],
            'sentiment': None,
            'preferences': {},
            'timestamp': datetime.now(KST).isoformat(),
            'interaction_count': 0,
            'conversation_context': {
                'recent_topics': [],
                'mentioned_items': []
            }
        }

    def get_interaction_count(self):
        """상호작용 횟수 조회"""
        try:
            if self.Message is None:
                logger.warning("Message 모델이 주입되지 않았습니다")
                return 0
            return self.Message.query.filter_by(user_id=self.user_id).count()
        except Exception as e:
            logger.error(f"상호작용 횟수 조회 중 오류: {str(e)}")
            return 0

    def extract_metadata_with_ai(self, message: str) -> dict:
        """AI를 사용하여 메시지에서 메타데이터 추출"""
        try:
            print("\n=== AI 메타데이터 추출 시작 ===")
            print(f"요청 메시지: {message}")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """메타데이터 관리자입니다. 
                    사용자의 메시지에서 다음 정보를 추출하여 정확한 JSON 형식으로만 반환하세요:
                    {
                        "topics": ["주제1", "주제2"],
                        "entities": ["대상1", "대상2"],
                        "sentiment": "positive/negative/neutral",
                        "preferences": {
                            "항목": "선호도"
                        },
                        "category": "카테고리명"
                    }
                    
                    예시:
                    입력: "나는 인터스텔라가 정말 좋아. 특히 우주 장면이 인상적이었어"
                    출력: {
                        "topics": ["영화", "우주", "SF"],
                        "entities": ["인터스텔라"],
                        "sentiment": "positive",
                        "preferences": {
                            "인터스텔라": "매우 좋아함",
                            "우주_장면": "인상적"
                        },
                        "category": "entertainment"
                    }"""},
                    {"role": "user", "content": message}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content
            print("\n AI 응답 내용:")
            print(result)
            
            try:
                extracted_data = json.loads(result)
                print("\n추출된 메타데이터:")
                print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
                return extracted_data
            except json.JSONDecodeError as je:
                logger.error(f"JSON 파싱 오류: {str(je)}")
                logger.error(f"파싱 시도한 문자열: {result}")
                # 기본 메타데이터 반환
                return {
                    "topics": [],
                    "entities": [],
                    "sentiment": "neutral",
                    "preferences": {},
                    "category": "unknown"
                }
                
        except Exception as e:
            logger.error(f"메타데이터 추출 중 오류: {str(e)}", exc_info=True)
            return {
                "topics": [],
                "entities": [],
                "sentiment": "neutral",
                "preferences": {},
                "category": "unknown"
            }

    def update_metadata(self, message: str) -> dict:
        try:
            # 가장 최근 메시지의 메타데이터 가져오기
            last_message = self.Message.query.filter_by(
                user_id=self.user_id
            ).order_by(
                self.Message.timestamp.desc()
            ).first()

            # 이전 메타데이터가 있으면 로드
            if last_message:
                previous_metadata = last_message.get_metadata()
                # 기존 정보 유지
                self.metadata.update({
                    'topics': previous_metadata.get('topics', []),
                    'entities': previous_metadata.get('entities', []),
                    'preferences': previous_metadata.get('preferences', {}),
                    'conversation_context': previous_metadata.get('conversation_context', {
                        'recent_topics': [],
                        'mentioned_items': []
                    })
                })
            
            # AI를 통한 새로운 메타데이터 추출
            extracted_metadata = self.extract_metadata_with_ai(message)
            
            # 새로운 정보 추가 (중복 제거)
            self.metadata['topics'] = list(set(self.metadata['topics'] + extracted_metadata.get('topics', [])))
            self.metadata['entities'] = list(set(self.metadata['entities'] + extracted_metadata.get('entities', [])))
            
            # 선호도 정보 업데이트
            self.metadata['preferences'].update(extracted_metadata.get('preferences', {}))
            
            # 기타 메타데이터 업데이트
            self.metadata.update({
                'timestamp': datetime.now(KST).isoformat(),
                'message_length': len(message),
                'sentiment': extracted_metadata.get('sentiment'),
                'category': extracted_metadata.get('category'),
                'interaction_count': self.get_interaction_count()
            })
            
            return self.metadata
        except Exception as e:
            print(f"메타데이터 업데이트 중 오류 발생: {str(e)}")
            return self.metadata

    def get_formatted_metadata(self) -> str:
        """메타데이터를 포맷팅된 문자열로 반환"""
        return json.dumps(self.metadata, indent=2, ensure_ascii=False)