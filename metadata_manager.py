# metadata_manager.py
from openai import OpenAI
from datetime import datetime
import json
from pytz import timezone

KST = timezone('Asia/Seoul')

class MetadataManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.client = OpenAI()  # OpenAI 클라이언트 초기화
        self.metadata = {
            'user_id': user_id,
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

    def extract_metadata_with_ai(self, message: str) -> dict:
        """AI를 사용하여 메시지에서 메타데이터 추출"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 또는 사용 가능한 최신 모델
                messages=[
                    {"role": "system", "content": """메타데이터 관리자입니다. 
                    사용자의 메시지에서 다음 정보를 추출하여 JSON 형식으로 반환하세요:
                    - topics: 언급된 주제들의 배열
                    - entities: 언급된 구체적 대상(영화, 책, 장소 등)의 배열
                    - sentiment: 감정 상태 (positive, negative, neutral)
                    - preferences: 사용자의 선호도 정보를 키-값 형태로
                    - category: 대화의 카테고리 (entertainment, daily_life, question, etc)
                    
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
            
            # AI 응답을 JSON으로 파싱
            extracted_data = json.loads(response.choices[0].message.content)
            return extracted_data
            
        except Exception as e:
            print(f"메타데이터 추출 중 오류 발생: {str(e)}")
            return {}

    def update_metadata(self, message: str) -> dict:
        """메시지 기반으로 메타데이터 업데이트"""
        try:
            # 기존 대화 수 조회
            from app import Message  # 순환 참조 방지를 위해 지역 임포트
            interaction_count = Message.query.filter_by(user_id=self.user_id).count()
            
            # AI를 사용하여 메타데이터 추출
            extracted_metadata = self.extract_metadata_with_ai(message)
            
            # 메타데이터 업데이트
            self.metadata.update({
                'interaction_count': interaction_count + 1,
                'timestamp': datetime.now(KST).isoformat(),
                'message_length': len(message),
                'topics': extracted_metadata.get('topics', []),
                'entities': extracted_metadata.get('entities', []),
                'sentiment': extracted_metadata.get('sentiment'),
                'category': extracted_metadata.get('category')
            })
            
            # preferences 업데이트 (기존 선호도 정보 유지하면서 새로운 정보 추가)
            new_preferences = extracted_metadata.get('preferences', {})
            if new_preferences:
                if 'preferences' not in self.metadata:
                    self.metadata['preferences'] = {}
                self.metadata['preferences'].update(new_preferences)
            
            # 대화 컨텍스트 업데이트
            if 'conversation_context' not in self.metadata:
                self.metadata['conversation_context'] = {'recent_topics': [], 'mentioned_items': []}
            
            # 최근 토픽 업데이트 (최대 5개 유지)
            recent_topics = self.metadata['conversation_context']['recent_topics']
            new_topics = extracted_metadata.get('topics', [])
            recent_topics.extend(new_topics)
            self.metadata['conversation_context']['recent_topics'] = list(dict.fromkeys(recent_topics))[-5:]
            
            # 언급된 아이템 업데이트
            mentioned_items = self.metadata['conversation_context']['mentioned_items']
            new_entities = extracted_metadata.get('entities', [])
            for entity in new_entities:
                if entity not in mentioned_items:
                    mentioned_items.append(entity)
            
            return self.metadata
            
        except Exception as e:
            print(f"메타데이터 업데이트 중 오류 발생: {str(e)}")
            return self.metadata

    def get_formatted_metadata(self) -> str:
        """메타데이터를 포맷팅된 문자열로 반환"""
        return json.dumps(self.metadata, indent=2, ensure_ascii=False)