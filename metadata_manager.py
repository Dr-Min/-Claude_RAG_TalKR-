# metadata_manager.py
from openai import OpenAI
from datetime import datetime
import json
import os

class MetadataManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.metadata_dir = 'metadata'
        os.makedirs(self.metadata_dir, exist_ok=True)
        self.metadata_file = f"{self.metadata_dir}/metadata_{user_id}.json"
        self.client = OpenAI()

    def load_metadata(self) -> dict:
        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "user_preferences": {},
                "personal_info": {},
                "conversation_context": {
                    "recent_topics": [],
                    "key_points": []
                }
            }

    def update_metadata(self, message: str) -> dict:
        current_metadata = self.load_metadata()
        
        try:
            # GPT-4o-mini를 사용한 메타데이터 추출
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """메타데이터 관리자입니다. 
                    사용자의 메시지에서 중요한 정보를 추출하여 구조화된 메타데이터로 변환합니다.
                    새로운 정보는 추가하고, 기존 정보는 유지하면서 업데이트합니다.
                    JSON 형식으로만 응답하세요."""},
                    {"role": "user", "content": f"""현재 메타데이터:
                    {json.dumps(current_metadata, indent=2, ensure_ascii=False)}
                    
                    새로운 메시지:
                    {message}
                    
                    위 메시지에서 추출할 수 있는 새로운 정보를 기존 메타데이터에 추가하거나 업데이트해서
                    JSON 형식으로만 반환해주세요."""}
                ],
                temperature=0
            )
            
            # GPT 응답에서 JSON 추출
            updated_metadata = json.loads(response.choices[0].message.content)
            self.save_metadata(updated_metadata)
            return updated_metadata
            
        except Exception as e:
            print(f"메타데이터 업데이트 중 오류 발생: {str(e)}")
            return current_metadata

    def save_metadata(self, metadata: dict):
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"메타데이터 저장 중 오류 발생: {str(e)}")

    def get_formatted_metadata(self) -> str:
        """AI 프롬프트용 메타데이터 포맷팅"""
        try:
            metadata = self.load_metadata()
            return json.dumps(metadata, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"메타데이터 포맷팅 중 오류 발생: {str(e)}")
            return "{}"

    def merge_metadata(self, old_metadata: dict, new_metadata: dict) -> dict:
        """두 메타데이터를 재귀적으로 병합"""
        if isinstance(old_metadata, dict) and isinstance(new_metadata, dict):
            result = old_metadata.copy()
            for key, value in new_metadata.items():
                if key in result and isinstance(value, dict):
                    result[key] = self.merge_metadata(result[key], value)
                else:
                    result[key] = value
            return result
        return new_metadata