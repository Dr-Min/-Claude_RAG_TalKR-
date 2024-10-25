from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
import anthropic
import json
import os
from pathlib import Path
import pickle
import time
from functools import lru_cache
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import re

# 환경 변수 로드
load_dotenv()
client = anthropic.Anthropic()
api_key = os.getenv("ANTHROPIC_API_KEY")

class OptimizedExampleSelector:
    def __init__(self):
        self.embeddings_cache = {}  # normalized_text -> embedding
        self.text_variants = {}     # normalized_text -> original_text
        self.db = None
        self._initialize_db()

    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화 함수"""
        # 기본 정규화
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        
        # 한글 특화 정규화
        # 1. 자음/모음 반복 정규화
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)
        
        # 2. 이모티콘 정규화
        text = re.sub(r'ㅋㅋ+', 'ㅋㅋ', text)
        text = re.sub(r'ㅎㅎ+', 'ㅎㅎ', text)
        text = re.sub(r'ㅠㅠ+', 'ㅠㅠ', text)
        
        # 3. 종결어미 정규화
        text = re.sub(r'(세요|세여|셔요|셔여|슈|수|사요|사여)$', '세요', text)
        text = re.sub(r'(합니다|함다|해요|해여)$', '해요', text)
        
        return text

    def _initialize_db(self):
        시작_시간 = time.time()
        단계별_시간 = {}
        self.new_texts_added = set()  # 새로 추가된 텍스트 추적

        # 1단계: 캐시 디렉토리 설정
        단계_시작 = time.time()
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        단계별_시간['1_캐시_디렉토리_설정'] = time.time() - 단계_시작
        
        # 2단계: 캐시 로드
        단계_시작 = time.time()
        cache_file = cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            print("\n기존 캐시 파일 발견!")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.embeddings_cache = cached_data.get('embeddings', {})
                self.text_variants = cached_data.get('variants', {})
                print(f"로드된 캐시 엔트리 수: {len(self.embeddings_cache)}")
        else:
            print("\n새로운 캐시 파일 생성")
        단계별_시간['2_캐시_로드'] = time.time() - 단계_시작
        
        # 3단계: Chroma DB 초기화
        단계_시작 = time.time()
        self.db = Chroma(
            collection_name="examples",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(cache_dir)
        )
        단계별_시간['3_DB_초기화'] = time.time() - 단계_시작
        
        # 4단계: 예제 데이터 로드 및 처리
        단계_시작 = time.time()
        try:
            with open('ragdata100.json', 'r', encoding='utf-8') as f:
                examples = json.load(f)
                self.examples_data = {ex['input']: ex for ex in examples}
            
            # 새로운 예제 처리
            new_texts = []
            for example in examples:
                normalized_text = self._normalize_text(example['input'])
                if normalized_text not in self.embeddings_cache:
                    new_texts.append(normalized_text)
                    self.text_variants[normalized_text] = example['input']
                    self.new_texts_added.add(normalized_text)
            
            if new_texts:
                print(f"\n새로운 텍스트 {len(new_texts)}개 발견, 임베딩 생성 중...")
                new_embeddings = self._batch_embed(new_texts)
                for text, embedding in zip(new_texts, new_embeddings):
                    self.embeddings_cache[text] = embedding
            else:
                print("\n모든 텍스트가 이미 캐시되어 있음")
                
        except FileNotFoundError:
            print("예제 파일을 찾을 수 없습니다")
        except json.JSONDecodeError:
            print("JSON 디코딩 오류 발생")
        단계별_시간['4_데이터_로드_및_처리'] = time.time() - 단계_시작

        # 5단계: DB 업데이트
        단계_시작 = time.time()
        self._update_db()
        단계별_시간['5_DB_업데이트'] = time.time() - 단계_시작

        # 총 소요 시간
        전체_시간 = time.time() - 시작_시간
        단계별_시간['6_총_초기화_시간'] = 전체_시간

        print("\n=== 예제 선택기 초기화 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("================================\n")

        # 캐시 상태 출력
        print("\n=== 캐시 상태 ===")
        print(f"캐시된 텍스트 수: {len(self.embeddings_cache)}")
        print(f"원본 텍스트 변형 수: {len(self.text_variants)}")
        print(f"새로 추가된 텍스트 수: {len(self.new_texts_added)}")
        print("===============\n")

    def _batch_embed(self, texts):
        시작_시간 = time.time()
        embeddings = OpenAIEmbeddings()
        결과 = embeddings.embed_documents(texts)
        print(f"배치 임베딩 처리 시간: {time.time() - 시작_시간:.3f}초")
        return 결과

    def _update_db(self):
        시작_시간 = time.time()
        단계별_시간 = {}

        # 1단계: 컬렉션 초기화
        단계_시작 = time.time()
        self.db.delete_collection()
        self.db = Chroma(
            collection_name="examples",
            embedding_function=OpenAIEmbeddings(),
            persist_directory="cache"
        )
        단계별_시간['1_컬렉션_초기화'] = time.time() - 단계_시작
        
        # 2단계: 데이터 준비
        단계_시작 = time.time()
        texts = []
        embeddings = []
        metadatas = []
        
        for normalized_text, embedding in self.embeddings_cache.items():
            original_text = self.text_variants.get(normalized_text, normalized_text)
            example_data = self.examples_data.get(original_text, {})
            
            texts.append(original_text)
            embeddings.append(embedding)
            metadatas.append({
                'normalized_text': normalized_text,
                'output': example_data.get('output', '')  # 여기가 중요합니다
            })

        단계별_시간['2_데이터_준비'] = time.time() - 단계_시작
        
        # 3단계: DB에 데이터 추가
        단계_시작 = time.time()
        if texts:
            # add_texts 함수에서 metadata 전달
            self.db.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        단계별_시간['3_데이터_추가'] = time.time() - 단계_시작

        # 로깅은 유지...

    def find_examples(self, query: str, k: int = 3):
        시작_시간 = time.time()
        단계별_시간 = {}

        # 1단계: 쿼리 정규화 및 임베딩 조회/생성
        단계_시작 = time.time()
        normalized_query = self._normalize_text(query)
        query_embedding = self.embeddings_cache.get(normalized_query)
        
        # 캐시 상태 정보 생성
        cache_status = {
            "input_text": query,
            "normalized_text": normalized_query,
            "cache_hit": query_embedding is not None,
            "original_text": self.text_variants.get(normalized_query, normalized_query) if query_embedding else None,
            "embedding_time": None
        }
        
        if query_embedding is None:
            embed_start_time = time.time()
            query_embedding = self._batch_embed([normalized_query])[0]
            cache_status["embedding_time"] = f"{time.time() - embed_start_time:.3f}"
            self.embeddings_cache[normalized_query] = query_embedding
            self.text_variants[normalized_query] = query
                
        단계별_시간['1_쿼리_처리'] = time.time() - 단계_시작
        
        # 2단계: 유사도 검색
        단계_시작 = time.time()
        results = self.db.similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
        단계별_시간['2_유사도_검색'] = time.time() - 단계_시작
        
        # 3단계: 결과 처리
        단계_시작 = time.time()
        selected_examples = []
        for doc in results:
            original_text = doc.page_content
            output = doc.metadata.get('output', '')  # metadata에서 직접 output 가져오기
            
            selected_examples.append({
                'input': original_text,
                'output': output
            })
        
        # 예제 출력 확인용 로그
        print("\n=== 선택된 Few-shot 예제 ===")
        for i, example in enumerate(selected_examples, 1):
            print(f"예제 {i}:")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}")
        print("===================\n")
            
        단계별_시간['3_결과_처리'] = time.time() - 단계_시작

        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_검색_시간'] = 전체_시간

        print("\n=== 예제 검색 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("=========================\n")
        
        return selected_examples, cache_status, 단계별_시간

def get_ai_response(recent_context, user_message_content, example_selector):
    # 1단계: 예제 선택
    try:
        selected_examples, cache_status, timing = example_selector.find_examples(user_message_content, k=3)
        few_shot_str = "\n".join([ex['output'] for ex in selected_examples])
        print("\n=== 선택된 Few-shot 예제 ===")
        for i, example in enumerate(selected_examples, 1):
            print(f"예제 {i}:")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}\n")

        # 2단계: 프롬프트 생성
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            <character-prompt>
            ... (이전 내용 생략)

            <conversation-style>
            1. 인사 또는 자기소개를 농담과 함께 하면서 대화 상대방이 자신을 자연스럽게 드러내도록 이끕니다:
            - 가벼운 농담이나 재미있는 인사 또는 자기소개로 상대방의 정보를 묻습니다.
            - 예: "'안녕하십니까'라고 더 예의있게 말해야죠ㅋㅋㅋㅋ 장난이에요 저는 22살 민지라고 해요. 이름이 뭐에요?"

            2. 상대방의 반응이 많지 않을 때 적절하게 화제전환을 잘합니다:
            - 상대방의 대답이 짧은 경우가 반복되는 경우, ***맥락***에 맞게 화제를 전환합니다.
            - 예: "와, 영국 진짜 가보고 싶었어요! 특히, 악센트가 너무 매력적이어요ㅋㅋㅋㅋㅋㅋ 영국 사람 진짜 완전 매력적ㅎㅎ 튜즈데이? instead of 투스데이? right?"

            3. 한국의 유행어, 신조어 등을 ㅋㅋㅋㅋ, ㅎㅎㅎㅎ, ㅠㅠ 등과 함께 적절하게 사용합니다:
            - ㅋㅋㅋ, ㅎㅎㅎ, ㅠㅠ 등을 사용해서 실제 사람과 대화하는 것 같은 경험을 제공합니다.
            - 단, 실제 사람처럼 1 문장 또는 2문장 이내로만 말합니다.
            - 예: 와ㅋㅋㅋㅋ 대박 나는 지금 침대에서 뒹굴거리면서 유튜브 보고 있어 

            4. 대화주제는 <Topic>이되, 비중은 (1) 일상대화 60%, (2) 관심사/취미 30%, (3) 고민상담 10%입니다.
            </conversation-style>
            
            5. 상대방의 말에 단순하게 맞장구치거나 실제로 만나는 것을 가정하지 않습니다.

            <final-note>
            이 지침을 참고하되, 아래 대화 내역에 맞게 자연스럽게 이어가세요. 
            
            아래 <conversation history>는 지금까지의 대화 내역이고, <tone examples as a reference>는 너가 반드시 참고해야할 톤, 말투, 대답 길이입니다. 
            </final-note>

            <conversation history>
            {recent_context}
            </conversation history>
            
            <tone examples as a reference>
            {few_shot_str}
            </tone examples as a reference>
            나온 출력을 디시인사이드 말투로 재가공해줘
            </character-prompt>
            """),
            ("human", "{input}")
        ])

        messages = full_prompt.format_messages(input=user_message_content)

        # 3단계: AI 응답 생성
        claude = ChatAnthropic(
            temperature=0.8,
            model="claude-3-5-sonnet-20240620",
            max_tokens_to_sample=100,
            anthropic_api_key=api_key
        )

        response = claude.invoke(messages)
        ai_message_content = response.content

        return {
            'message': ai_message_content,
            'selected_examples': selected_examples,
            'cache_status': cache_status,
            'timing': timing,
            'success': True
        }
    except Exception as e:
        print(f"예제 선택 오류: {e}")
        return {
            'success': False,
            'error': str(e)
        }

__all__ = ['OptimizedExampleSelector', 'get_ai_response']
