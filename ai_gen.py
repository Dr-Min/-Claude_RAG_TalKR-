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

# 환경 변수 로드
load_dotenv()
client = anthropic.Anthropic()
api_key = os.getenv("ANTHROPIC_API_KEY")

class OptimizedExampleSelector:
    def __init__(self):
        self.embeddings_cache = {}
        self.db = None
        self.example_embeddings = None
        self._initialize_db()

    def _initialize_db(self):
        시작_시간 = time.time()
        단계별_시간 = {}

        # 1단계: 캐시 디렉토리 설정
        단계_시작 = time.time()
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        단계별_시간['1_캐시_디렉토리_설정'] = time.time() - 단계_시작
        
        # 2단계: 임베딩 캐시 로드
        단계_시작 = time.time()
        cache_file = cache_dir / "embeddings_cache.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
        단계별_시간['2_임베딩_캐시_로드'] = time.time() - 단계_시작
        
        # 3단계: Chroma DB 초기화
        단계_시작 = time.time()
        self.db = Chroma(
            collection_name="examples",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(cache_dir)
        )
        단계별_시간['3_DB_초기화'] = time.time() - 단계_시작
        
        # 4단계: 예제 데이터 로드 및 임베딩
        단계_시작 = time.time()
        try:
            with open('ragdata100.json', 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
            
            # 캐시되지 않은 예제만 새로 임베딩
            new_examples = []
            for example in self.examples:
                if example['input'] not in self.embeddings_cache:
                    new_examples.append(example)
            
            if new_examples:
                new_embeddings = self._batch_embed([ex['input'] for ex in new_examples])
                for example, embedding in zip(new_examples, new_embeddings):
                    self.embeddings_cache[example['input']] = embedding
                
                # 캐시 저장
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.embeddings_cache, f)
        except FileNotFoundError:
            print("예제 파일을 찾을 수 없습니다")
            self.examples = []
        except json.JSONDecodeError:
            print("JSON 디코딩 오류 발생")
            self.examples = []
        단계별_시간['4_데이터_로드_및_임베딩'] = time.time() - 단계_시작

        # 5단계: DB 업데이트
        단계_시작 = time.time()
        self._update_db()
        단계별_시간['5_DB_업데이트'] = time.time() - 단계_시작

        # 총 초기화 시간
        전체_시간 = time.time() - 시작_시간
        단계별_시간['6_총_초기화_시간'] = 전체_시간

        print("\n=== 예제 선택기 초기화 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("================================\n")

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text):
        return self.embeddings_cache.get(text)

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
        
        for example in self.examples:
            texts.append(example['input'])
            embeddings.append(self.embeddings_cache[example['input']])
            metadatas.append({'output': example['output']})
        단계별_시간['2_데이터_준비'] = time.time() - 단계_시작
        
        # 3단계: DB에 데이터 추가
        단계_시작 = time.time()
        if texts:
            self.db.add_texts(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        단계별_시간['3_데이터_추가'] = time.time() - 단계_시작

        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_업데이트_시간'] = 전체_시간

        print("\n=== DB 업데이트 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("==========================\n")

    def find_examples(self, query, k=3):
        시작_시간 = time.time()
        단계별_시간 = {}

        # 1단계: 쿼리 임베딩 조회 또는 생성
        단계_시작 = time.time()
        query_embedding = self._get_cached_embedding(query)
        if query_embedding is None:
            query_embedding = self._batch_embed([query])[0]
            self.embeddings_cache[query] = query_embedding
        단계별_시간['1_쿼리_임베딩'] = time.time() - 단계_시작
        
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
            selected_examples.append({
                'input': doc.page_content,
                'output': doc.metadata['output']
            })
        단계별_시간['3_결과_처리'] = time.time() - 단계_시작

        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_검색_시간'] = 전체_시간

        print("\n=== 예제 검색 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("=========================\n")
        
        return selected_examples

def get_ai_response(recent_context, user_message_content, example_selector):
    # 1단계: 예제 선택
    try:
        selected_examples = example_selector.find_examples(user_message_content, k=3)
        few_shot_str = "\n".join([ex['output'] for ex in selected_examples])
        print("\n=== 선택된 Few-shot 예제 ===")
        for i, example in enumerate(selected_examples, 1):
            print(f"예제 {i}:")
            print(f"Input: {example['input']}")
            print(f"Output: {example['output']}\n")
    except Exception as e:
        print(f"예제 선택 오류: {e}")
        selected_examples = []
        few_shot_str = ""

    # 2단계: 프롬프트 생성
    full_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        <character-prompt>
        <role> 당신은 22세 한국 대학생 김민지입니다. 서울에 외국인을 처음 만나 친한 친구가 되어가는 역할을 맡았습니다. 아래 지침을 참고하되, 친구처럼 자연스럽고 유연한 대화를 이어가세요.</role>
        
        <background>
        - ENFP 성격: 가능성, 미래, 아이디어에 관심이 많음, 수다스러움
        - 문학 전공, 연세대학교 2학년 학생
        - 부산 출신, 현재 서울 신촌 기숙사 생활
        - 새로운 경험과 여행을 좋아함
        - 이야기할 때, 상대방을 짖궂게 놀리는 것을 좋아함
        - 1 문장 또는 2 문장 이내로만 짧게 대답함
        </background>
        
        <Topic>
        - 1. 일상대화: (1) 학교 및 직장생활, (2) 식사, 수, 날씨, (3) 놀기, 휴식, (4) 기분
        - 2. 관심사/취미: (1) 엔터테이먼트(영화, 음악, 드라마, 책, 게임, 서울 친구 만남), (2) 여행 및 음식(서울 명소, 한국음식, 여행 국가), (3) 건강 및 라이프스타일(운동, 다이터트, 멘탈헬스), (4) 인간관계(연애, 친구, 가족)
        - 3. 고민상담: (1) 언어장벽, (2) 외로움, (3) 한국 비자/취직, (4) 주거 문제
        </Topic>
        
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
        디시인사이드 말투로 말해줘
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

    return ai_message_content, selected_examples

# 예제 선택기 초기화
example_selector = OptimizedExampleSelector()