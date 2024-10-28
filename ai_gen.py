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
from hnswlib import Index
import logging
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# # 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_gen.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
client = anthropic.Anthropic()
api_key = os.getenv("ANTHROPIC_API_KEY")

class IndexingDebugger:
    def __init__(self):
        self.metrics = {
            'index_size': 0,
            'query_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'index_updates': 0,
            'memory_usage': 0,
            'last_gc_time': time.time(),
            'performance_alerts': []
        }
        self.perf_thresholds = {
            'query_time': 0.1,     # 100ms
            'memory_usage': 1000,   # 1GB
            'cache_hit_rate': 0.5   # 50%
        }

    def log_query(self, query_time: float) -> None:
        self.metrics['query_times'].append(query_time)
        self._check_performance()

    def log_cache_hit(self) -> None:
        self.metrics['cache_hits'] += 1

    def log_cache_miss(self) -> None:
        self.metrics['cache_misses'] += 1

    def log_index_update(self) -> None:
        self.metrics['index_updates'] += 1

    def update_metrics(self, index_size: int) -> None:
        self.metrics['index_size'] = index_size
        self.metrics['memory_usage'] = psutil.Process().memory_info().rss / (1024 * 1024)
        self._check_performance()

    def _check_performance(self) -> None:
        stats = self.get_statistics()
        alerts = []

        if stats['average_query_time'] > self.perf_thresholds['query_time']:
            alerts.append("⚠️ 쿼리 성능 저하 감지")
            logger.warning(f"쿼리 성능 저하: {stats['average_query_time']:.3f}초")

        if stats['memory_usage'] > self.perf_thresholds['memory_usage']:
            alerts.append("⚠️ 높은 메모리 사용량 감지")
            logger.warning(f"높은 메모리 사용량: {stats['memory_usage']:.1f}MB")

        if stats['cache_hit_rate'] < self.perf_thresholds['cache_hit_rate']:
            alerts.append("⚠️ 낮은 캐시 효율 감지")
            logger.warning(f"낮은 캐시 히트율: {stats['cache_hit_rate']*100:.1f}%")

        self.metrics['performance_alerts'] = alerts

    def get_statistics(self) -> Dict[str, float]:
        recent_queries = self.metrics['query_times'][-100:]
        total_cache_attempts = self.metrics['cache_hits'] + self.metrics['cache_misses']
        
        return {
            'index_size': self.metrics['index_size'],
            'average_query_time': sum(recent_queries) / len(recent_queries) if recent_queries else 0,
            'cache_hit_rate': self.metrics['cache_hits'] / total_cache_attempts if total_cache_attempts > 0 else 0,
            'total_updates': self.metrics['index_updates'],
            'memory_usage': self.metrics['memory_usage']
        }

    def print_debug_info(self) -> None:
        stats = self.get_statistics()
        logger.info("\n=== 인덱싱 디버그 정보 ===")
        logger.info(f"인덱스 크기: {stats['index_size']} 벡터")
        logger.info(f"평균 쿼리 시간: {stats['average_query_time']:.3f}초")
        logger.info(f"캐시 히트율: {stats['cache_hit_rate']*100:.1f}%")
        logger.info(f"총 인덱스 업데이트: {stats['total_updates']}회")
        logger.info(f"메모리 사용량: {stats['memory_usage']:.1f}MB")
        
        if self.metrics['performance_alerts']:
            logger.warning("\n성능 경고:")
            for alert in self.metrics['performance_alerts']:
                logger.warning(alert)
        logger.info("=====================\n")

@dataclass
class CacheStatus:
    input_text: str
    normalized_text: str
    cache_hit: bool
    original_text: Optional[str]
    embedding_time: Optional[float]

class OptimizedExampleSelector:
    def __init__(self):
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.text_variants: Dict[str, str] = {}
        self.examples_data: Dict[str, Dict] = {}
        self.db = None
        self.hnsw_index = None
        self.vector_dim = 1536
        self.max_elements = 10000
        self.ef_construction = 200
        self.M = 16
        self.debugger = IndexingDebugger()
        self.id_to_text: Dict[int, str] = {}

        self._initialize_db()
        self._initialize_hnsw_index()

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ])\1+', r'\1', text)
        text = re.sub(r'ㅋㅋ+', 'ㅋㅋ', text)
        text = re.sub(r'ㅎㅎ+', 'ㅎㅎ', text)
        text = re.sub(r'ㅠㅠ+', 'ㅠㅠ', text)
        text = re.sub(r'(세요|세여|셔요|셔여|슈|수|사요|쇼|사여)$', '세요', text)
        text = re.sub(r'(합니다|함다|해요|해여)$', '해요', text)
        return text

    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        시작_시간 = time.time()
        try:
            embeddings = OpenAIEmbeddings()
            결과 = embeddings.embed_documents(texts)
            logger.info(f"배치 임베딩 처리 시간: {time.time() - 시작_시간:.3f}초")
            return 결과
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise

    def _initialize_db(self) -> None:
        시작_시간 = time.time()
        단계별_시간: Dict[str, float] = {}
        self.new_texts_added = set()

        try:
            # 캐시 디렉토리 설정
            단계_시작 = time.time()
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            단계별_시간['1_캐시_디렉토리_설정'] = time.time() - 단계_시작

            # 캐시 로드
            단계_시작 = time.time()
            cache_file = cache_dir / "embeddings_cache.pkl"
            if cache_file.exists():
                logger.info("\n기존 캐시 파일 발견!")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.embeddings_cache = cached_data.get('embeddings', {})
                    self.text_variants = cached_data.get('variants', {})
                logger.info(f"로드된 캐시 엔트리 수: {len(self.embeddings_cache)}")
            단계별_시간['2_캐시_로드'] = time.time() - 단계_시작

            # Chroma DB 초기화
            단계_시작 = time.time()
            self.db = Chroma(
                collection_name="examples",
                embedding_function=OpenAIEmbeddings(),
                persist_directory=str(cache_dir)
            )
            단계별_시간['3_DB_초기화'] = time.time() - 단계_시작

            # 예제 데이터 로드 및 처리
            단계_시작 = time.time()
            try:
                with open('ragdata100.json', 'r', encoding='utf-8') as f:
                    examples = json.load(f)
                    self.examples_data = {ex['input']: ex for ex in examples}
                
                new_texts = []
                for example in examples:
                    normalized_text = self._normalize_text(example['input'])
                    if normalized_text not in self.embeddings_cache:
                        new_texts.append(normalized_text)
                        self.text_variants[normalized_text] = example['input']
                        self.new_texts_added.add(normalized_text)
                
                if new_texts:
                    logger.info(f"\n새로운 텍스트 {len(new_texts)}개 발견, 임베딩 생성 중...")
                    new_embeddings = self._batch_embed(new_texts)
                    for text, embedding in zip(new_texts, new_embeddings):
                        self.embeddings_cache[text] = embedding
                else:
                    logger.info("\n모든 텍스트가 이미 캐시되어 있음")
                    
            except FileNotFoundError:
                logger.error("예제 파일을 찾을 수 없습니다")
            except json.JSONDecodeError:
                logger.error("JSON 디코딩 오류 발생")
            단계별_시간['4_데이터_로드_및_처리'] = time.time() - 단계_시작

            # DB 업데이트
            단계_시작 = time.time()
            self._update_db()
            단계별_시간['5_DB_업데이트'] = time.time() - 단계_시작

            # 캐시 저장
            단계_시작 = time.time()
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings_cache,
                    'variants': self.text_variants
                }, f)
            단계별_시간['6_캐시_저장'] = time.time() - 단계_시작

            전체_시간 = time.time() - 시작_시간
            단계별_시간['7_총_초기화_시간'] = 전체_시간

            # 성능 메트릭 업데이트
            self.debugger.update_metrics(len(self.embeddings_cache))

            logger.info("\n=== 초기화 시간 분석 ===")
            for 단계, 소요시간 in sorted(단계별_시간.items()):
                logger.info(f"{단계}: {소요시간:.3f}초")

            logger.info("\n=== 캐시 상태 ===")
            logger.info(f"캐시된 텍스트 수: {len(self.embeddings_cache)}")
            logger.info(f"원본 텍스트 변형 수: {len(self.text_variants)}")
            logger.info(f"새로 추가된 텍스트 수: {len(self.new_texts_added)}")

        except Exception as e:
            logger.error(f"DB 초기화 중 오류 발생: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """임베딩 캐시 조회 (메모리 효율성 개선)"""
        return self.embeddings_cache.get(text)

    def _initialize_hnsw_index(self) -> None:
            시작_시간 = time.time()
            단계별_시간: Dict[str, float] = {}

            try:
                # 1단계: 인덱스 파일 확인
                단계_시작 = time.time()
                index_path = Path("cache/hnsw_index.bin")
                mapping_path = Path("cache/hnsw_mapping.pkl")
                단계별_시간['1_경로_확인'] = time.time() - 단계_시작

                # 2단계: HNSW 인덱스 초기화
                단계_시작 = time.time()
                self.hnsw_index = Index(space='cosine', dim=self.vector_dim)
                
                if index_path.exists() and mapping_path.exists():
                    try:
                        self.hnsw_index.load_index(str(index_path))
                        with open(mapping_path, 'rb') as f:
                            self.id_to_text = pickle.load(f)
                        logger.info("기존 HNSW 인덱스를 로드했습니다.")
                    except Exception as e:
                        logger.error(f"인덱스 로드 중 오류 발생: {str(e)}, 새 인덱스를 생성합니다.")
                        self._create_new_index()
                else:
                    self._create_new_index()
                
                단계별_시간['2_인덱스_초기화'] = time.time() - 단계_시작

                # 3단계: 인덱스 구축 또는 업데이트
                단계_시작 = time.time()
                if self.embeddings_cache and not index_path.exists():
                    self._build_index()
                단계별_시간['3_인덱스_구축'] = time.time() - 단계_시작

                # 4단계: 검색 최적화 설정
                단계_시작 = time.time()
                self.hnsw_index.set_ef(50)
                단계별_시간['4_검색_최적화'] = time.time() - 단계_시작

                # 성능 메트릭 업데이트
                self.debugger.update_metrics(self.hnsw_index.get_current_count())

                전체_시간 = time.time() - 시작_시간
                단계별_시간['5_총_초기화_시간'] = 전체_시간

                logger.info("\n=== HNSW 인덱스 상태 ===")
                logger.info(f"현재 인덱스 크기: {self.hnsw_index.get_current_count()} 벡터")
                logger.info(f"최대 벡터 수: {self.max_elements}")
                logger.info(f"초기화 시간: {전체_시간:.3f}초")

            except Exception as e:
                logger.error(f"HNSW 인덱스 초기화 중 오류 발생: {str(e)}")
                self.hnsw_index = None

    def _create_new_index(self) -> None:
        """새로운 HNSW 인덱스 생성"""
        self.hnsw_index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.id_to_text = {}
        logger.info("새로운 HNSW 인덱스를 생성했습니다.")

    def _build_index(self) -> None:
        """HNSW 인덱스 구축"""
        vectors = []
        texts = []
        for text, embedding in self.embeddings_cache.items():
            vectors.append(embedding)
            texts.append(text)
        
        vectors = np.array(vectors, dtype='float32')
        self.hnsw_index.add_items(vectors, list(range(len(vectors))))
        self.id_to_text = {i: text for i, text in enumerate(texts)}
        
        # 인덱스 저장
        Path("cache").mkdir(exist_ok=True)
        self.hnsw_index.save_index("cache/hnsw_index.bin")
        with open("cache/hnsw_mapping.pkl", 'wb') as f:
            pickle.dump(self.id_to_text, f)
        
        logger.info(f"HNSW 인덱스를 구축했습니다. (벡터 수: {len(vectors)})")

    def _update_db(self) -> None:
        시작_시간 = time.time()
        단계별_시간: Dict[str, float] = {}

        try:
            # 컬렉션 초기화
            단계_시작 = time.time()
            self.db.delete_collection()
            self.db = Chroma(
                collection_name="examples",
                embedding_function=OpenAIEmbeddings(),
                persist_directory="cache"
            )
            단계별_시간['1_컬렉션_초기화'] = time.time() - 단계_시작
            
            # 데이터 준비
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
                    'output': example_data.get('output', '')
                })
            단계별_시간['2_데이터_준비'] = time.time() - 단계_시작
            
            # DB에 데이터 추가
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

            logger.info("\n=== DB 업데이트 완료 ===")
            logger.info(f"처리된 텍스트 수: {len(texts)}")
            logger.info(f"총 소요 시간: {전체_시간:.3f}초")

        except Exception as e:
            logger.error(f"DB 업데이트 중 오류 발생: {str(e)}")
            raise

    def _update_hnsw_index(self, new_vectors: List[np.ndarray], new_texts: List[str]) -> None:
        if not self.hnsw_index or not new_vectors:
            return

        시작_시간 = time.time()
        try:
            current_size = self.hnsw_index.get_current_count()
            vectors = np.array(new_vectors, dtype='float32')
            
            # 인덱스에 새 벡터 추가
            self.hnsw_index.add_items(vectors, list(range(current_size, current_size + len(vectors))))
            
            # 매핑 정보 업데이트
            for i, text in enumerate(new_texts):
                self.id_to_text[current_size + i] = text
            
            # 인덱스 저장
            self.hnsw_index.save_index("cache/hnsw_index.bin")
            with open("cache/hnsw_mapping.pkl", 'wb') as f:
                pickle.dump(self.id_to_text, f)
            
            # 디버깅 정보 업데이트
            self.debugger.log_index_update()
            self.debugger.update_metrics(self.hnsw_index.get_current_count())
            
            logger.info(f"\n인덱스 업데이트 완료:")
            logger.info(f"추가된 벡터 수: {len(new_vectors)}")
            logger.info(f"현재 총 벡터 수: {self.hnsw_index.get_current_count()}")
            logger.info(f"업데이트 시간: {time.time() - 시작_시간:.3f}초")
            
        except Exception as e:
            logger.error(f"HNSW 인덱스 업데이트 중 오류 발생: {str(e)}")

    def get_examples(self, query: str, k: int = 3) -> Tuple[List[Dict], Dict, Dict[str, float]]:
        """이전의 find_examples 메서드와 동일한 구현"""
        시작_시간 = time.time()
        단계별_시간: Dict[str, float] = {}

        try:
            # 1단계: 쿼리 정규화 및 임베딩
            단계_시작 = time.time()
            normalized_query = self._normalize_text(query)
            query_embedding = self.embeddings_cache.get(normalized_query)
            
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
                cache_status["embedding_time"] = time.time() - embed_start_time
                self.embeddings_cache[normalized_query] = query_embedding
                self.text_variants[normalized_query] = query
            
            단계별_시간['1_쿼리_처리'] = time.time() - 단계_시작
            
            # 2단계: HNSW 검색 수행
            단계_시작 = time.time()
            selected_examples = []
            
            if self.hnsw_index:
                labels, distances = self.hnsw_index.knn_query(
                    np.array([query_embedding], dtype='float32'), k=k
                )
                
                for idx in labels[0]:
                    normalized_text = self.id_to_text[idx]
                    original_text = self.text_variants.get(normalized_text, normalized_text)
                    example_data = self.examples_data.get(original_text, {})
                    
                    selected_examples.append({
                        'input': original_text,
                        'output': example_data.get('output', '')
                    })
            else:
                results = self.db.similarity_search_by_vector(
                    embedding=query_embedding,
                    k=k
                )
                
                for doc in results:
                    original_text = doc.page_content
                    output = doc.metadata.get('output', '')
                    selected_examples.append({
                        'input': original_text,
                        'output': output
                    })
                
            단계별_시간['2_벡터_검색'] = time.time() - 단계_시작

            # 3단계: 결과 처리
            단계_시작 = time.time()
            print("\n=== 선택된 Few-shot 예제 ===")
            for i, example in enumerate(selected_examples, 1):
                print(f"예제 {i}:")
                print(f"Input: {example['input']}")
                print(f"Output: {example['output']}\n")
            단계별_시간['3_결과_처리'] = time.time() - 단계_시작

            전체_시간 = time.time() - 시작_시간
            단계별_시간['4_총_검색_시간'] = 전체_시간

            print("\n=== 벡터 검색 시간 분석 ===")
            for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
                print(f"{단계}: {소요시간:.3f}초")
            print("========================\n")
            
            return selected_examples, cache_status, 단계별_시간

        except Exception as e:
            logger.error(f"예제 검색 중 오류 발생: {str(e)}")
            return [], None, {}

def get_ai_response(recent_context: str, user_message_content: str, example_selector: OptimizedExampleSelector) -> Dict[str, Any]:
    시작_시간 = time.time()
    단계별_시간: Dict[str, float] = {}

    try:
        # find_examples를 get_examples로 변경
        단계_시작 = time.time()
        selected_examples, cache_status, timing = example_selector.get_examples(user_message_content, k=3)
        few_shot_str = "\n".join([ex['output'] for ex in selected_examples])
        단계별_시간['1_예제_선택'] = time.time() - 단계_시작

        # 2단계: 프롬프트 생성
        단계_시작 = time.time()
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
           주어지는 예제를 참고해서 대답해
            {recent_context}
          

          
            """),
            ("human", "{input}")
        ])

        messages = full_prompt.format_messages(input=user_message_content)
        단계별_시간['2_프롬프트_생성'] = time.time() - 단계_시작

        # 3단계: AI 응답 생성
        단계_시작 = time.time()
        claude = ChatAnthropic(
            temperature=0.8,
            model="claude-3-5-sonnet-20240620",
            max_tokens_to_sample=100,
            anthropic_api_key=api_key
        )

        response = claude.invoke(messages)
        ai_message_content = response.content
        단계별_시간['3_AI_응답_생성'] = time.time() - 단계_시작

        # 총 처리 시간 계산
        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_처리_시간'] = 전체_시간

        logger.info("\n=== AI 응답 생성 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items()):
            logger.info(f"{단계}: {소요시간:.3f}초")

        return {
            'message': ai_message_content,
            'selected_examples': selected_examples,
            'cache_status': cache_status,
            'timing': timing,
            'success': True
        }

    except Exception as e:
        logger.error(f"AI 응답 생성 중 오류 발생: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

__all__ = ['OptimizedExampleSelector', 'get_ai_response']
