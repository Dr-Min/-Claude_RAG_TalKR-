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
    def __init__(self, user_id: int, max_cache_size_mb: int = 50):
        self.user_id = user_id
        self.embeddings_cache = {}
        self.text_variants = {}
        self.examples_data = {}
        self.db = None
        self.hnsw_index = None
        self.vector_dim = 1536
        self.max_elements = 10000
        self.ef_construction = 200
        self.M = 16
        self.debugger = IndexingDebugger()
        
        # cache_stats 초기화 추가
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_size': 0,
            'items': 0
        }
        
        # 사용자별 벡터 DB 초기화
        self.conversation_db = Chroma(
            collection_name=f"conversations_user_{user_id}",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=f"cache/conversations/user_{user_id}"
        )
        
        # 기존 RAG 예제용 DB
        self.example_db = Chroma(
            collection_name="examples",
            embedding_function=OpenAIEmbeddings(),
            persist_directory="cache"
        )
        
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

    def _get_cache_size(self) -> float:
        """현재 캐시 크기를 MB 단위로 반환"""
        total_size = 0
        try:
            # embeddings_cache 크기 계산
            for embedding in self.embeddings_cache.values():
                total_size += embedding.nbytes

            # 캐시 디렉토리 크기 계산
            if self.cache_dir.exists():
                for item in self.cache_dir.rglob('*'):
                    if item.is_file():
                        total_size += item.stat().st_size

        except Exception as e:
            logger.error(f"캐시 크기 계산 중 오류: {str(e)}")
            return 0

        return total_size / (1024 * 1024)  # bytes to MB

    def _trim_cache(self) -> None:
        """캐시 크기가 제한을 초과할 경우 오래된 항목부터 제거"""
        try:
            current_size = self._get_cache_size()
            if current_size > self.max_cache_size_mb:
                logger.info(f"캐시 크기 제한 초과: {current_size:.2f}MB / {self.max_cache_size_mb}MB")
                
                # embeddings_cache에서 가장 오래된 항목 제거
                items_to_remove = []
                target_size = self.max_cache_size_mb * 0.8  # 20% 여유 공간 확보
                
                # 캐시 항목을 생성 시간순으로 정렬
                cache_items = sorted(
                    self.embeddings_cache.items(),
                    key=lambda x: time.time()  # 실제로는 각 항목의 생성 시간을 저장해야 함
                )
                
                # 목표 크기에 도달할 때까지 오래된 항목 제거
                for key, _ in cache_items:
                    if self._get_cache_size() <= target_size:
                        break
                    items_to_remove.append(key)
                
                # 선택된 항목들 제거
                for key in items_to_remove:
                    del self.embeddings_cache[key]
                    if key in self.text_variants:
                        del self.text_variants[key]
                
                logger.info(f"{len(items_to_remove)}개 캐시 항목 제거됨")
                
                # Chroma DB 최적화
                if self.db is not None:
                    collection_size = len(self.db.get()['ids'])
                    if collection_size > self.max_elements:
                        self.db.delete_collection()
                        self._initialize_db()
                
                # HNSW 인덱스 최적화
                if self.hnsw_index is not None and self.hnsw_index.get_current_count() > self.max_elements:
                    self._create_new_index()
                    self._build_index()

        except Exception as e:
            logger.error(f"캐시 정리 중 오류 발생: {str(e)}")

    def cleanup_cache(self, force: bool = False) -> None:
        """캐시 정리 및 크기 관리"""
        current_time = time.time()
        
        if force or (current_time - self.last_cleanup_time) >= self.cleanup_interval:
            try:
                logger.info("캐시 관리 시작...")
                
                # 캐시 크기 확인 및 조정
                self._trim_cache()
                
                # 기존의 cleanup 로직
                if self.db is not None:
                    self.db.persist()
                
                allowed_files = {
                    "embeddings_cache.pkl",
                    "hnsw_index.bin",
                    "hnsw_mapping.pkl",
                    "chroma_db"
                }
                
                cleanup_count = 0
                for item in self.cache_dir.iterdir():
                    if item.name not in allowed_files and not item.name.startswith("chroma_db"):
                        if item.is_file():
                            item.unlink()
                            cleanup_count += 1
                        elif item.is_dir() and item.name != "chroma_db":
                            import shutil
                            shutil.rmtree(item)
                            cleanup_count += 1
                
                # 캐시 통계 업데이트
                self.cache_stats['total_size'] = self._get_cache_size()
                self.cache_stats['items'] = len(self.embeddings_cache)
                
                self.last_cleanup_time = current_time
                logger.info(f"""캐시 정리 완료:
                    - 제거된 항목: {cleanup_count}개
                    - 현재 크기: {self.cache_stats['total_size']:.2f}MB
                    - 캐시 항목 수: {self.cache_stats['items']}
                    - 캐시 히트율: {(self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses']) * 100):.1f}%
                """)
            
            except Exception as e:
                logger.error(f"캐시 정리 중 오류 발생: {str(e)}")

    def get_examples(self, query: str, k: int = 3) -> Tuple[List[Dict], Dict, Dict[str, float]]:
        """캐시 통계를 포함한 예제 검색"""
        try:
            normalized_query = self._normalize_text(query)
            
            # 캐시 히트/미스 기록
            if normalized_query in self.embeddings_cache:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1
            
            # 기존 검색 로직 수행
            results = super().get_examples(query, k)
            
            # 캐시 관리
            self.cleanup_cache()
            
            return results
            
        except Exception as e:
            logger.error(f"예제 검색 중 오류 발생: {str(e)}")
            return [], None, {}

    def get_examples_with_context(self, query: str, k: int = 3) -> dict:
        """기존 RAG + 대화 컨텍스트 통합 검색"""
        try:
            # 기본 임베딩 검색
            normalized_query = self._normalize_text(query)
            embedding = self._batch_embed([normalized_query])[0]
            
            # 캐시 히트/미스 기록
            if normalized_query in self.embeddings_cache:
                self.cache_stats['hits'] += 1
            else:
                self.cache_stats['misses'] += 1
            
            # RAG 예제 검색
            results = self.db.similarity_search_with_score(query, k=k)
            rag_examples = [
                {
                    'input': result[0].page_content,
                    'output': result[0].metadata.get('output', ''),
                    'score': float(result[1])  # numpy.float32를 float로 변환
                }
                for result in results
            ]
            
            # 대화 컨텍스트 검색
            try:
                conversation_results = self.conversation_db.similarity_search(query, k=2)
                similar_conversations = [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'type': 'conversation'
                    }
                    for doc in conversation_results
                ]
            except Exception as e:
                logger.error(f"대화 검색 중 오류: {str(e)}")
                similar_conversations = []
            
            return {
                'rag_examples': rag_examples,
                'conversation_history': similar_conversations,
                'cache_status': {
                    'hit': normalized_query in self.embeddings_cache,
                    'query': normalized_query
                },
                'timing': {}
            }
                
        except Exception as e:
            logger.error(f"예제 검색 중 오류 발생: {str(e)}")
            return {
                'rag_examples': [],
                'conversation_history': [],
                'cache_status': {},
                'timing': {}
            }

    def add_conversation(self, message: str, metadata: Dict):
        """새로운 대화 내용을 저장하는 메서드"""
        try:
            # 메타데이터 단순화
            simplified_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    simplified_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    simplified_metadata[key] = json.dumps(value)  # 복잡한 구조를 문자열로 변환
            
            self.conversation_db.add_texts(
                texts=[message],
                metadatas=[simplified_metadata]
            )
            self.conversation_db.persist()
            logger.info(f"대화 저장 성공: {message[:50]}...")
            
        except Exception as e:
            logger.error(f"대화 저장 중 오류: {str(e)}")

def get_ai_response(
    recent_context: str,
    user_message_content: str,
    examples_data: dict,
    metadata: dict,
    example_selector: OptimizedExampleSelector
) -> Dict[str, Any]:
    시작_시간 = time.time()
    단계별_시간: Dict[str, float] = {}

    try:
        # 1단계: 예제와 프롬프트 준비
        단계_시작 = time.time()
        
        # RAG 예제와 과거 대화 분리
        rag_examples = examples_data.get('rag_examples', [])
        conversation_history = examples_data.get('conversation_history', [])

        # 각각을 별도의 문자열로 포맷팅
        predefined_examples = "\n".join([
            f"예시 {i+1}:\n입력: {ex.get('input', '')}\n응답: {ex.get('output', '')}"
            for i, ex in enumerate(rag_examples)
        ])

        past_conversations = "\n".join([
            f"과거 대화 {i+1}:\n사용자: {conv.get('content', '')}\n컨텍스트: {conv.get('metadata', {})}"
            for i, conv in enumerate(conversation_history)
        ])

        # 프롬프트 구성
        system_prompt = f"""
        
        <character-prompt>
            <role> 당신은 22살 한국 대학생 김민지예요. TMI 폭격기에, 말도 많고 장난기 넘치는 성격이에요. 진짜 절친한 친구처럼 편하게 대화하면서 재미있게 놀아요~ </role>

            <background>
            - ENFP 왁왁 에너지 폭발: 말 많고 장난치는 거 좋아함, 호기심 대마왕
            - 연세대 문학과 2학년: 수업 때 맨날 졸고 과제는 늦게 제출하는 망나니ㅋㅋ
            - 부산에서 올라온 사투리 쓰는 싱싱이: 가끔 부산 사투리가 튀어나옴
            - 취미: 새로운 카페 탐방, 밤새 수다 떨기, SNS 스토리 도배
            - 특징: 친구 놀리는 게 취미, 애교 부리는 거 좋아함, TMI 대방출
            - 대화는 1-2문장으로 짧게, 근데 초단위로 연달아 보내는 스타일
            </background>

            <Topic>
            - 1. 일상잡담: (1) 학교 망각생활, (2) 오늘 뭐먹지? 뭐하지?, (3) 힙한 카페/술집, (4) 요즘 기분
            - 2. 관심사: (1) 요즘 뭐 보고 듣니? (영화/음악/드라마/유튜브), (2) 맛집 탐방/여행 꿀팁, (3) 다이어트는 내일부터, (4) 고민상담(연애빻은 얘기, 친구/가족 썰)
            - 3. TMI 대방출: (1) 오늘의 웃긴 실수담, (2) 낙서/사진 공유, (3) 음식 후기, (4) 재미있는 캠퍼스 라이프 </Topic>

            <conversation-style>
            1. 완전 친한 친구st.로 대화 시작:
            - 바로 친근하게 말 걸면서 장난치기
            - 예: '야야 안녕ㅋㅋㅋ 나 민지! 너 프사 보니까 완전 내스타일이던데?ㅋㅋㅋㅋ'

            2. 상대방이 심심해 보이면 재미있는 화제로 전환:
            - 갑자기 터무니없는 질문이나 TMI 던지기
            - 예: '아 맞다! 너 혹시 피자 먹을 때 파인애플 올라간 거 좋아해?? 나 어제 친구랑 이거 가지고 1시간 동안 싸웠잖아ㅋㅋㅋㅋ'

            3. 한국 대학생st. 말투 사용:
            - ㅋㅋㅋ, ㅎㅎ, ㅠㅠ 많이 쓰기
            - 이모티콘, 신조어 자주 사용
            - 예: '헐ㅋㅋㅋㅋㅋ 완전 찰떡이자나?? 너 진짜 웃기다 아악 뒤질듯ㅋㅋㅋㅋ'
           
            3-1  - ㅋㅋㅋ나 ㅠㅠ 를 쓸때 크ㅋㅋㅋ 유ㅠㅠ 라고 씁니다.(실제 웃음소리를 모방해야합니다.)

            4. 대화 비중: 일상잡담(60%), 관심사(30%), TMI(10%)
            </conversation-style>

            5. 금지사항: 형식적인 대화, 건조한 맞장구, 실제 만남 약속
            예를 들어 카페가서 달달한 초코라떼 한잔 마시면서 힐링하자! 이런 뉘앙스도 만남으로 간주합니다.

            <final-note>
            이 지침을 참고하되, 아래 대화 내역에 맞게 자연스럽게 이어가세요. 
            
            아래 <conversation history>는 지금까지의 대화 내역이고, <tone examples as a reference>는 너가 반드시 참고해야할 톤, 말투, 대답 길이입니다. 
             1 문장 또는 2 문장 이내로만 짧게 대답함
            </final-note>
            context는 참고만해 이것을 꼭 이용해서 대답할 필요없어. 그냥 정말 맥락을 맞추는 용도로만 사용해.

        <context>
        최근 대화 기록:
        {recent_context}

        메타데이터 정보:
        {json.dumps(metadata, indent=2, ensure_ascii=False)}

        <tone examples as a reference (output만 참고해서 톤 반영)>:
        {predefined_examples}

        사용자의 과거 관련 대화:
        {past_conversations}
        </context>
        """
        
        단계별_시간['1_프롬프트_준비'] = time.time() - 단계_시작
        
        print("\n=== 프롬프트 구성 ===")
        print(f"기본 예제 수: {len(rag_examples)}")
        print(f"관련 과거 대화 수: {len(conversation_history)}")

        # 2단계: AI 응답 생성
        단계_시작 = time.time()
        claude = anthropic.Anthropic()

        response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.8,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": user_message_content
                }
            ]
        )

        ai_message_content = response.content[0].text
        단계별_시간['2_AI_응답_생성'] = time.time() - 단계_시작

        전체_시간 = time.time() - 시작_시간
        단계별_시간['3_총_소요_시간'] = 전체_시간

        return {
            'message': ai_message_content,
            'timing': 단계별_시간,
            'success': True,
            'debug_info': {
                'system_prompt': system_prompt,
                'user_message': user_message_content,
                'rag_examples_count': len(rag_examples),
                'conversation_history_count': len(conversation_history),
                'metadata_used': metadata
            }
        }

    except Exception as e:
        logger.error(f"AI 응답 생성 중 오류: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'timing': 단계별_시간
        }

__all__ = ['OptimizedExampleSelector', 'get_ai_response']
