from langchain_openai import ChatOpenAI
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from pydantic import SecretStr
import anthropic

import json
import os
import base64
import secrets
import click
import re
from collections import Counter
from datetime import datetime, timedelta
from itertools import groupby
from operator import attrgetter
from threading import Thread

from flask import Flask, send_file, render_template, request, jsonify, session, url_for, redirect
from flask_migrate import Migrate
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from dotenv import load_dotenv
from sqlalchemy import desc
from flask_mail import Mail, Message as FlaskMessage
from flask_admin import BaseView, Admin, AdminIndexView, expose
from flask_admin.contrib.sqla import ModelView
from flask_admin.form import SecureForm
from flask.cli import with_appcontext
from pytz import timezone
import schedule
import time
from bs4 import BeautifulSoup
import requests

from functools import wraps
from flask_socketio import SocketIO
from functools import lru_cache
import numpy as np
from pathlib import Path
import pickle

LANGCHAIN_TRACING_V2=True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-diligent-larch-83"

load_dotenv()
client = anthropic.Anthropic()
api_key = os.getenv("ANTHROPIC_API_KEY")

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)  # Cross-Origin Resource Sharing 설정

# 한국 시간대 설정
KST = timezone('Asia/Seoul')

# 애플리케이션 설정
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# 환경 변수 로드 및 OpenAI 클라이언트 초기화
client = OpenAI()
migrate = Migrate(app, db)

# Flask-Mail 설정
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'mks010103@gmail.com' # 실제 이메일 주소로 변경
app.config['MAIL_PASSWORD'] = 'vhnk zrko wxxt oank' # 실제 앱 비밀번호로 변경
app.config['MAIL_DEFAULT_SENDER'] = 'mks010103@gmail.com' # 실제 이메일 주소로 변경

mail = Mail(app)

# 사용자 모델 정의
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    total_usage_time = db.Column(db.Integer, default=0)
    conversations = db.relationship('Conversation', backref='user', lazy=True)
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiration = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    # reports = db.relationship('Report', backref='user', lazy=True)
    messages = db.relationship('Message', backref='user', lazy=True)

    def set_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiration = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()

    def check_reset_token(self, token):
        return (self.reset_token == token and
                self.reset_token_expiration > datetime.utcnow())

# 대화 모델 정의
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    messages = db.relationship('Message', backref='conversation', lazy=True, order_by="Message.timestamp")

# 메시지 모델 정의
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(KST))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# 관리자 뷰 보안 설정
class SecureModelView(ModelView):
    form_base_class = SecureForm
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin

# 관리자 인덱스 뷰 설��
class MyAdminIndexView(AdminIndexView):
    @expose('/')
    def index(self):
        if not current_user.is_authenticated or not current_user.is_admin:
            return redirect(url_for('login', next=request.url))
        return super(MyAdminIndexView, self).index()

# 관리자 페이지 설정
admin = Admin(app, name='TalKR Admin', template_mode='bootstrap3', index_view=MyAdminIndexView())
admin.add_view(SecureModelView(User, db.session))
admin.add_view(SecureModelView(Conversation, db.session))
admin.add_view(SecureModelView(Message, db.session))

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def home():
    """
    홈페이지 라우트
    """
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    """
    로그인 처 라우트
    """
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        return jsonify({"success": True, "username": user.username})
    return jsonify({"success": False})

@app.route('/check_login', methods=['GET'])
def check_login():
    """
    로그인 상태 확인 라우트
    """
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """
    로그아웃 처리 라우트
    """
    logout_user()
    return jsonify({"success": True})

@app.route('/signup', methods=['POST'])
def signup():
    """
    회원가입 처리 라우트
    """
    data = request.json
    username = data['username']
    email = data['email']
    password = data['password']

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"success": False, "error": "email_taken"})
    
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({"success": False, "error": "username_taken"})
    
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"success": True, "message": "User created successfully"})

# 메모리 기능: 이전 대화 15개 가져오는 함수 정의
def get_recent_context(conversation_id, limit=20):
    recent_messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.desc()).limit(limit).all()
    recent_messages.reverse()
    context = []
    for msg in recent_messages:
        msg_type = "사용자" if msg.is_user else "AI"
        context.append(f"{msg_type}: {msg.content}")
    return "\n".join(context)

# 채팅처리를 위한 라우트: 사용자 메세지를 받아 AI 응답을 생성하고 반환합니다.

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    단계별_시간 = {}
    전체_시작_시간 = time.time()
    메시지_처리_시작_시간 = time.time()
    user_message_content = request.json['message']
    
    try:
        # 1단계: 대화 세션 확인/생성
        시작_시간 = time.time()
        active_conversation = Conversation.query.filter_by(
            user_id=current_user.id, 
            end_time=None
        ).first()
        
        if not active_conversation:
            active_conversation = Conversation(user_id=current_user.id)
            db.session.add(active_conversation)
            db.session.commit()
        단계별_시간['1_세션_확인'] = time.time() - 시작_시간
            
        # 2단계: 사용자 메시지 저장
        시작_시간 = time.time()
        user_message = Message(
            conversation_id=active_conversation.id,
            content=user_message_content,
            is_user=True,
            user_id=current_user.id
        )
        db.session.add(user_message)
        db.session.commit()
        단계별_시간['2_메시지_저장'] = time.time() - 시작_시간

        # 3단계: 최근 컨텍스트 가져오기
        시작_시간 = time.time()
        recent_context = get_recent_context(active_conversation.id)
        단계별_시간['3_컨텍스트_조회'] = time.time() - 시작_시간
        
        # 4단계: 최적화된 예제 선택
        시작_시간 = time.time()
        selected_examples_details = []  # 선택된 예제 상세 정보 저장
        try:
            selected_examples = example_selector.find_examples(user_message_content, k=3)
            few_shot_str = "\n".join([ex['output'] for ex in selected_examples])
            selected_examples_details = selected_examples  # 선택된 예제 저장
            
            # 예제 선택 결과 로깅
            print("\n=== 선택된 Few-shot 예제 ===")
            for i, example in enumerate(selected_examples, 1):
                print(f"예제 {i}:")
                print(f"Input: {example['input']}")
                print(f"Output: {example['output']}\n")
            print("===========================")
            
        except Exception as e:
            print(f"예제 선택 오류: {e}")
            selected_examples = []
            few_shot_str = ""
        단계별_시간['4_예제_선택'] = time.time() - 시작_시간

        # 5단계: 프롬프트 생성
        시작_시간 = time.time()
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
            
            </character-prompt>
            """),
            ("human", "{input}")
        ])
        
        messages = full_prompt.format_messages(input=user_message_content)
        단계별_시간['5_프롬프트_생성'] = time.time() - 시작_시간

        # 6단계: AI 응답 생성
        시작_시간 = time.time()
        claude = ChatAnthropic(
            temperature=0.8,
            model="claude-3-5-sonnet-20240620",
            max_tokens_to_sample=100,
            anthropic_api_key=api_key
        )
        
        response = claude.invoke(messages)
        ai_message_content = response.content
        단계별_시간['6_AI_응답_생성'] = time.time() - 시작_시간

        # 7단계: AI 응답 저장
        시작_시간 = time.time()
        ai_message = Message(
            conversation_id=active_conversation.id,
            content=ai_message_content,
            is_user=False,
            user_id=current_user.id
        )
        db.session.add(ai_message)
        db.session.commit()
        단계별_시간['7_응답_저장'] = time.time() - 시작_시간

        # 메시지 처리 총 시간 계산
        메시지_처리_총시간 = time.time() - 메시지_처리_시작_시간
        단계별_시간['8_메시지_처리_총시간'] = 메시지_처리_총시간

        # 전체 처리 시간 계산
        전체_처리_시간 = time.time() - 전체_시작_시간
        단계별_시간['9_전체_처리_시간'] = 전체_처리_시간

        # 처리 시간 로깅
        print("\n=== 처리 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0]) if x[0].split('_')[0].isdigit() else 999):
            print(f"{단계}: {소요시간:.3f}초")
        print("===================\n")

        # 응답 반환
        return jsonify({
            'message': ai_message_content,
            'message_id': ai_message.id,
            'success': True,
            'timing': 단계별_시간,
            'selected_examples': selected_examples_details  # 선택된 예제 정보도 함께 반환
        })

    except Exception as e:
        db.session.rollback()
        print(f"채팅 처리 중 오류 발생: {str(e)}")
        return jsonify({
            'message': '죄송합니다. 오류가 발생했습니다.', 
            'success': False,
            'error': str(e)
        }), 500

# 음성 생성 엔드포인트
@app.route('/generate_voice', methods=['POST'])
@login_required
def generate_voice():
    시작_시간 = time.time()
    음성_처리_시간 = {}
    
    try:
        # 1단계: 요청 데이터 처리
        단계_시작 = time.time()
        message_content = request.json['message']
        message_id = request.json['message_id']
        음성_처리_시간['1_요청_처리'] = time.time() - 단계_시작
        
        # 2단계: 음성 생성
        단계_시작 = time.time()
        speech_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=message_content,
            speed=0.9
        )
        음성_처리_시간['2_음성_생성'] = time.time() - 단계_시작
        
        # 3단계: 인코딩
        단계_시작 = time.time()
        audio_base64 = base64.b64encode(speech_response.content).decode('utf-8')
        음성_처리_시간['3_인코딩'] = time.time() - 단계_시작
        
        # 총 처리 시간
        전체_처리_시간 = time.time() - 시작_시간
        음성_처리_시간['4_총_처리_시간'] = 전체_처리_시간
        
        # 처리 시간 로깅
        print("\n=== 음성 생성 처리 시간 분석 ===")
        for 단계, 소요시간 in sorted(음성_처리_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("============================\n")
        
        return jsonify({
            'audio': audio_base64,
            'message_id': message_id,
            'timing': 음성_처리_시간,
            'success': True
        })
        
    except Exception as e:
        print(f"음성 생성 중 오류 발생: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timing': 음성_처리_시간
        }), 500 

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

        # 초기화 시간 로깅
        print("\n=== 예제 선택기 초기화 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
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

        # 총 업데이트 시간
        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_업데이트_시간'] = 전체_시간

        # 업데이트 시간 로깅
        print("\n=== DB 업데이트 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
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

        # 총 검색 시간
        전체_시간 = time.time() - 시작_시간
        단계별_시간['4_총_검색_시간'] = 전체_시간

        # 검색 시간 로깅
        print("\n=== 예제 검색 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("=========================\n")
        
        return selected_examples

# 예제 선택기 초기화
example_selector = OptimizedExampleSelector()

@app.route('/translate', methods=['POST'])
@login_required
def translate():
    """
    텍스트 번역을 위한 라우트
    """
    시작_시간 = time.time()
    단계별_시간 = {}

    try:
        # 1단계: 요청 처리
        단계_시작 = time.time()
        text = request.json['text']
        단계별_시간['1_요청_처리'] = time.time() - 단계_시작

        # 2단계: 번역 요청
        단계_시작 = time.time()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator. Translate the given Korean conversational text to conversational and casual English."},
                {"role": "user", "content": f"Translate this to easy and casual English: {text}"}
            ]
        )
        translation = response.choices[0].message.content
        단계별_시간['2_번역_생성'] = time.time() - 단계_시작

        # 총 처리 시간
        전체_처리_시간 = time.time() - 시작_시간
        단계별_시간['3_총_처리_시간'] = 전체_처리_시간

        # 처리 시간 로깅
        print("\n=== 번역 처리 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("========================\n")

        return jsonify({
            'translation': translation,
            'timing': 단계별_시간,
            'success': True
        })
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return jsonify({
            'error': 'Translation failed',
            'timing': 단계별_시간,
            'success': False
        }), 500

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    """
    사용자의 대화 기록을 가져오는 라우트
    """
    시작_시간 = time.time()
    단계별_시간 = {}
    
    try:
        # 1단계: 요청 파라미터 처리
        단계_시작 = time.time()
        date = request.args.get('date')
        단계별_시간['1_파라미터_처리'] = time.time() - 단계_시작
        
        # 2단계: 쿼리 구성
        단계_시작 = time.time()
        query = Conversation.query.filter_by(user_id=current_user.id)
        if date:
            query = query.filter(Conversation.start_time < datetime.strptime(date, '%Y-%m-%d'))
        conversations = query.order_by(desc(Conversation.start_time)).limit(10).all()
        단계별_시간['2_쿼리_실행'] = time.time() - 단계_시작
        
        # 3단계: 데이터 처리
        단계_시작 = time.time()
        history = []
        for conv in conversations:
            messages = sorted(conv.messages, key=attrgetter('timestamp'))
            grouped_messages = groupby(messages, key=lambda m: m.timestamp.astimezone(KST).date())
            for date, msgs in grouped_messages:
                history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'messages': [{
                        'content': msg.content,
                        'is_user': msg.is_user,
                        'timestamp': msg.timestamp.strftime('%H:%M')
                    } for msg in msgs]
                })
        단계별_시간['3_데이터_처리'] = time.time() - 단계_시간

        # 총 처리 시간
        전체_처리_시간 = time.time() - 시작_시간
        단계별_시간['4_총_처리_시간'] = 전체_처리_시간

        # 처리 시간 로깅
        print("\n=== 히스토리 조회 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), 
                                  key=lambda x: int(x[0].split('_')[0])):
            print(f"{단계}: {소요시간:.3f}초")
        print("===========================\n")
        
        return jsonify({
            'history': history,
            'timing': 단계별_시간,
            'success': True
        })
    except Exception as e:
        print(f"History retrieval error: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve history',
            'timing': 단계별_시간,
            'success': False
        }), 500

@app.route('/update_usage_time', methods=['POST'])
@login_required
def update_usage_time():
    """
    사용자의 총 사용 시간을 업데이트하는 라우트
    """
    data = request.json
    current_user.total_usage_time += data['time']
    db.session.commit()
    return jsonify({"success": True})

# @app.route('/translate', methods=['POST'])
# @login_required
# def translate():
#     """
#     텍스트 번역을 위한 라우트
#     """
#     text = request.json['text']
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a translator. Translate the given Korean conversational text to conversational and casual English."},
#                 {"role": "user", "content": f"Translate this to easy and casual English: {text}"}
#             ]
#         )
#         translation = response.choices[0].message.content
#         return jsonify({'translation': translation})
#     except Exception as e:
#         print(f"Translation error: {str(e)}")
#         return jsonify({'error': 'Translation failed'}), 500

# @app.route('/get_history', methods=['GET'])
# @login_required
# def get_history():
#     """
#     사용자의 대화 기록을 가져오는 라우트
#     """
#     date = request.args.get('date')
    
#     query = Conversation.query.filter_by(user_id=current_user.id)
#     if date:
#         query = query.filter(Conversation.start_time < datetime.strptime(date, '%Y-%m-%d'))
    
#     conversations = query.order_by(desc(Conversation.start_time)).limit(10).all()
    
#     history = []
#     for conv in conversations:
#         messages = sorted(conv.messages, key=attrgetter('timestamp'))
#         grouped_messages = groupby(messages, key=lambda m: m.timestamp.astimezone(KST).date())
#         for date, msgs in grouped_messages:
#             history.append({
#                 'date': date.strftime('%Y-%m-%d'),
#                 'messages': [{'content': msg.content, 'is_user': msg.is_user, 'timestamp': msg.timestamp.strftime('%H:%M')} for msg in msgs]
#             })
    
#     return jsonify({'history': history})

def send_async_email(app, msg):
    """
    비동기적으로 이메일을 보내는 함수
    """
    with app.app_context():
        try:
            mail.send(msg)
            print("Email sent successfully")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")

def send_password_reset_email(user):
    """
    비밀번호 재설정 이메일을 보내는 함수
    """
    token = user.reset_token
    msg = FlaskMessage(subject='Password Reset Request',
                       recipients=[user.email],
                       body=f'''To reset your password, visit the following link:
{url_for('reset_password_form', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
''')
    mail.send(msg)

@app.route('/request_reset', methods=['POST'])
def request_reset():
    """
    비밀번호 재설정 요청을 처리하는 라우트
    """
    try:
        email = request.json.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_reset_token()
            send_password_reset_email(user)
            return jsonify({"message": "Reset link sent to your email"})
        return jsonify({"message": "Email not found"}), 404
    except Exception as e:
        print(f"Error in request_reset: {str(e)}")
        return jsonify({"message": "An error occurred"}), 500

@app.route('/reset_password/<token>', methods=['GET'])
def reset_password_form(token):
    """
    비밀번호 재설정 폼을 표시하는 라우트
    """
    user = User.query.filter_by(reset_token=token).first()
    if user and user.check_reset_token(token):
        return render_template('reset_password.html', token=token)
    return "Invalid or expired token", 400

@app.route('/reset_password', methods=['POST'])
def reset_password():
    """
    비밀번호 재설정을 처리하는 라우트
    """
    token = request.json.get('token')
    new_password = request.json.get('new_password')
    user = User.query.filter_by(reset_token=token).first()
    if user and user.check_reset_token(token):
        user.password = generate_password_hash(new_password)
        user.reset_token = None
        user.reset_token_expiration = None
        db.session.commit()
        return jsonify({"message": "Password reset successful"})
    return jsonify({"message": "Invalid or expired token"}), 400

@app.route('/admin/backup_db')
@login_required
def backup_db():
    """
    데이터베스 백업을 위한 관리자 라우트
    """
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized access"}), 403
    
    try:
        db_path = os.path.join(app.instance_path, 'users.db')
        
        if not os.path.exists(db_path):
            return jsonify({"error": "Database file not found"}), 404

        return send_file(db_path, as_attachment=True, download_name='users.db')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@click.command('create-admin')
@with_appcontext
def create_admin_command():
    """관리자 사용자 생성을 위한 CLI 명령"""
    username = click.prompt('Enter admin username', type=str)
    email = click.prompt('Enter admin email', type=str)
    password = click.prompt('Enter admin password', type=str, hide_input=True, confirmation_prompt=True)
    
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        if click.confirm('User with this email already exists. Do you want to make this user an admin?'):
            existing_user.is_admin = True
            db.session.commit()
            click.echo('User updated to admin successfully')
        else:
            click.echo('Admin user creation cancelled')
    else:
        admin_user = User(username=username, email=email, password=generate_password_hash(password), is_admin=True)
        db.session.add(admin_user)
        db.session.commit()
        click.echo('Admin user created successfully')

app.cli.add_command(create_admin_command)

class UserConversationsView(BaseView):
    @expose('/')
    def index(self):
        users = User.query.all()
        return self.render('admin/user_conversations.html', users=users)
    
    @expose('/<int:user_id>')
    def user_conversations(self, user_id):
        user = User.query.get_or_404(user_id)
        conversations = Conversation.query.filter_by(user_id=user_id).all()
        
        all_messages = []
        for conv in conversations:
            all_messages.extend(conv.messages)
        
        all_messages.sort(key=attrgetter('timestamp'))
        
        grouped_messages = groupby(all_messages, key=lambda m: m.timestamp.date())
        
        grouped_conversations = {date: list(messages) for date, messages in grouped_messages}
        
        return self.render('admin/user_conversation_details.html', user=user, grouped_conversations=grouped_conversations)

admin.add_view(UserConversationsView(name='User Conversations', endpoint='user_conversations'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
