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
from datetime import datetime, timedelta
import click
import os
import base64
import secrets
from ai_gen import OptimizedExampleSelector, get_ai_response
import time  # 시간 측정용

# 환경 변수 로드
load_dotenv()

# Flask 애플리케이션 초기화
app = Flask(__name__)
CORS(app)

# 한국 시간대 설정
KST = timezone('Asia/Seoul')

# 애플리케이션 설정
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# OpenAI 클라이언트 초기화
client = OpenAI()
migrate = Migrate(app, db)
example_selector = OptimizedExampleSelector()

# Flask-Mail 설정
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'mks010103@gmail.com'
app.config['MAIL_PASSWORD'] = 'vhnk zrko wxxt oank'
app.config['MAIL_DEFAULT_SENDER'] = 'mks010103@gmail.com'

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
    messages = db.relationship('Message', backref='user', lazy=True)

    def set_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiration = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()

    def check_reset_token(self, token):
        return (self.reset_token == token and
                self.reset_token_expiration > datetime.utcnow())

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    messages = db.relationship('Message', backref='conversation', lazy=True, order_by="Message.timestamp")

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey('conversation.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(KST))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message_metadata = db.Column(db.JSON, nullable=True)

# MetadataManager 클래스 추가
class MetadataManager:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.metadata = {
            'user_id': user_id,
            'topics': [],
            'sentiment': None,
            'timestamp': datetime.now(KST).isoformat(),
            'interaction_count': 0
        }

    def update_metadata(self, message: str) -> dict:
        """메시지 기반으로 메타데이터 업데이트"""
        # 기존 대화 수 조회
        interaction_count = Message.query.filter_by(user_id=self.user_id).count()
        
        # 메타데이터 업데이트
        self.metadata.update({
            'interaction_count': interaction_count + 1,
            'timestamp': datetime.now(KST).isoformat(),
            'message_length': len(message)
        })
        
        return self.metadata

# 관리자 뷰 설정
class SecureModelView(ModelView):
    form_base_class = SecureForm
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin

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

def get_recent_context(conversation_id, limit=20):
    recent_messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp.desc()).limit(limit).all()
    recent_messages.reverse()
    context = []
    for msg in recent_messages:
        msg_type = "사용자" if msg.is_user else "AI"
        context.append(f"{msg_type}: {msg.content}")
    return "\n".join(context)

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    단계별_시간 = {}
    전체_시작_시간 = time.time()
    메시지_처리_시작_시간 = time.time()
    user_message_content = request.json['message']
    
    try:
        # 1단계: 메타데이터 관리자 초기화
        metadata_manager = MetadataManager(current_user.id)
        
        # 2단계: 대화 세션 확인/생성
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
            
        # 3단계: 메타데이터 업데이트
        시작_시간 = time.time()
        updated_metadata = metadata_manager.update_metadata(user_message_content)
        단계별_시간['2_메타데이터_업데이트'] = time.time() - 시작_시간

        # 4단계: 사용자 메시지 저장
        시작_시간 = time.time()
        user_message = Message(
            conversation_id=active_conversation.id,
            content=user_message_content,
            is_user=True,
            user_id=current_user.id,
            message_metadata=updated_metadata
        )
        db.session.add(user_message)
        db.session.commit()
        단계별_시간['3_메시지_저장'] = time.time() - 시작_시간

        # 5단계: 최근 컨텍스트 가져오기
        시작_시간 = time.time()
        recent_context = get_recent_context(active_conversation.id)
        단계별_시간['4_컨텍스트_조회'] = time.time() - 시작_시간

        # 6단계: RAG + 대화 검색
        시작_시간 = time.time()
        examples_data = example_selector.get_examples_with_context(user_message_content)
        단계별_시간['5_예제_검색'] = time.time() - 시작_시간

        # 7단계: AI 응답 생성
        시작_시간 = time.time()
        ai_response = get_ai_response(
            recent_context=recent_context,
            user_message_content=user_message_content,
            examples_data=examples_data,
            metadata=updated_metadata,
            example_selector=example_selector
        )
        단계별_시간['6_AI_응답_생성'] = time.time() - 시작_시간

        # 8단계: AI 응답 저장
        시작_시간 = time.time()
        ai_message = Message(
            conversation_id=active_conversation.id,
            content=ai_response['message'],
            is_user=False,
            user_id=current_user.id,
            message_metadata=None
        )
        db.session.add(ai_message)
        db.session.commit()
        단계별_시간['7_응답_저장'] = time.time() - 시작_시간

        # 9단계: 벡터 DB에 대화 추가
        example_selector.add_conversation(
            message=user_message_content,
            metadata=updated_metadata
        )

        메시지_처리_총시간 = time.time() - 메시지_처리_시작_시간
        단계별_시간['8_메시지_처리_총시간'] = 메시지_처리_총시간

        return jsonify({
            'message': ai_response['message'],
            'message_id': ai_message.id,
            'success': True,
            'timing': 단계별_시간,
            'metadata': updated_metadata
        })

    except Exception as e:
        db.session.rollback()
        print(f"채팅 처리 중 오류 발생: {str(e)}")
        return jsonify({
            'message': '죄송합니다. 오류가 발생했습니다.', 
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_voice', methods=['POST'])
@login_required
def generate_voice():
    시작_시간 = time.time()
    음성_처리_시간 = {}
    
    try:
        단계_시작 = time.time()
        message_content = request.json['message']
        message_id = request.json['message_id']
        음성_처리_시간['1_요청_처리'] = time.time() - 단계_시작
        
        단계_시작 = time.time()
        speech_response = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=message_content,
            speed=0.9
        )
        음성_처리_시간['2_음성_생성'] = time.time() - 단계_시작
        
        단계_시작 = time.time()
        audio_base64 = base64.b64encode(speech_response.content).decode('utf-8')
        음성_처리_시간['3_인코딩'] = time.time() - 단계_시작
        
        전체_처리_시간 = time.time() - 시작_시간
        음성_처리_시간['4_총_처리_시간'] = 전체_처리_시간
        
        print("\n=== 음성 생성 처리 시간 분석 ===")
        for 단계, 소요시간 in sorted(음성_처리_시간.items(), key=lambda x: int(x[0].split('_')[0])):
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

@app.route('/translate', methods=['POST'])
@login_required
def translate():
    시작_시간 = time.time()
    단계별_시간 = {}

    try:
        단계_시작 = time.time()
        text = request.json['text']
        단계별_시간['1_요청_처리'] = time.time() - 단계_시작

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

        전체_처리_시간 = time.time() - 시작_시간
        단계별_시간['3_총_처리_시간'] = 전체_처리_시간

        print("\n=== 번역 처리 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
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

@app.route('/login', methods=['POST'])
def login():
    """로그인 처리 라우트"""
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        login_user(user, remember=True)
        return jsonify({"success": True, "username": user.username})
    return jsonify({"success": False})

@app.route('/check_login', methods=['GET'])
def check_login():
    """로그인 상태 확인 라우트"""
    if current_user.is_authenticated:
        return jsonify({"logged_in": True, "username": current_user.username})
    return jsonify({"logged_in": False})

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    """로그아웃 처리 라우트"""
    logout_user()
    return jsonify({"success": True})

@app.route('/signup', methods=['POST'])
def signup():
    """회원가입 처리 라우트"""
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

@app.route('/get_history', methods=['GET'])
@login_required
def get_history():
    """사용자의 대화 기록을 가져오는 라우트"""
    시작_시간 = time.time()
    단계별_시간 = {}
    
    try:
        단계_시작 = time.time()
        date = request.args.get('date')
        단계별_시간['1_파라미터_처리'] = time.time() - 단계_시작
        
        단계_시작 = time.time()
        query = Conversation.query.filter_by(user_id=current_user.id)
        if date:
            query = query.filter(Conversation.start_time < datetime.strptime(date, '%Y-%m-%d'))
        conversations = query.order_by(desc(Conversation.start_time)).limit(10).all()
        단계별_시간['2_쿼리_실행'] = time.time() - 단계_시작
        
        단계_시작 = time.time()
        history = []
        for conv in conversations:
            messages = sorted(conv.messages, key=lambda m: m.timestamp)
            grouped_messages = {}
            for msg in messages:
                date_key = msg.timestamp.astimezone(KST).date().strftime('%Y-%m-%d')
                if date_key not in grouped_messages:
                    grouped_messages[date_key] = []
                grouped_messages[date_key].append({
                    'content': msg.content,
                    'is_user': msg.is_user,
                    'timestamp': msg.timestamp.strftime('%H:%M')
                })
            
            for date, msgs in grouped_messages.items():
                history.append({
                    'date': date,
                    'messages': msgs
                })
        단계별_시간['3_데이터_처리'] = time.time() - 단계_시작

        전체_처리_시간 = time.time() - 시작_시간
        단계별_시간['4_총_처리_시간'] = 전체_처리_시간

        print("\n=== 히스토리 조회 시간 분석 ===")
        for 단계, 소요시간 in sorted(단계별_시간.items(), key=lambda x: int(x[0].split('_')[0])):
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
    """사용자의 총 사용 시간을 업데이트하는 라우트"""
    data = request.json
    current_user.total_usage_time += data['time']
    db.session.commit()
    return jsonify({"success": True})

@app.route('/request_reset', methods=['POST'])
def request_reset():
    """비밀번호 재설정 요청을 처리하는 라우트"""
    try:
        email = request.json.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            user.set_reset_token()
            send_reset_email(user)
            return jsonify({"message": "Reset link sent to your email"})
        return jsonify({"message": "Email not found"}), 404
    except Exception as e:
        print(f"Error in request_reset: {str(e)}")
        return jsonify({"message": "An error occurred"}), 500

def send_reset_email(user):
    """비밀번호 재설정 이메일을 보내는 함수"""
    token = user.reset_token
    msg = FlaskMessage(
        subject='Password Reset Request',
        recipients=[user.email],
        body=f'''To reset your password, visit the following link:
{url_for('reset_password_form', token=token, _external=True)}

If you did not make this request then simply ignore this email and no changes will be made.
'''
    )
    mail.send(msg)

@app.route('/reset_password/<token>', methods=['GET'])
def reset_password_form(token):
    """비밀번호 재설정 폼을 표시하는 라우트"""
    user = User.query.filter_by(reset_token=token).first()
    if user and user.check_reset_token(token):
        return render_template('reset_password.html', token=token)
    return "Invalid or expired token", 400

@app.route('/reset_password', methods=['POST'])
def reset_password():
    """비밀번호 재설정을 처리하는 라우트"""
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
    """데이터베이스 백업을 위한 관리자 라우트"""
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
        
        all_messages.sort(key=lambda m: m.timestamp)
        
        grouped_messages = {}
        for msg in all_messages:
            date_key = msg.timestamp.date()
            if date_key not in grouped_messages:
                grouped_messages[date_key] = []
            grouped_messages[date_key].append(msg)
        
        return self.render('admin/user_conversation_details.html', 
                         user=user, 
                         grouped_conversations=grouped_messages)

admin.add_view(UserConversationsView(name='User Conversations', endpoint='user_conversations'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
