import streamlit as st
import pandas as pd
import re
import time
from io import StringIO, BytesIO
from openai import OpenAI
from datetime import datetime
import json
import zipfile

# --- 페이지 설정 ---
st.set_page_config(
    page_title="🏥 MedLLM Benchmark",
    page_icon="🏥",
    layout="wide",
)

# --- 심플한 CSS 스타일링 ---
st.markdown("""
<style>
    /* 전체 배경 밝게 */
    .main {
        background-color: #ffffff;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 메인 헤더 */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #4285f4 0%, #0f9d58 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* 제출 섹션 */
    .submit-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* 입력과 버튼 같은 높이 */
    .input-row {
        display: flex;
        gap: 12px;
        align-items: flex-end;
    }
    
    .input-field {
        flex: 1;
    }
    
    .submit-btn {
        flex-shrink: 0;
    }
    
    /* 벤치마크 카드 */
    .benchmark-item {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .rank-badge {
        background: #4285f4;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .model-info {
        flex: 1;
    }
    
    .model-name a {
        color: #1a73e8;
        text-decoration: none;
        font-weight: 600;
        transition: color 0.2s ease;
    }
    
    .model-name a:hover {
        color: #1557b0;
        text-decoration: underline;
    }
    
    .model-details {
        font-size: 0.9rem;
        color: #333333;
    }
    
    .accuracy-display {
        text-align: center;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        min-width: 120px;
    }
    
    .accuracy-score {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .accuracy-high { color: #0f9d58; }
    .accuracy-medium { color: #ff9800; }
    .accuracy-low { color: #ea4335; }
    
    .accuracy-detail {
        font-size: 0.8rem;
        color: #333333;
    }
    
    /* 관리자 버튼 */
    .admin-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    
    /* 심플한 버튼 스타일 */
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #dadce0;
        background: white;
        color: #3c4043;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-color: #d2e3fc;
    }
    
    .stButton > button[kind="primary"] {
        background: #1a73e8;
        color: white;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #1557b0;
    }
    
    /* 큐 정보 */
    .queue-info {
        background: #e8f0fe;
        border-left: 4px solid #1a73e8;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 세션 상태 초기화 ---
if 'benchmark_results' not in st.session_state:
    # 샘플 벤치마크 결과 생성 (Qwen3 30B 이하 모델들)
    st.session_state.benchmark_results = [
        {
            'id': 1,
            'model_name': 'Qwen/Qwen3-14B',
            'accuracy': 86.7,
            'total_questions': 240,
            'correct_answers': 208,
            'evaluation_date': '2025-01-25 14:30:22',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 1
        },
        {
            'id': 2,
            'model_name': 'Qwen/Qwen3-8B',
            'accuracy': 82.1,
            'total_questions': 240,
            'correct_answers': 197,
            'evaluation_date': '2025-01-24 16:45:11',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 2
        },
        {
            'id': 3,
            'model_name': 'Qwen/Qwen3-4B',
            'accuracy': 77.5,
            'total_questions': 240,
            'correct_answers': 186,
            'evaluation_date': '2025-01-23 09:15:33',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 1
        },
        {
            'id': 4,
            'model_name': 'Qwen/Qwen3-1.7B',
            'accuracy': 71.3,
            'total_questions': 240,
            'correct_answers': 171,
            'evaluation_date': '2025-01-22 11:20:45',
            'dataset_name': 'Medical QA Dataset v1.1',
            'api_errors': 4
        },
        {
            'id': 5,
            'model_name': 'Qwen/Qwen3-0.6B',
            'accuracy': 63.8,
            'total_questions': 240,
            'correct_answers': 153,
            'evaluation_date': '2025-01-21 13:50:17',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 7
        }
    ]

if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

if 'test_dataset' not in st.session_state:
    st.session_state.test_dataset = None

if 'pending_evaluations' not in st.session_state:
    st.session_state.pending_evaluations = []

# --- HuggingFace OpenAI 호환 API 클래스 ---
class MedicalEvaluator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Hugging Face API 키가 필요합니다.")
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )

    def query_model(self, model_name, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API_ERROR: {str(e)}"

    def evaluate(self, model_name, questions, correct_answers, progress_bar):
        results = []
        total_questions = len(questions)
        api_errors = 0

        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            prompt = f"""{question}\n\nAnswer with the letter of the correct option (A, B, C, or D).\n\nAnswer:"""
            
            response_text = self.query_model(model_name, prompt)
            
            predicted_answer = "EXTRACT_FAIL"
            error_message = None

            if response_text.startswith("API_ERROR:"):
                error_message = response_text
                predicted_answer = "API_ERROR"
                api_errors += 1
            else:
                match = re.search(r'\b([A-D])\b', response_text.upper())
                if match:
                    predicted_answer = match.group(1)

            is_correct = predicted_answer == correct_answer
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'raw_response': response_text,
                'error': error_message
            })
            
            progress_bar.progress((i + 1) / total_questions, text=f"Processing {i+1}/{total_questions}")
            time.sleep(0.2)

        return results, api_errors

def get_accuracy_class(accuracy):
    if accuracy >= 85:
        return "accuracy-high"
    elif accuracy >= 70:
        return "accuracy-medium"
    else:
        return "accuracy-low"

def add_benchmark_result(model_name, accuracy, total_questions, correct_answers, dataset_name, api_errors):
    new_id = max([r['id'] for r in st.session_state.benchmark_results]) + 1 if st.session_state.benchmark_results else 1
    new_result = {
        'id': new_id,
        'model_name': model_name,
        'accuracy': accuracy,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_name': dataset_name,
        'api_errors': api_errors
    }
    st.session_state.benchmark_results.append(new_result)

def add_evaluation_request(model_name):
    if model_name not in [req['model_name'] for req in st.session_state.pending_evaluations]:
        request_id = len(st.session_state.pending_evaluations) + 1
        st.session_state.pending_evaluations.append({
            'id': request_id,
            'model_name': model_name,
            'status': 'pending',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluated_at': None
        })
        return True
    return False

def load_secure_dataset(uploaded_file, password=None):
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'csv':
            if password:
                try:
                    with zipfile.ZipFile(BytesIO(uploaded_file.read()), 'r') as zip_file:
                        zip_file.setpassword(password.encode())
                        csv_files = [f for f in zip_file.namelist() if f.lower().endswith('.csv')]
                        if not csv_files:
                            return False, "ZIP 파일 내에 CSV 파일이 없습니다."
                        
                        with zip_file.open(csv_files[0]) as csv_file:
                            df = pd.read_csv(csv_file)
                except Exception as e:
                    return False, f"암호화된 ZIP 파일 읽기 실패: {str(e)}"
            else:
                df = pd.read_csv(uploaded_file)
                
        elif file_extension in ['xlsx', 'xls']:
            if password:
                return False, "암호화된 엑셀 파일을 처리하려면 msoffcrypto 라이브러리가 필요합니다. 현재 ZIP 암호화된 CSV 파일을 사용해주세요."
            else:
                df = pd.read_excel(uploaded_file)
        else:
            return False, "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
        
        if 'question' not in df.columns or 'answer' not in df.columns:
            return False, "파일에 'question'과 'answer' 컬럼이 있어야 합니다."
        
        st.session_state.test_dataset = {
            'questions': df['question'].tolist(),
            'answers': df['answer'].tolist(),
            'total_count': len(df),
            'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_type': file_extension,
            'is_encrypted': password is not None
        }
        
        encryption_info = " (암호화됨)" if password else ""
        return True, f"{len(df)}개의 질문이 포함된 테스트 데이터셋이 로드되었습니다.{encryption_info}"
        
    except Exception as e:
        return False, f"데이터셋 로드 중 오류: {str(e)}"

def delete_benchmark_result(result_id):
    st.session_state.benchmark_results = [r for r in st.session_state.benchmark_results if r['id'] != result_id]

# --- 메인 헤더 ---
st.markdown('<div class="main-header"><h1>🏥 MedLLM Benchmark Results</h1></div>', unsafe_allow_html=True)

# --- 관리자 버튼 (우상단) ---
if st.button("⚙️", help="Admin", key="admin_toggle"):
    if st.session_state.admin_authenticated:
        st.session_state.admin_mode = not st.session_state.admin_mode
        if not st.session_state.admin_mode:
            st.session_state.admin_authenticated = False
    else:
        st.session_state.admin_mode = True

# --- 관리자 인증 ---
if st.session_state.admin_mode and not st.session_state.admin_authenticated:
    st.subheader("🔐 Admin Authentication")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        password = st.text_input("Password:", type="password")
        col_login, col_cancel = st.columns(2)
        with col_login:
            if st.button("Login", type="primary", use_container_width=True):
                if password == "passpass":
                    st.session_state.admin_authenticated = True
                    st.success("✅ 로그인 성공!")
                    st.rerun()
                else:
                    st.error("❌ 잘못된 비밀번호입니다.")
        with col_cancel:
            if st.button("Cancel", use_container_width=True):
                st.session_state.admin_mode = False
                st.rerun()

# --- 모델 제출 섹션 ---
st.markdown("""
<div class="submit-box">
    <h3 style="color: #202124;">🚀 Submit Your Model for Evaluation</h3>
    <p style="color: #333333; margin-bottom: 1rem;">HuggingFace 모델 주소를 제출하면 자동으로 평가가 진행됩니다.</p>
</div>
""", unsafe_allow_html=True)

# 입력 필드와 버튼
col1, col2 = st.columns([4, 1])
with col1:
    model_input = st.text_input(
        "Model ID",
        placeholder="예: Qwen/Qwen3-30B",
        label_visibility="collapsed"
    )
with col2:
    submit_btn = st.button("📤 Submit", type="primary", use_container_width=True)

if submit_btn and model_input.strip():
    success = add_evaluation_request(model_input.strip())
    if success:
        st.success(f"✅ {model_input} 평가 요청이 제출되었습니다!")
        st.rerun()
    else:
        st.warning("⚠️ 이미 제출된 모델입니다.")
elif submit_btn:
    st.error("❌ 모델 ID를 입력해주세요.")

# --- 대기열 정보 ---
if st.session_state.pending_evaluations:
    pending_count = len([req for req in st.session_state.pending_evaluations if req['status'] == 'pending'])
    if pending_count > 0:
        st.markdown(f"""
        <div class="queue-info">
            🕐 현재 <strong>{pending_count}개</strong>의 모델이 평가 대기중입니다.
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- 벤치마크 결과 ---
# st.subheader("📊 Current Benchmark Rankings")
st.markdown('<h3 style="color: #333333; font-weight: 600; margin: 2rem 0 1rem 0;">📊 Current Benchmark Rankings</h3>', unsafe_allow_html=True)

sorted_results = sorted(st.session_state.benchmark_results, key=lambda x: x['accuracy'], reverse=True)

for i, result in enumerate(sorted_results):
    accuracy_class = get_accuracy_class(result['accuracy'])
    
    col1, col2, col3, col4 = st.columns([0.5, 3.5, 1.5, 0.5])
    
    with col1:
        st.markdown(f'<div class="rank-badge">#{i+1}</div>', unsafe_allow_html=True)
    
    with col2:
        huggingface_url = f"https://huggingface.co/{result['model_name']}"
        st.markdown(f"""
        <div class="benchmark-item">
            <div class="model-info">
                <div class="model-name">
                    <a href="{huggingface_url}" target="_blank" style="color: #1a73e8; text-decoration: none;">
                        🤖 {result['model_name']}
                    </a>
                </div>
                <div class="model-details">
                    {result['dataset_name']} • {result['evaluation_date']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="accuracy-display">
            <div class="accuracy-score {accuracy_class}">{result['accuracy']:.1f}%</div>
            <div class="accuracy-detail">{result['correct_answers']}/{result['total_questions']} 정답</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.admin_mode and st.session_state.admin_authenticated:
            if st.button("🗑️", key=f"del_{result['id']}", help="삭제"):
                delete_benchmark_result(result['id'])
                st.rerun()

# --- 관리자 기능 ---
if st.session_state.admin_mode and st.session_state.admin_authenticated:
    st.markdown("---")
    # st.subheader("🔧 Admin Panel")
    st.markdown('<h3 style="color: #333333; font-weight: 600; margin: 2rem 0 1rem 0;">🔧 Admin Panel</h3>', unsafe_allow_html=True)
    
    # 테스트 데이터셋 관리
    if st.session_state.test_dataset:
        st.success(f"✅ 데이터셋 로드됨: {st.session_state.test_dataset['total_count']}개 질문")
        if st.button("🗑️ Remove Dataset"):
            st.session_state.test_dataset = None
            st.rerun()
    else:
        # st.warning("⚠️ 테스트 데이터셋이 없습니다.")
        st.markdown('<p style="color: #333333; background: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;">⚠️ 테스트 데이터셋이 없습니다.</p>', unsafe_allow_html=True)
        # uploaded_file = st.file_uploader("데이터셋 업로드", type=['csv', 'xlsx'])
        st.markdown('<label style="color: #333333; font-weight: 500;">📁 데이터셋 업로드</label>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx'], label_visibility="collapsed")
        # password = st.text_input("파일 암호 (선택사항)", type="password")
        st.markdown('<label style="color: #333333; font-weight: 500;">🔒 파일 암호 (선택사항)</label>', unsafe_allow_html=True)
        password = st.text_input("", type="password", label_visibility="collapsed")

        if st.button("📁 Load Dataset") and uploaded_file:
            success, message = load_secure_dataset(uploaded_file, password if password else None)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    # 대기열 관리
    if st.session_state.pending_evaluations:
        st.subheader("📋 Pending Queue")
        for req in st.session_state.pending_evaluations:
            if req['status'] == 'pending':
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(f"📦 {req['model_name']}")
                with col2:
                    if st.button("✅", key=f"approve_{req['id']}", help="승인"):
                        req['status'] = 'approved'
                        st.rerun()
                with col3:
                    if st.button("❌", key=f"reject_{req['id']}", help="거부"):
                        st.session_state.pending_evaluations.remove(req)
                        st.rerun()

else:
    if not st.session_state.admin_mode:
        st.markdown('<p style="color: #333333; background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #1a73e8;">💡 새로운 평가를 원하시면 관리자에게 문의하세요.</p>', unsafe_allow_html=True)
