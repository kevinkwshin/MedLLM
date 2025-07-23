import streamlit as st
import pandas as pd
import requests
import json
import time
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# 페이지 설정
st.set_page_config(
    page_title="🏥 MedLLM Leaderboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .question-preview {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        white-space: pre-line;
    }
    
    .answer-highlight {
        background: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .rank-badge {
        display: inline-block;
        padding: 0.8rem;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
        min-width: 4rem;
        min-height: 4rem;
        line-height: 2.4rem;
        font-size: 1.1rem;
    }
    
    .rank-1 {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%);
        color: #92400e;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
    }
    
    .rank-2 {
        background: linear-gradient(135deg, #c0c0c0 0%, #e5e7eb 100%);
        color: #374151;
        box-shadow: 0 4px 15px rgba(192, 192, 192, 0.4);
    }
    
    .rank-3 {
        background: linear-gradient(135deg, #cd7f32 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(205, 127, 50, 0.4);
    }
    
    .performance-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'leaderboard_data' not in st.session_state:
    st.session_state.leaderboard_data = []
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'models_to_evaluate' not in st.session_state:
    st.session_state.models_to_evaluate = []

# HuggingFace API 클래스 (간소화된 형식용)
class SimplifiedMedicalQA:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = 'https://api-inference.huggingface.co/models/'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }
    
    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            response = requests.post(
                f"{self.api_url}gpt2",
                headers=self.headers,
                json={"inputs": "Test"},
                timeout=10
            )
            return response.status_code in [200, 503]
        except:
            return False
    
    def query_model(self, model_name, inputs, max_retries=3):
        """모델 쿼리"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": inputs,
                    "parameters": {
                        "max_new_tokens": 50,
                        "temperature": 0.1,
                        "do_sample": False,
                        "return_full_text": False,
                        "stop": ["\n\n", "질문:", "Question:", "다음"]
                    },
                    "options": {
                        "wait_for_model": True
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}{model_name}",
                    headers=self.headers,
                    json=payload,
                    timeout=90
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        st.info(f"⏳ 모델 로딩 중... ({attempt + 1}/{max_retries})")
                        time.sleep(30)
                        continue
                    return {"error": "Model loading timeout"}
                else:
                    return {"error": f"API Error: {response.status_code}"}
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"⏰ 타임아웃... 재시도 중 ({attempt + 1}/{max_retries})")
                    time.sleep(15)
                    continue
                return {"error": "Request timeout"}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Max retries exceeded"}
    
    def get_model_info(self, model_name):
        """모델 정보 확인"""
        try:
            response = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def evaluate_qa_simple(self, model_name, questions, correct_answers, progress_callback=None):
        """간소화된 QA 평가"""
        results = []
        total_questions = len(questions)
        
        # 모델 정보 확인
        model_info = self.get_model_info(model_name)
        if not model_info:
            return [{"error": f"Model {model_name} not found"}]
        
        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            if progress_callback:
                progress_callback(i + 1, total_questions)
            
            # 간단한 프롬프트 (질문에 이미 선택지가 포함됨)
            prompt = f"""{question}

정답을 A, B, C, D 중 하나로만 답하세요.
정답: """
            
            # API 호출
            response = self.query_model(model_name, prompt)
            
            predicted_answer = "No answer"
            error_msg = None
            raw_response_text = ""
            
            if isinstance(response, dict) and "error" in response:
                error_msg = response["error"]
                predicted_answer = "ERROR"
            elif isinstance(response, list) and len(response) > 0:
                if 'generated_text' in response[0]:
                    generated_text = response[0]['generated_text'].strip()
                    raw_response_text = generated_text
                    
                    # 답변 추출 패턴 (더 간단하고 확실한 패턴들)
                    patterns = [
                        r'^([ABCD])[\.\)\s]',      # 첫 글자가 A, B, C, D
                        r'정답[\s:]*([ABCD])',     # 정답: A
                        r'답[\s:]*([ABCD])',       # 답: B  
                        r'([ABCD])[\s]*번',        # A번
                        r'선택[\s:]*([ABCD])',     # 선택: C
                        r'\b([ABCD])\b'            # 단독 A, B, C, D
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, generated_text, re.IGNORECASE)
                        if match:
                            predicted_answer = match.group(1).upper()
                            break
                    
                    # 마지막 시도: 전체 텍스트에서 A,B,C,D 중 첫 번째 찾기
                    if predicted_answer == "No answer":
                        abcd_matches = re.findall(r'[ABCD]', generated_text.upper())
                        if abcd_matches:
                            predicted_answer = abcd_matches[0]
            
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'correct': predicted_answer == correct_answer,
                'raw_response': raw_response_text,
                'error': error_msg
            })
            
            # API 제한 고려
            time.sleep(3)
        
        return results
    
    def calculate_metrics(self, results):
        """메트릭 계산"""
        valid_results = [r for r in results if not r.get('error')]
        correct = sum(1 for r in valid_results if r['correct'])
        total = len(valid_results)
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1_score': accuracy,     # 간소화
            'precision': accuracy,    # 간소화
            'recall': accuracy,       # 간소화
            'total_questions': total,
            'correct_answers': correct,
            'error_count': len(results) - len(valid_results),
            'score': accuracy
        }

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🏥 MedLLM Leaderboard</h1>
    <p>간소화된 한국어 의료 QA로 AI 모델 평가 (Question + Answer 형식)</p>
</div>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 설정
    st.subheader("🔑 HuggingFace API Key")
    api_key = st.text_input(
        "API Key:",
        type="password",
        value=st.session_state.api_key,
        help="https://huggingface.co/settings/tokens"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            evaluator = SimplifiedMedicalQA(api_key)
            if evaluator.test_api_connection():
                st.success("✅ API 연결 성공!")
            else:
                st.error("❌ API 연결 실패")
        else:
            st.warning("⚠️ API Key를 입력해주세요")
    
    st.divider()
    
    # 데이터셋 업로드
    st.subheader("📊 데이터셋 (간소화)")
    
    # 새로운 간소화된 샘플 생성
    simplified_sample = '''question,answer
"25세 여성이 심한 복통과 함께 응급실에 내원했습니다. 마지막 월경은 6주 전이었고, 소변 임신반응 검사는 양성입니다. 혈압은 90/60 mmHg, 맥박은 110회/분입니다. 복부 검사에서 좌하복부에 압통이 있습니다. 가장 가능성이 높은 진단은?

A. 자궁외임신
B. 충수염  
C. 난소낭종 파열
D. 골반염","A"
"45세 남성이 운동 후 심한 흉통을 호소합니다. 통증은 왼쪽 어깨와 팔로 방사되며, 식은땀을 흘리고 있습니다. 심전도에서 V2-V4 유도에서 ST 분절 상승이 관찰됩니다. 가장 적절한 초기 치료는?

A. 아스피린 투여
B. 니트로글리세린 설하정
C. 산소 공급
D. 즉시 심도자술","D"
"60세 남성이 6개월간 체중감소와 황달을 보입니다. 복부 CT에서 췌장 두부에 종괴가 발견되었습니다. CA 19-9 수치가 현저히 상승되어 있습니다. 가장 가능성이 높은 진단은?

A. 췌장염
B. 췌장암
C. 담석증  
D. 간경화","B"'''
    
    st.download_button(
        label="📥 간소화된 의료 QA 샘플",
        data=simplified_sample,
        file_name="simplified_medical_qa.csv",
        mime="text/csv"
    )
    
    st.info("""
    **새로운 CSV 형식:**
    - `question`: 질문 + 선택지 포함
    - `answer`: 정답 (A, B, C, D)
    """)
    
    uploaded_file = st.file_uploader(
        "CSV 파일 업로드:",
        type=['csv'],
        help="question, answer 컬럼만 필요"
    )
    
    if uploaded_file is not None:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            
            # 필수 컬럼 확인
            if 'question' not in df.columns or 'answer' not in df.columns:
                st.error("❌ 'question', 'answer' 컬럼이 필요합니다")
            else:
                # 데이터 정제
                df_clean = df.dropna(subset=['question', 'answer'])
                
                if len(df_clean) == 0:
                    st.error("❌ 유효한 데이터가 없습니다")
                else:
                    st.session_state.current_dataset = df_clean
                    st.success(f"✅ {len(df_clean)}개 문제 로드!")
                    
                    # 답변 분포
                    answer_dist = df_clean['answer'].value_counts()
                    st.write("**정답 분포:**")
                    for ans, count in answer_dist.items():
                        st.write(f"- {ans}: {count}개")
        
        except Exception as e:
            st.error(f"❌ 파일 오류: {str(e)}")
    
    st.divider()
    
    # 모델 추가
    st.subheader("🤖 AI 모델")
    
    # 추천 모델
    if st.session_state.api_key:
        with st.expander("🌟 추천 모델"):
            models = [
                ("google/gemma-2b-it", "Gemma 2B", "⭐⭐⭐"),
                ("microsoft/BioGPT", "BioGPT", "⭐⭐⭐"),
                ("microsoft/DialoGPT-medium", "DialoGPT-M", "⭐⭐"),
                ("distilgpt2", "DistilGPT-2", "⭐⭐"),
                ("gpt2", "GPT-2", "⭐")
            ]
            
            for model_id, name, rating in models:
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{name}** {rating}")
                    st.caption(model_id)
                with col2:
                    pass
                with col3:
                    if st.button("➕", key=f"add_{model_id}"):
                        if model_id not in [m['id'] for m in st.session_state.models_to_evaluate]:
                            st.session_state.models_to_evaluate.append({
                                'id': model_id,
                                'name': name
                            })
                            st.success(f"✅ {name} 추가!")
                            st.rerun()
    
    # 커스텀 모델 추가
    with st.form("add_model"):
        model_id = st.text_input("모델 ID:", placeholder="google/gemma-2b-it")
        model_name = st.text_input("표시 이름:", placeholder="커스텀 이름")
        
        if st.form_submit_button("➕ 모델 추가"):
            if model_id:
                name = model_name if model_name else model_id.split('/')[-1]
                if model_id not in [m['id'] for m in st.session_state.models_to_evaluate]:
                    st.session_state.models_to_evaluate.append({
                        'id': model_id,
                        'name': name
                    })
                    st.success(f"✅ {name} 추가!")
                    st.rerun()
    
    # 추가된 모델 목록
    if st.session_state.models_to_evaluate:
        st.subheader("📝 평가 대기")
        for i, model in enumerate(st.session_state.models_to_evaluate):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"🤖 {model['name']}")
                st.caption(model['id'])
            with col2:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.models_to_evaluate.pop(i)
                    st.rerun()

# 메인 영역
col1, col2 = st.columns([3, 2])

with col1:
    st.header("🏆 성능 리더보드")
    
    if not st.session_state.leaderboard_data:
        st.info("""
        🎯 **간소화된 의료 QA 평가**
        
        1. API Key 설정 ✨
        2. question, answer 형식 CSV 업로드 📊  
        3. AI 모델 추가 🤖
        4. 평가 시작! 🚀
        """)
    else:
        df_results = pd.DataFrame(st.session_state.leaderboard_data)
        df_results = df_results.sort_values('score', ascending=False).reset_index(drop=True)
        
        for idx, row in df_results.iterrows():
            rank = idx + 1
            
            # 순위 배지
            if rank == 1:
                badge = "🥇"
                badge_class = "rank-1"
            elif rank == 2:
                badge = "🥈"
                badge_class = "rank-2"
            elif rank == 3:
                badge = "🥉"
                badge_class = "rank-3"
            else:
                badge = f"#{rank}"
                badge_class = ""
            
            # 성능 카드
            with st.container():
                st.markdown("---")
                col_rank, col_info, col_metrics = st.columns([1, 2, 2])
                
                with col_rank:
                    st.markdown(f'<div class="rank-badge {badge_class}">{badge}</div>', unsafe_allow_html=True)
                
                with col_info:
                    st.subheader(f"🤖 {row['modelName']}")
                    st.caption(f"📍 {row['modelId']}")
                    st.caption(f"📅 {row['evaluatedAt']}")
                    
                    # 성능 등급
                    accuracy = row['accuracy'] * 100
                    if accuracy >= 80:
                        grade = "🟢 A급 (우수)"
                    elif accuracy >= 60:
                        grade = "🟡 B급 (양호)"
                    elif accuracy >= 40:
                        grade = "🟠 C급 (보통)"
                    else:
                        grade = "🔴 D급 (개선필요)"
                    
                    st.markdown(f"**{grade}**")
                
                with col_metrics:
                    st.metric("정확도", f"{row['accuracy']:.1%}")
                    st.metric("정답/전체", f"{row['correct_answers']}/{row['total_questions']}")
                    
                    if row.get('error_count', 0) > 0:
                        st.warning(f"⚠️ {row['error_count']}개 오류")
                
                # 삭제 버튼
                if st.button(f"🗑️ {row['modelName']} 삭제", key=f"delete_{idx}"):
                    st.session_state.leaderboard_data = [
                        item for item in st.session_state.leaderboard_data 
                        if item['modelId'] != row['modelId']
                    ]
                    st.rerun()

with col2:
    st.header("📊 데이터셋 정보")
    
    if st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        st.success(f"✅ **{len(df)}개** 문제 로드됨")
        
        # 답변 분포 차트
        with st.expander("📈 정답 분포", expanded=True):
            answer_counts = df['answer'].value_counts()
            fig = px.pie(
                values=answer_counts.values,
                names=answer_counts.index,
                title="정답 분포"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # 문제 미리보기
        with st.expander("👀 문제 미리보기", expanded=True):
            if len(df) > 0:
                sample_idx = st.selectbox(
                    "문제 선택:",
                    range(len(df)),
                    format_func=lambda x: f"문제 {x+1}"
                )
                
                sample = df.iloc[sample_idx]
                
                st.markdown(f'<div class="question-preview">{sample["question"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-highlight">정답: {sample["answer"]}</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ 데이터셋을 업로드해주세요")
        st.info("""
        **필요한 형식:**
        ```
        question,answer
        "질문 + 선택지","A"
        ```
        """)
    
    st.divider()
    
    # 평가 실행
    st.header("🚀 평가 실행")
    
    can_run = (
        st.session_state.api_key and 
        st.session_state.current_dataset is not None and 
        st.session_state.models_to_evaluate
    )
    
    if not can_run:
        missing = []
        if not st.session_state.api_key:
            missing.append("🔑 API Key")
        if st.session_state.current_dataset is None:
            missing.append("📊 데이터셋")
        if not st.session_state.models_to_evaluate:
            missing.append("🤖 모델")
        
        st.error("**필요한 항목:**\n" + "\n".join([f"- {item}" for item in missing]))
    else:
        num_questions = len(st.session_state.current_dataset)
        num_models = len(st.session_state.models_to_evaluate)
        estimated_minutes = (num_questions * num_models * 3) // 60
        
        st.success("✅ 모든 준비 완료!")
        st.info(f"⏱️ 예상 시간: 약 {estimated_minutes}분")
        st.caption(f"📊 {num_models}개 모델 × {num_questions}개 문제")
    
    if st.button("🎯 평가 시작!", disabled=not can_run, type="primary", use_container_width=True):
        if can_run:
            evaluator = SimplifiedMedicalQA(st.session_state.api_key)
            
            # 진행 상황 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            current_model_info = st.empty()
            
            # 데이터 준비
            df = st.session_state.current_dataset
            questions = df['question'].tolist()
            correct_answers = df['answer'].tolist()
            
            total_models = len(st.session_state.models_to_evaluate)
            
            try:
                for model_idx, model in enumerate(st.session_state.models_to_evaluate):
                    current_model_info.info(f"🔄 **{model['name']}** 평가 중... ({model_idx + 1}/{total_models})")
                    
                    # 모델 정보 확인
                    model_info = evaluator.get_model_info(model['id'])
                    if not model_info:
                        st.error(f"❌ '{model['id']}' 모델을 찾을 수 없습니다")
                        continue
                    
                    def progress_callback(current, total):
                        overall_progress = (model_idx + current/total) / total_models
                        progress_bar.progress(overall_progress)
                        status_text.text(f"📝 문제 {current}/{total} 처리 중...")
                    
                    # 평가 실행
                    results = evaluator.evaluate_qa_simple(
                        model['id'],
                        questions,
                        correct_answers,
                        progress_callback
                    )
                    
                    # 메트릭 계산
                    metrics = evaluator.calculate_metrics(results)
                    
                    # 결과 표시
                    accuracy_pct = metrics['accuracy'] * 100
                    if accuracy_pct >= 70:
                        result_emoji = "🎉"
                    elif accuracy_pct >= 50:
                        result_emoji = "👍"
                    else:
                        result_emoji = "💪"
                    
                    st.success(f"{result_emoji} **{model['name']}** 완료: {accuracy_pct:.1f}% ({metrics['correct_answers']}/{metrics['total_questions']})")
                    
                    # 결과 저장
                    new_entry = {
                        'modelId': model['id'],
                        'modelName': model['name'],
                        'accuracy': metrics['accuracy'],
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'score': metrics['score'],
                        'total_questions': metrics['total_questions'],
                        'correct_answers': metrics['correct_answers'],
                        'error_count': metrics['error_count'],
                        'evaluatedAt': datetime.now().strftime('%Y-%m-%d %H:%M'),
                        'pipeline_tag': model_info.get('pipeline_tag', 'unknown')
                    }
                    
                    # 기존 결과 제거 후 새 결과 추가
                    st.session_state.leaderboard_data = [
                        item for item in st.session_state.leaderboard_data 
                        if item['modelId'] != model['id']
                    ]
                    st.session_state.leaderboard_data.append(new_entry)
                
                # 평가 완료
                st.session_state.models_to_evaluate = []
                progress_bar.progress(1.0)
                status_text.text("✅ 모든 평가 완료!")
                current_model_info.success(f"🎉 **{total_models}개 모델** 평가 완료!")
                
                # 최고 성능 모델 표시
                if st.session_state.leaderboard_data:
                    best_model = max(st.session_state.leaderboard_data, key=lambda x: x['score'])
                    st.balloons()
                    st.info(f"🏆 **최고 성능:** {best_model['modelName']} ({best_model['accuracy']:.1%} 정확도)")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 평가 중 오류: {str(e)}")
                status_text.text("❌ 평가 실패")

# 성능 분석 차트
if st.session_state.leaderboard_data:
    st.header("📈 성능 분석")
    
    df_analysis = pd.DataFrame(st.session_state.leaderboard_data)
    
    # 성능 비교 차트들
    col1, col2 = st.columns(2)
    
    with col1:
        # 정확도 바 차트
        fig1 = px.bar(
            df_analysis.sort_values('accuracy', ascending=True),
            x='accuracy',
            y='modelName',
            orientation='h',
            title="🎯 모델별 정확도 비교",
            labels={'accuracy': '정확도', 'modelName': '모델'},
            color='accuracy',
            color_continuous_scale='RdYlGn',
            text='accuracy'
        )
        fig1.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # 정답 수 산점도
        fig2 = px.scatter(
            df_analysis,
            x='total_questions',
            y='correct_answers',
            size='accuracy',
            color='accuracy',
            hover_name='modelName',
            title="📊 정답 수 분포",
            labels={'total_questions': '전체 문제', 'correct_answers': '정답 수'},
            color_continuous_scale='RdYlGn'
        )
        
        # 완벽한 성능 라인 추가
        if len(df_analysis) > 0:
            max_questions = df_analysis['total_questions'].max()
            fig2.add_shape(
                type="line",
                x0=0, y0=0, x1=max_questions, y1=max_questions,
                line=dict(color="gray", dash="dash"),
                name="완벽한 성능"
            )
        
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # 성능 등급별 분류
    st.subheader("🏅 성능 등급 분류")
    
    # 등급별로 그룹화
    grade_data = []
    for _, row in df_analysis.iterrows():
        accuracy = row['accuracy'] * 100
        if accuracy >= 80:
            grade = "A급 (우수)"
            color = "🟢"
        elif accuracy >= 60:
            grade = "B급 (양호)"
            color = "🟡"
        elif accuracy >= 40:
            grade = "C급 (보통)"
            color = "🟠"
        else:
            grade = "D급 (개선필요)"
            color = "🔴"
        
        grade_data.append({
            'model': row['modelName'],
            'accuracy': accuracy,
            'grade': grade,
            'color': color,
            'correct': row['correct_answers'],
            'total': row['total_questions']
        })
    
    # 등급별로 정렬 (A급부터)
    grade_order = {"A급 (우수)": 4, "B급 (양호)": 3, "C급 (보통)": 2, "D급 (개선필요)": 1}
    grade_data.sort(key=lambda x: (grade_order.get(x['grade'], 0), x['accuracy']), reverse=True)
    
    for item in grade_data:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        
        with col1:
            st.write(f"**{item['model']}**")
        with col2:
            st.write(f"{item['color']} **{item['grade']}**")
        with col3:
            st.write(f"**{item['accuracy']:.1f}%** ({item['correct']}/{item['total']})")
        with col4:
            # 간단한 성능 게이지
            if item['accuracy'] >= 80:
                st.success("✨")
            elif item['accuracy'] >= 60:
                st.info("👍")
            elif item['accuracy'] >= 40:
                st.warning("🤔")
            else:
                st.error("💪")
    
    # 상세 통계
    st.subheader("📋 상세 통계")
    
    avg_accuracy = df_analysis['accuracy'].mean()
    best_accuracy = df_analysis['accuracy'].max()
    worst_accuracy = df_analysis['accuracy'].min()
    total_evaluated = len(df_analysis)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("평균 정확도", f"{avg_accuracy:.1%}")
    with col2:
        st.metric("최고 정확도", f"{best_accuracy:.1%}")
    with col3:
        st.metric("최저 정확도", f"{worst_accuracy:.1%}")
    with col4:
        st.metric("평가된 모델", f"{total_evaluated}개")
    
    # 성능 히스토그램
    fig3 = px.histogram(
        df_analysis,
        x='accuracy',
        nbins=10,
        title="📊 정확도 분포",
        labels={'accuracy': '정확도', 'count': '모델 수'},
        color_discrete_sequence=['#3b82f6']
    )
    fig3.update_layout(height=300)
    st.plotly_chart(fig3, use_container_width=True)

# 도움말 및 팁
with st.expander("💡 사용 팁 & 도움말"):
    st.markdown("""
    ### 🎯 효과적인 평가를 위한 팁
    
    **1. 데이터셋 준비**
    - 질문에 선택지(A, B, C, D)를 포함시키세요
    - 명확하고 구체적인 질문을 작성하세요
    - 정답은 반드시 A, B, C, D 중 하나로 설정하세요
    
    **2. 모델 선택**
    - **의료 특화**: `microsoft/BioGPT` (추천)
    - **범용 우수**: `google/gemma-2b-it` (추천)
    - **빠른 테스트**: `distilgpt2`, `gpt2`
    
    **3. 평가 결과 해석**
    - **80% 이상**: 우수한 의료 지식
    - **60-80%**: 양호한 성능
    - **40-60%**: 기본적인 이해 수준
    - **40% 미만**: 추가 학습 필요
    
    **4. 문제 해결**
    - API 오류 시: 잠시 기다린 후 재시도
    - 모델 로딩 중: 30초 정도 대기 필요
    - 답변 추출 실패: 질문 형식 확인
    
    ### 📄 CSV 형식 예시
    ```csv
    question,answer
    "환자의 증상은 다음과 같습니다...
    
    A. 진단 1
    B. 진단 2  
    C. 진단 3
    D. 진단 4","A"
    ```
    """)

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <h4>🏥 MedLLM Leaderboard (Simplified)</h4>
    <p><strong>간소화된 Question + Answer 형식으로 의료 AI 모델 평가</strong></p>
    <p>📊 더 간단하고 직관적인 데이터 구조 | 🚀 빠른 평가 및 비교</p>
    <p><em>※ 연구 및 교육 목적으로만 사용하세요. 실제 의료 진단에는 사용 금지.</em></p>
</div>
""", unsafe_allow_html=True)