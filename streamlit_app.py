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
    page_title="🎯 Simple Classification Leaderboard",
    page_icon="🎯",
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
    
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .rank-1 {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4a 100%);
        color: #92400e;
        padding: 0.5rem;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
    }
    
    .rank-2 {
        background: linear-gradient(135deg, #c0c0c0 0%, #e5e7eb 100%);
        color: #374151;
        padding: 0.5rem;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
    }
    
    .rank-3 {
        background: linear-gradient(135deg, #cd7f32 0%, #d97706 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 50%;
        text-align: center;
        font-weight: bold;
    }
    
    .stAlert > div {
        border-radius: 10px;
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

# HuggingFace API 클래스 (분류 문제용)
class HuggingFaceClassifier:
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
                f"{self.api_url}distilbert-base-uncased",
                headers=self.headers,
                json={"inputs": "Hello"},
                timeout=10
            )
            return response.status_code in [200, 503]  # 503은 모델 로딩 중
        except:
            return False
    
    def query_model(self, model_name, inputs, max_retries=3):
        """모델 쿼리 (재시도 로직 포함)"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "inputs": inputs,
                    "parameters": {
                        "max_new_tokens": 50,
                        "temperature": 0.1,
                        "do_sample": False,
                        "return_full_text": False
                    },
                    "options": {
                        "wait_for_model": True
                    }
                }
                
                response = requests.post(
                    f"{self.api_url}{model_name}",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        st.info(f"모델 로딩 중... ({attempt + 1}/{max_retries})")
                        time.sleep(20)
                        continue
                    return {"error": "Model is still loading"}
                else:
                    return {"error": f"API Error: {response.status_code} - {response.text}"}
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    st.warning(f"타임아웃... 재시도 중 ({attempt + 1}/{max_retries})")
                    time.sleep(10)
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
    
    def classify_text(self, model_name, text, classes, progress_callback=None):
        """텍스트 분류 수행"""
        results = []
        total_texts = len(text)
        
        # 모델 정보 확인
        model_info = self.get_model_info(model_name)
        if not model_info:
            return [{"error": f"Model {model_name} not found or not accessible"}]
        
        for i, (txt, correct_class) in enumerate(zip(text, classes)):
            if progress_callback:
                progress_callback(i + 1, total_texts)
            
            # 분류를 위한 프롬프트 생성
            prompt = f"""Classify the following text into one of these categories: {', '.join(set(classes))}

Text: "{txt}"

Classification:"""
            
            # API 호출
            response = self.query_model(model_name, prompt)
            
            predicted_class = "Unknown"
            error_msg = None
            
            if isinstance(response, dict) and "error" in response:
                error_msg = response["error"]
                predicted_class = "ERROR"
            elif isinstance(response, list) and len(response) > 0:
                if 'generated_text' in response[0]:
                    generated_text = response[0]['generated_text'].strip()
                    
                    # 가능한 클래스 중에서 가장 유사한 것 찾기
                    unique_classes = list(set(classes))
                    for cls in unique_classes:
                        if cls.lower() in generated_text.lower():
                            predicted_class = cls
                            break
                    
                    # 여전히 찾지 못했다면 첫 번째 단어 사용
                    if predicted_class == "Unknown" and generated_text:
                        first_word = generated_text.split()[0] if generated_text.split() else ""
                        # 가장 유사한 클래스 찾기
                        for cls in unique_classes:
                            if first_word.lower() in cls.lower() or cls.lower() in first_word.lower():
                                predicted_class = cls
                                break
            
            results.append({
                'text': txt,
                'true_class': correct_class,
                'predicted_class': predicted_class,
                'correct': predicted_class == correct_class,
                'raw_response': response,
                'error': error_msg
            })
            
            # API 호출 간격 조절
            time.sleep(3)
        
        return results
    
    def calculate_metrics(self, results):
        """분류 메트릭 계산"""
        valid_results = [r for r in results if not r.get('error')]
        correct = sum(1 for r in valid_results if r['correct'])
        total = len(valid_results)
        
        accuracy = correct / total if total > 0 else 0
        
        # 클래스별 정밀도, 재현율 계산
        classes = list(set(r['true_class'] for r in valid_results))
        precision_sum = 0
        recall_sum = 0
        
        for cls in classes:
            tp = sum(1 for r in valid_results if r['true_class'] == cls and r['predicted_class'] == cls)
            fp = sum(1 for r in valid_results if r['true_class'] != cls and r['predicted_class'] == cls)
            fn = sum(1 for r in valid_results if r['true_class'] == cls and r['predicted_class'] != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision_sum += precision
            recall_sum += recall
        
        avg_precision = precision_sum / len(classes) if classes else 0
        avg_recall = recall_sum / len(classes) if classes else 0
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'total_samples': total,
            'correct_predictions': correct,
            'score': accuracy
        }

# 헤더
st.markdown("""
<div class="main-header">
    <h1>🎯 Simple Classification Leaderboard</h1>
    <p>간단한 텍스트 분류 문제로 AI 모델 성능 평가</p>
</div>
""", unsafe_allow_html=True)

# 사이드바 - 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 설정
    st.subheader("🔑 HuggingFace API Key")
    api_key = st.text_input(
        "API Key를 입력하세요:",
        type="password",
        value=st.session_state.api_key,
        help="https://huggingface.co/settings/tokens 에서 발급받으세요"
    )
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            classifier = HuggingFaceClassifier(api_key)
            if classifier.test_api_connection():
                st.success("✅ API Key가 설정되고 연결이 확인되었습니다!")
            else:
                st.error("❌ API Key가 유효하지 않거나 연결에 실패했습니다")
        else:
            st.warning("⚠️ API Key를 입력해주세요")
    
    st.divider()
    
    # 데이터셋 업로드
    st.subheader("📊 데이터셋 업로드")
    
    # 샘플 CSV 생성
    sample_csv = """text,label
"I love this movie! It's amazing!",positive
"This movie is terrible and boring.",negative
"The weather is sunny today.",neutral
"I hate waiting in long lines.",negative
"This book is fantastic and well-written.",positive
"The service was okay, nothing special.",neutral
"I'm so excited about the concert!",positive
"The food was cold and tasteless.",negative
"It's a normal day at work.",neutral
"This product exceeded my expectations!",positive
"I'm disappointed with the quality.",negative
"The presentation was informative.",neutral
"I can't wait for the weekend!",positive
"The traffic is really bad today.",negative
"The weather is nice for a walk.",positive
"""
    
    st.download_button(
        label="📥 감정 분류 샘플 CSV 다운로드",
        data=sample_csv,
        file_name="sentiment_classification_sample.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요:",
        type=['csv'],
        help="text, label 컬럼이 필요합니다 (감정분류, 주제분류 등)"
    )
    
    if uploaded_file is not None:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            df = pd.read_csv(stringio)
            
            # 필수 컬럼 확인
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
            else:
                # 빈 데이터 필터링
                df_clean = df.dropna(subset=required_columns)
                
                if len(df_clean) == 0:
                    st.error("❌ 유효한 데이터가 없습니다.")
                else:
                    st.session_state.current_dataset = df_clean
                    st.success(f"✅ 데이터셋 로드 완료! ({len(df_clean)}개 샘플)")
                    
                    # 클래스 분포 표시
                    class_counts = df_clean['label'].value_counts()
                    st.write("**클래스 분포:**")
                    for cls, count in class_counts.items():
                        st.write(f"- {cls}: {count}개")
                    
                    if len(df_clean) < len(df):
                        st.warning(f"⚠️ {len(df) - len(df_clean)}개의 불완전한 행이 제외되었습니다.")
        
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {str(e)}")
    
    st.divider()
    
    # 모델 추가
    st.subheader("🤖 모델 추가")
    
    # 추천 모델들
    if st.session_state.api_key:
        with st.expander("🌟 추천 분류 모델"):
            recommended_models = [
                ("google/gemma-2b-it", "Gemma 2B Instruct"),
                ("microsoft/DialoGPT-medium", "DialoGPT Medium"),
                ("distilgpt2", "DistilGPT-2"),
                ("gpt2", "GPT-2"),
                ("microsoft/DialoGPT-small", "DialoGPT Small"),
                ("EleutherAI/gpt-neo-125M", "GPT-Neo 125M")
            ]
            
            for model_id, description in recommended_models:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{description}**")
                    st.caption(model_id)
                with col2:
                    if st.button("➕", key=f"rec_{model_id}"):
                        if model_id not in [m['id'] for m in st.session_state.models_to_evaluate]:
                            st.session_state.models_to_evaluate.append({
                                'id': model_id,
                                'name': description
                            })
                            st.success(f"✅ {description} 추가됨!")
                            st.rerun()
                        else:
                            st.warning("이미 추가된 모델입니다")
    
    with st.form("add_model_form"):
        model_id = st.text_input(
            "HuggingFace Model ID:",
            placeholder="google/gemma-2b-it",
            help="예: google/gemma-2b-it"
        )
        
        display_name = st.text_input(
            "Display Name (선택사항):",
            placeholder="사용자 정의 이름"
        )
        
        submitted = st.form_submit_button("➕ 모델 추가")
        
        if submitted and model_id:
            if '/' not in model_id and not model_id in ['gpt2', 'distilgpt2']:
                st.error("❌ 올바른 모델 ID 형식이 아닙니다. (예: organization/model-name)")
            else:
                if not display_name:
                    display_name = model_id.split('/')[-1].replace('-', ' ').title()
                
                existing_ids = [m['id'] for m in st.session_state.models_to_evaluate]
                if model_id in existing_ids:
                    st.error("❌ 이미 추가된 모델입니다.")
                else:
                    st.session_state.models_to_evaluate.append({
                        'id': model_id,
                        'name': display_name
                    })
                    st.success(f"✅ {display_name} 모델이 추가되었습니다!")
                    st.rerun()
    
    # 추가된 모델 목록
    if st.session_state.models_to_evaluate:
        st.subheader("📝 평가 대기 모델")
        for i, model in enumerate(st.session_state.models_to_evaluate):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"• {model['name']}")
                st.caption(model['id'])
            with col2:
                if st.button("🗑️", key=f"remove_{i}"):
                    st.session_state.models_to_evaluate.pop(i)
                    st.rerun()

# 메인 컨텐츠
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🏆 분류 성능 리더보드")
    
    if not st.session_state.leaderboard_data:
        st.info("📊 아직 평가된 모델이 없습니다. 모델을 추가하고 평가를 시작해보세요!")
    else:
        # 리더보드 데이터프레임 생성
        df_leaderboard = pd.DataFrame(st.session_state.leaderboard_data)
        df_leaderboard = df_leaderboard.sort_values('score', ascending=False).reset_index(drop=True)
        
        # 리더보드 테이블
        for idx, row in df_leaderboard.iterrows():
            rank = idx + 1
            
            # 랭크 아이콘
            if rank == 1:
                rank_badge = "🥇"
            elif rank == 2:
                rank_badge = "🥈"
            elif rank == 3:
                rank_badge = "🥉"
            else:
                rank_badge = f"#{rank}"
            
            with st.container():
                col_rank, col_model, col_metrics, col_actions = st.columns([1, 3, 4, 1])
                
                with col_rank:
                    st.markdown(f"<h3 style='text-align: center;'>{rank_badge}</h3>", unsafe_allow_html=True)
                
                with col_model:
                    st.subheader(row['modelName'])
                    st.caption(row['modelId'])
                    st.caption(f"평가일: {row['evaluatedAt']}")
                
                with col_metrics:
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("정확도", f"{row['accuracy']:.1%}")
                        st.metric("F1 Score", f"{row['f1_score']:.1%}")
                    with metric_col2:
                        st.metric("정밀도", f"{row['precision']:.1%}")
                        st.metric("재현율", f"{row['recall']:.1%}")
                
                with col_actions:
                    if st.button("🗑️ 삭제", key=f"delete_{idx}"):
                        st.session_state.leaderboard_data = [
                            item for item in st.session_state.leaderboard_data 
                            if item['modelId'] != row['modelId']
                        ]
                        st.rerun()
                
                st.divider()

with col2:
    st.header("📊 현재 데이터셋")
    
    if st.session_state.current_dataset is not None:
        st.success(f"✅ {len(st.session_state.current_dataset)}개 샘플 로드됨")
        
        # 데이터셋 미리보기
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(st.session_state.current_dataset.head(10))
    else:
        st.warning("⚠️ 데이터셋을 업로드해주세요")
    
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
            missing.append("API Key")
        if st.session_state.current_dataset is None:
            missing.append("데이터셋")
        if not st.session_state.models_to_evaluate:
            missing.append("평가할 모델")
        
        st.error(f"❌ 다음이 필요합니다: {', '.join(missing)}")
    
    if st.button("🎯 분류 평가 시작", disabled=not can_run, type="primary"):
        if can_run:
            classifier = HuggingFaceClassifier(st.session_state.api_key)
            
            # 프로그레스 바
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 데이터 준비
            texts = st.session_state.current_dataset['text'].tolist()
            labels = st.session_state.current_dataset['label'].tolist()
            
            total_models = len(st.session_state.models_to_evaluate)
            
            try:
                for model_idx, model in enumerate(st.session_state.models_to_evaluate):
                    status_text.text(f"평가 중: {model['name']} ({model_idx + 1}/{total_models})")
                    
                    # 모델 정보 확인
                    model_info = classifier.get_model_info(model['id'])
                    
                    if not model_info:
                        st.error(f"❌ 모델 '{model['id']}'를 찾을 수 없습니다.")
                        continue
                    
                    pipeline_tag = model_info.get('pipeline_tag', 'unknown')
                    st.info(f"📋 모델 정보: {model['name']} (Pipeline: {pipeline_tag})")
                    
                    def progress_callback(current, total):
                        overall_progress = (model_idx + current/total) / total_models
                        progress_bar.progress(overall_progress)
                        status_text.text(f"평가 중: {model['name']} - {current}/{total} 샘플 처리 중")
                    
                    # 분류 평가 실행
                    results = classifier.classify_text(
                        model['id'],
                        texts,
                        labels,
                        progress_callback
                    )
                    
                    # 오류 확인
                    error_count = sum(1 for r in results if r.get('error'))
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count}개 샘플에서 오류 발생")
                    
                    # 메트릭 계산
                    metrics = classifier.calculate_metrics(results)
                    
                    # 결과 표시
                    st.success(f"✅ {model['name']} 평가 완료: {metrics['accuracy']:.1%} 정확도")
                    
                    # 결과 저장
                    new_entry = {
                        'modelId': model['id'],
                        'modelName': model['name'],
                        'accuracy': metrics['accuracy'],
                        'f1_score': metrics['f1_score'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'score': metrics['score'],
                        'total_samples': metrics['total_samples'],
                        'correct_predictions': metrics['correct_predictions'],
                        'evaluatedAt': datetime.now().strftime('%Y-%m-%d'),
                        'pipeline_tag': pipeline_tag
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
                status_text.text("✅ 모든 모델 평가 완료!")
                st.success(f"🎉 {total_models}개 모델 평가가 완료되었습니다!")
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 평가 중 오류가 발생했습니다: {str(e)}")
                status_text.text("❌ 평가 실패")

# 성능 차트
if st.session_state.leaderboard_data:
    st.header("📈 성능 비교 차트")
    
    df_chart = pd.DataFrame(st.session_state.leaderboard_data)
    
    # 성능 비교 차트
    fig = px.bar(
        df_chart.sort_values('score', ascending=True),
        x='score',
        y='modelName',
        orientation='h',
        title="모델별 분류 정확도 비교",
        labels={'score': '정확도', 'modelName': '모델명'},
        color='score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 메트릭 비교
    metrics_df = df_chart[['modelName', 'accuracy', 'f1_score', 'precision', 'recall']].melt(
        id_vars=['modelName'],
        var_name='metric',
        value_name='value'
    )
    
    fig2 = px.line(
        metrics_df,
        x='modelName',
        y='value',
        color='metric',
        title="메트릭별 성능 비교",
        markers=True
    )
    
    fig2.update_layout(height=400)
    fig2.update_xaxis(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

# 푸터
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎯 Simple Classification Leaderboard</p>
    <p>텍스트 분류 문제로 AI 모델 성능을 간단하게 평가합니다</p>
</div>
""", unsafe_allow_html=True)