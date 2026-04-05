import streamlit as st
import sqlite3
import os
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDF2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- 1. 데이터베이스 초기화 ---
def init_db():
    conn = sqlite3.connect('pharmacy_study.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS review_notes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  drug_name TEXT, 
                  question TEXT, 
                  correct_answer TEXT, 
                  logic_tip TEXT,
                  created_at TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# --- 2. 페이지 설정 및 UI ---
st.set_page_config(page_title="간호 약리학 마스터", layout="wide")
st.title("💊 SN 약리학: 생리 논리 & 오답 노트")

with st.sidebar:
    st.header("⚙️ 설정")
    # Streamlit Secrets나 직접 입력에서 API 키 가져오기
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("전공서적 PDF 업로드", type="pdf")
    st.divider()
    mode = st.radio("학습 모드", ["개념 정리", "실전 퀴즈", "오답 노트 확인"])

# --- 3. 프롬프트 정의 ---
custom_prompt_template = """
당신은 약리학 전문 교수입니다. 반드시 제공된 [학습 자료]를 바탕으로 답변하세요.
1. 생리학적 논리: 기전(MOA)을 생리학적으로 설명하세요.
2. 국시 포인트: 국가고시 빈출 내용과 간호 중재를 포함하세요.
3. 실무 디테일: UpToDate 기반 최신 실무 주의사항을 대조하세요.

질문: {question}
자료: {context}
답변:"""
PROMPT = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# --- 4. RAG 및 기능 구현 ---
if uploaded_file and api_key:
    # PDF 임시 저장
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # RAG 로직
    loader = PyPDF2Loader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(openai_api_key=api_key)
    )
    
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", 
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    if mode == "개념 정리":
        user_input = st.chat_input("약물명을 입력하세요")
        if user_input:
            st.chat_message("user").write(user_input)
            with st.spinner("생리 논리 분석 중..."):
                response = qa_chain.invoke(user_input)
                st.chat_message("assistant").write(response["result"])
            
    elif mode == "실전 퀴즈":
        if st.button("새로운 국시 스타일 문제 생성"):
            with st.spinner("문제 생성 중..."):
                res = qa_chain.invoke("이 단원의 핵심 내용을 바탕으로 간호 국시 스타일 사례형 문제 1개와 정답, 해설을 만들어줘.")
                st.session_state['last_quiz'] = res["result"]
        
        if 'last_quiz' in st.session_state:
            st.markdown(st.session_state['last_quiz'])
            if st.button("오답 노트에 저장"):
                c = conn.cursor()
                c.execute("INSERT INTO review_notes (drug_name, question, correct_answer, logic_tip, created_at) VALUES (?, ?, ?, ?, ?)",
                          ("학습 약물", st.session_state['last_quiz'], "해설 참조", "자료 기반", datetime.now().strftime("%Y-%m-%d")))
                conn.commit()
                st.success("저장되었습니다!")

elif mode == "오답 노트 확인":
    st.subheader("📚 나의 복습 리스트")
    c = conn.cursor()
    c.execute("SELECT * FROM review_notes ORDER BY created_at DESC")
    rows = c.fetchall()
    for row in rows:
        with st.expander(f"[{row[5]}] 복습 문항"):
            st.write(row[2])
            if st.button(f"삭제 {row[0]}"):
                c.execute("DELETE FROM review_notes WHERE id=?", (row[0],))
                conn.commit()
                st.rerun()
else:
    st.info("API 키와 PDF를 업로드하면 학습을 시작할 수 있습니다.")