import streamlit as st
import requests

BACKEND_URL = st.secrets.get("backend_url", "http://127.0.0.1:8030")

st.set_page_config(page_title="ROHIT MLOps Assistant", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = "session_1"

if "history" not in st.session_state:
    st.session_state.history = []

st.title("ROHIT Agentic MLOps Assistant")

session = st.text_input("Session ID", st.session_state.session_id)
st.session_state.session_id = session

question = st.text_input("Ask an MLOps question", "How do I track experiments with MLflow?")

if st.button("Ask") and question:
    payload = {"question": question, "session_id": st.session_state.session_id}
    r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        st.session_state.history.append((question, data.get("answer", "")))
    else:
        st.error(r.text)

if st.button("Crew Ask") and question:
    payload = {"question": question, "session_id": st.session_state.session_id}
    r = requests.post(f"{BACKEND_URL}/crew-ask", json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        st.write("**Analyst**")
        st.write(data.get("analyst", ""))
        st.write("**Explainer**")
        st.write(data.get("explainer", ""))
        st.session_state.history.append((question, data.get("explainer", "")))
    else:
        st.error(r.text)

st.markdown("---")
for i, (q, a) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q{i+1}**: {q}")
    st.markdown(f"**A{i+1}**: {a}")
