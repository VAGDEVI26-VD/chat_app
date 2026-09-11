import streamlit as st
import requests

st.title("📄 Chat With Your Docs (RAG)")

BACKEND_URL = "http://localhost:8000"

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

uploaded_file = st.file_uploader("Upload TXT or PDF", type=["txt", "pdf"])

# ---------- UPLOAD ----------
if uploaded_file:
    try:
        res = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": uploaded_file}
        )

        if res.status_code != 200:
            st.error("Backend error during upload")
        else:
            data = res.json()

            if "document_id" in data:
                st.session_state.doc_id = data["document_id"]
                st.success("Document uploaded successfully!")
            else:
                st.error(f"Upload failed: {data}")

    except Exception as e:
        st.error(f"Connection error: {e}")

# ---------- QUESTION ----------
question = st.text_input("Ask a question")

if st.button("Ask"):
    if not st.session_state.doc_id:
        st.warning("Please upload a document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            res = requests.post(
                f"{BACKEND_URL}/ask",
                json={
                    "document_id": st.session_state.doc_id,
                    "question": question
                }
            )

            if res.status_code != 200:
                st.error("Backend error while answering")
            else:
                data = res.json()

                st.subheader("Answer")
                st.write(data.get("answer", "No answer returned"))

                st.subheader("Citations")
                st.write(data.get("citations", []))

        except Exception as e:
            st.error(f"Connection error: {e}")