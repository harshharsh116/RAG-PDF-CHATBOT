import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
st.set_page_config(
    page_title="RAG PDF CHATBOT",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

        :root {
            --primary: #6366f1;
        }

        .stApp {
            background-color: #0a0a14;
            color: #f8fafc;
        }

        /* Main Title */
        h1 {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(90deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            text-align: center;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* File Uploader */
        .stFileUploader {
            background-color: #1e1e2e;
            padding: 1.5rem;
            border-radius: 16px;
            border: 2px dashed #6366f1;
        }

        .stFileUploader label {
            color: #e2e8f0 !important;
            font-weight: 500;
        }

        /* Text Input */
        .stTextInput > div > div > input {
            background-color: #1e1e2e !important;
            color: #f1f5f9 !important;
            border: 1px solid #475569 !important;
            border-radius: 12px;
            padding: 14px 16px;
        }

        /* Answer Box */
        .answer-box {
            background-color: #1a1a2e;
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid #6366f1;
            color: #f1f5f9;
            font-size: 1.05rem;
            line-height: 1.7;
            box-shadow: 0 4px 25px rgba(99, 102, 241, 0.15);
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border: none;
            padding: 14px 32px;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 12px;
            width: 100%;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 25px rgba(99, 102, 241, 0.4);
        }

        /* Success / Info Messages */
        .stSuccess, .success-msg {
            background-color: #14532d;
            color: #86efac;
            border: 1px solid #4ade80;
            border-radius: 12px;
            padding: 1rem;
        }

        /* Fix for all labels and text */
        label, .stMarkdown, p {
            color: #e2e8f0 !important;
        }

        /* Streamlit default elements override */
        .css-1q8cf1e, .css-1y4p8p9, .st-emotion-cache-1wmy5h9 {
            color: #f1f5f9 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="custom-card">
    <div class="main-title">📚 RAG PDF CHATBOT</div>
    <div class="sub-text">
        Upload PDFs • Create Vector Embeddings • Ask Questions Instantly
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='text-align: center;'>RAG PDF PROJECT</h1>",
    unsafe_allow_html=True
)
if "vectors" not in st.session_state:
    st.session_state.vectors = None
@st.cache_resource
def load_embeddings():
   embeddings = HuggingFaceEmbeddings(
        model_name = "BAAI/bge-base-en-v1.5"
    )
   return embeddings

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"],accept_multiple_files=True)
b1  = st.button("process pdfs")
all_docs = []
if uploaded_files and b1:
    for pdf in uploaded_files:
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(pdf.read())
        loader = PyPDFLoader(temp.name)
        docs = loader.load()
        all_docs.extend(docs)
    st.subheader("Uploaded PDF files successfully")



    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    docs = splitter.split_documents(all_docs)

    embeddings = load_embeddings()

    vectors = FAISS.from_documents(
        docs,
        embeddings,
    )

    st.session_state.vectors = vectors

    st.success("Vector DB Created")

    st.write("Total Chunks:", len(docs))

query = st.text_input("Ask a Question")


client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
if query and st.session_state.vectors is not None:
    with st.spinner("Generating Answer..."):
        retriever = st.session_state.vectors.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4}
        )

        retrieved = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in retrieved])


        prompt = f"""You are a precise assistant. Answer the question in a short and clear way.
Use ONLY the information from the context below.
Do not add extra information. 
Keep your answer in maximum 2-3 sentences. Be direct.

Context:
{context}

Question: {query}

Answer:"""
try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_tokens=350
        )

        generated_text = response.choices[0].message.content

        # Clean the answer
        if prompt in generated_text:
            answer = generated_text.split(prompt)[-1].strip()
        else:
            answer = generated_text.strip()

        # Extra cleaning to remove unfinished sentences
        if answer.endswith(('.', '!', '?')):
            final_answer = answer
        else:
            # Cut off at last complete sentence
            last_sentence = answer.rsplit('.', 1)[0] + '.' if '.' in answer else answer
            final_answer = last_sentence

        st.markdown(f"""
        <div class="answer-box">
            <div class="answer-title">💡 Answer</div>
            <div class="answer-text">{final_answer}</div>
        </div>
        """, unsafe_allow_html=True)
except Exception as e:
    st.error(str(e))
