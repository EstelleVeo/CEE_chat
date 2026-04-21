import streamlit as st
import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- IMPORT INTELLIGENT DE LA MÉMOIRE ---
try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    from langchain_community.memory import ConversationBufferMemory

# --- IMPORT INTELLIGENT DE LA CHAINE ---
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    # Pour les versions très récentes (1.0+)
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# --- 1. CONFIGURATION ---
PDF_FOLDER = "./fiches_cee"
DB_PATH = "./vector_db"

@st.cache_resource
def get_chatbot():
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    
    # Si la base n'existe pas, on la crée proprement
    if not os.path.exists(DB_PATH):
        st.info("📦 Première utilisation : Création de la base de connaissances...")
        documents = []
        for file in os.listdir(PDF_FOLDER):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
                documents.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        
        # LE CORRECTIF CRUCIAL : Forcer les IDs en chaînes de caractères
        ids = [str(i) for i in range(len(chunks))]
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH,
            ids=ids
        )
        vector_db.persist()
        st.success(f"✅ {len(chunks)} fragments indexés avec succès !")
    else:
        # Si elle existe, on la charge simplement
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

# --- 2. INTERFACE STREAMLIT ---
st.set_page_config(page_title="Assistant CEE", page_icon="🤖")
st.title("🤖 Assistant CEE - Tertiaire")

# On s'assure que la clé API est présente
if not os.getenv("MISTRAL_API_KEY"):
    st.error("🔑 Clé API Mistral manquante. Lancez : export MISTRAL_API_KEY='votre_clé'")
    st.stop()

chatbot = get_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question sur les fiches BAT-TH..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            response = chatbot.invoke({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
