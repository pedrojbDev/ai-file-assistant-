import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

# Configurar a página do Streamlit
st.set_page_config(page_title="Perguntas sobre Arquivos", layout="wide")

# Configurar a API da OpenAI
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except Exception as e:
    st.error("Chave da API da OpenAI não encontrada. Verifique o arquivo secrets.toml.")
    st.stop()

# Inicializar o modelo GPT e os embeddings
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Função para carregar e processar o arquivo
def process_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file.read())
        temp_file.flush()
        if ext == ".pdf":
            loader = PyPDFLoader(temp_file.name)
        elif ext == ".txt":
            loader = TextLoader(temp_file.name)
        else:
            raise ValueError("Tipo de arquivo não suportado. Use PDF ou TXT.")
    return loader.load()

# Interface Streamlit
st.title("Sistema de Perguntas sobre Arquivos 📄🤖")
st.write("Carregue um arquivo e faça perguntas sobre o conteúdo.")

# Carregar o arquivo
uploaded_file = st.file_uploader("Faça upload de um arquivo (PDF ou texto)", type=["txt", "pdf"])

if uploaded_file:
    try:
        st.write("Processando o arquivo...")
        documents = process_file(uploaded_file)
        st.success("Arquivo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.stop()
    
    # Criar vetor de índice e recuperador
    with st.spinner("Criando índice para o arquivo..."):
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
        except Exception as e:
            st.error(f"Erro ao criar índice vetorial: {e}")
            st.stop()
    
    # Inicializar histórico de conversa
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Criar sessão de perguntas
    st.write("### Faça perguntas sobre o arquivo")
    user_question = st.text_input("Digite sua pergunta aqui:")

    if user_question:
        with st.spinner("Pensando..."):
            try:
                # Configurar o chain para busca e respostas
                retriever = vectorstore.as_retriever()
                chain = ConversationalRetrievalChain.from_llm(llm, retriever)
                
                # Adicionar histórico ao chain
                response = chain.run({"question": user_question, "chat_history": st.session_state["chat_history"]})
                
                # Atualizar histórico
                st.session_state["chat_history"].append((user_question, response))

                # Exibir resposta
                st.write("**Resposta:**")
                st.write(response)
            except Exception as e:
                st.error(f"Erro ao processar sua pergunta: {e}")
