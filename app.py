import os
import tempfile
import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Configurar a página do Streamlit
st.set_page_config(page_title="Perguntas sobre Arquivos", layout="wide")

# Verificar a API Key da OpenAI
try:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("A chave da API da OpenAI não foi configurada. Verifique os 'secrets'.")
    st.stop()

# Inicializar modelos e embeddings
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def process_file(uploaded_file):
    """
    Processa um arquivo carregado (PDF ou texto) e retorna os documentos extraídos.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.flush()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(temp_file.name)
            elif ext == ".txt":
                loader = TextLoader(temp_file.name)
            else:
                raise ValueError("Formato de arquivo não suportado. Use PDF ou TXT.")
            documents = loader.load()
        finally:
            os.remove(temp_file.name)  # Excluir o arquivo temporário após o uso
    return documents


def create_vectorstore(documents):
    """
    Cria e retorna um índice vetorial a partir dos documentos fornecidos.
    """
    try:
        return FAISS.from_documents(documents, embeddings)
    except Exception as e:
        raise RuntimeError(f"Erro ao criar índice vetorial: {e}")


def initialize_retriever():
    """
    Inicializa o retriever na sessão, caso ainda não exista.
    """
    if "retriever" not in st.session_state:
        st.session_state["retriever"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def main():
    st.title("Sistema de Perguntas sobre Arquivos 📄🤖")
    st.write("Carregue um arquivo (PDF ou TXT) e faça perguntas sobre o conteúdo.")

    initialize_retriever()

    # Upload do arquivo
    uploaded_file = st.file_uploader("Faça upload de um arquivo", type=["txt", "pdf"])
    if not uploaded_file:
        st.info("Por favor, faça upload de um arquivo para começar.")
        return

    if st.button("Processar Arquivo"):
        try:
            st.write("Processando o arquivo...")
            documents = process_file(uploaded_file)
            st.success("Arquivo carregado com sucesso!")
            st.write("Criando índice vetorial...")
            vectorstore = create_vectorstore(documents)
            st.session_state["retriever"] = vectorstore.as_retriever()
            st.success("Índice vetorial criado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            return

    # Verificar se o índice vetorial foi criado
    if not st.session_state["retriever"]:
        st.warning("Faça upload de um arquivo e processe-o para começar.")
        return

    # Entrada de perguntas do usuário
    st.write("### Faça perguntas sobre o arquivo")
    user_question = st.text_input("Digite sua pergunta aqui:")

    if user_question:
        with st.spinner("Pensando..."):
            try:
                chain = ConversationalRetrievalChain.from_llm(llm, st.session_state["retriever"])

                # Limitar histórico de conversa para economizar tokens
                MAX_HISTORY_LENGTH = 5
                if len(st.session_state["chat_history"]) > MAX_HISTORY_LENGTH:
                    st.session_state["chat_history"] = st.session_state["chat_history"][-MAX_HISTORY_LENGTH:]

                response = chain.run({"question": user_question, "chat_history": st.session_state["chat_history"]})

                # Atualizar histórico de conversa
                st.session_state["chat_history"].append((user_question, response))

                # Exibir resposta
                st.write("**Resposta:**")
                st.write(response)
            except Exception as e:
                st.error(f"Erro ao processar sua pergunta: {e}")


# Executa a aplicação
if __name__ == "__main__":
    main()
