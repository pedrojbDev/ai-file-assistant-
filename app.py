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

# Função para processar o arquivo
def process_file(uploaded_file):
    """
    Processa um arquivo carregado (PDF ou texto) e retorna os documentos extraídos.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file.flush()

        if ext == ".pdf":
            loader = PyPDFLoader(temp_file.name)
        elif ext == ".txt":
            loader = TextLoader(temp_file.name)
        else:
            os.remove(temp_file.name)  # Exclui o arquivo temporário
            raise ValueError("Formato de arquivo não suportado. Use PDF ou TXT.")

        documents = loader.load()

    # Excluir o arquivo temporário após o uso
    os.remove(temp_file.name)
    return documents

# Função principal para criar o índice vetorial e permitir perguntas
def main():
    st.title("Sistema de Perguntas sobre Arquivos 📄🤖")
    st.write("Carregue um arquivo (PDF ou TXT) e faça perguntas sobre o conteúdo.")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Faça upload de um arquivo", type=["txt", "pdf"])
    if not uploaded_file:
        st.info("Por favor, faça upload de um arquivo para começar.")
        return

    try:
        st.write("Processando o arquivo...")
        documents = process_file(uploaded_file)
        st.success("Arquivo carregado com sucesso!")
    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        return

    # Criar índice vetorial
    with st.spinner("Criando índice vetorial..."):
        try:
            vectorstore = FAISS.from_documents(documents, embeddings)
        except Exception as e:
            st.error(f"Erro ao criar índice vetorial: {e}")
            return

    # Gerenciar histórico de conversa
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "retriever" not in st.session_state:
        retriever = vectorstore.as_retriever()
        st.session_state["retriever"] = retriever

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
main()

