import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


os.environ['OPENAI_API_KEY'] = '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'

model = ChatOpenAI(model='gpt-4')

pdf_path = 'manual.pdf'
loader = PyPDFLoader(pdf_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200,
)

chunks =text_splitter.split_documents(
    documents=docs
)

embedding = OpenAIEmbeddings()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='laptop_manual'
)

retriever = vector_store.as_retriever()

prompt = hub.pull('rlm/rag-prompt')

rag_chain = (
    {
        'context':retriever,
        'question':RunnablePassthrough(),
    }
    |prompt
    |model
    |StrOutputParser()
)


try:
    while True:
        question= input("faça sua pergunta sobre o manual do laptop: \n\n")
        response = rag_chain.invoke(question)
        print(response)
except KeyboardInterrupt:
    exit()

