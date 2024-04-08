import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from  langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter 
st.set_page_config(page_title="LOCAL RAG",layout="centered")
flag=False
#MODEL
st.title("DocQuest 📚 🔍")
st.caption("Perform RAG with Local Models!")

model=Ollama(model="openhermes:latest")
# OLLAMA EMBEDDIGS
embeddings=OllamaEmbeddings()
#output parser
parser=StrOutputParser()

#template for the model
template="""
You are a helpful AI Assistant named DocQuest! You answer questions precisely and in a formatted manner.
Answer the Question  based on the context Given below . If you cannot answer the question, reply "IDK BRUH".

Context: {context}

Question: {question}
"""

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        
prompt=PromptTemplate.from_template(template)

def getpath():
    path=st.text_input("Please enter the path to your PDF file",key=2)
    return path

    


@st.cache_resource(show_spinner=True)
def load_vectorDB(file_name):
    loader=PyPDFLoader(file_name)
    pages=loader.load_and_split()
    st.caption("File loaded sucessfully!")
    vectorstore=DocArrayInMemorySearch.from_documents(pages,embedding=embeddings)
    retriever=vectorstore.as_retriever()
    return retriever
    
try:
    file=getpath()
    if file:
        retriever1=load_vectorDB(file)
        st.caption("Vector database created successfully!")
        chain=(
    {
        "context": itemgetter("question")|retriever1,
        "question":itemgetter("question")
    }
    | prompt
    | model
    | parser
)       
        prompt = st.chat_input("Ask a question")
        with st.spinner("Generating response.."):
            if prompt:
                response=chain.invoke({"question":prompt})
                bot=f"BOT : {response}"
                with st.chat_message("user"):
                    st.markdown(prompt)
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("assistant"):
                    st.markdown(bot)
        # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": bot})
            
      
except Exception as e:
    st.warning(e)



#D:\RAG\Abstract_.pdf