import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain

import os
import openai
import sys
sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

llm_name = "gpt-3.5-turbo"
print(llm_name)
loader = PyPDFLoader("data/Science-ML-2015.pdf")
pages = loader.load()
print(pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)
splits = text_splitter.split_documents(pages)

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

llm = ChatOpenAI(model_name=llm_name, temperature=0.1)

# question = "โรคผื่นภูมิแพ้ผิวหนัง"
# docs = vectordb.similarity_search(question,k=6)

#------------------------------------------------------------

# template = """
#   {Your_Prompt}
  
#   CONTEXT:
#   {context}
  
#   QUESTION: 
#   {query}

#   CHAT HISTORY: 
#   {chat_history}
  
#   ANSWER:
#   """

# prompt = PromptTemplate(input_variables=["chat_history", "query", "context"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history", 
                                  input_key="query")
# chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)

#----------------------------------------------------------------------------------

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    memory=memory
)

chat_history = []
while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt: ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    print('1111111111111111111111111111111111111111111')
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'])
    chat_history.append((query, result['answer']))
    print(chat_history)

# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm, 
#     memory = memory,
#     verbose=True,
# )
# while True:
#     query = input('Enter ')
#     re = conversation.predict(input=f"{query}")
#     print(re)
