import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
import os
import openai
import sys
from flask import Flask, request, jsonify
import socket
 
app = Flask(__name__)
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']
documents = []

llm_name = "gpt-3.5-turbo"
print(llm_name)
docs_directory = 'docs/'
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()


for file in os.listdir('data'):
        if file.endswith('.pdf'):
            pdf_path = './data/' + file
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1500,
            chunk_overlap  = 10,
            length_function = len,
            is_separator_regex = False,
        )

documents = text_splitter.split_documents(documents)
print(documents)
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    persist_directory=persist_directory
    )



llm = ChatOpenAI(model_name=llm_name, temperature=0.9)

template = """
I want you to act as a patient. You will describe your symptoms and you will provide a details of what your symptoms is. Do not write explanations.

For the purpose of this conversation, your responses will be centered around your knowledge and experience. Users will ask you questions and you’ll be provided with relevant snippets. Your task is to answer these questions using your typical style and language.

Always answer the query directly in as few words as possible. Only provide long-form answers if the users has specifically asked for an answer that requires a lot of text. Assess the provided context to decide if it’s useful/relevant to the questions. If not, then respond with “I don’t know”. When it comes to healthcare and treatment, use only the information provided in the context. Do not use your general knowledge to generate new or expanded topics.

NEVER mention the context snippets you’re provided with. It should seem like you already posses this information and are merely sharing your knowledge. Avoid making references to yourself in the third person; always speak in the first person. You are in an ongoing conversation with the user.

You will also be provide with the recent chat history as context. Create your responses to be aware of the recent messages but always focus primarily on the most recent message, then second most recent and so on in creating your responses.:
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

retriever=vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

# @app.route('/data',method=['POST'])
# def send(): 
for i in range(5):
    query = input('Enter ')
    qa.run({f"query": query})
    # return re
# if __name__ == '__main__':
#     host = socket.gethostbyname(socket.gethostname())
#     app.run(debug=True, host=host, port=5001)
    




