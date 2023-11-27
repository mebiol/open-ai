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
loader = PyPDFLoader("data/Symtomp4.pdf")
pages = loader.load()
documents.extend(documents)

documents = []
for file in os.listdir('data'):
    if file.endswith('.pdf'):
        pdf_path = './data/' + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 10
    )

chunked_documents = text_splitter.split_documents(documents)
print(chunked_documents)

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
        documents=chunked_documents,
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
    verbose=True,
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
    re = qa.run({f"query": query})
    print(re)
    # return re
# if __name__ == '__main__':
#     host = socket.gethostbyname(socket.gethostname())
#     app.run(debug=True, host=host, port=5001)
    
#----------------------------------------------------------------------------
# memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
#     # chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)

#     #----------------------------------------------------------------------------------

# qa_chain = ConversationalRetrievalChain.from_llm(
#         ChatOpenAI(),
#         vectordb.as_retriever(search_kwargs={'k': 1}),
#         return_source_documents=True
# )

# chat_history = []

# @app.route('/data', methods=['POST'])
# def send():
#     global chat_history
#     query = request.json.get('msg')
#     # give us a way to exit the script
#     if query == "exit" or query == "quit" or query == "q":
#         print('Exiting')
#         sys.exit()
#     # we pass in the query to the LLM, and print out the response. As well as
#     # our query, the context of semantically relevant information from our
#     # vector store will be passed in, as well as list of our chat history
#     full = f"{query} with old query : {' '.join([str(elem) for elem in chat_history])}"
#     print(full)
#     result = qa_chain({'question': full, 'chat_history': chat_history})
#     print('Answer: ' + result['answer'])
#     # we build up the chat_history list, based on our question and response
#     # from the LLM, and the script then returns to the start of the loop
#     # and is again ready to accept user input.
#     chat_history.append((query, result['answer']))
#     print(chat_history)

#     return jsonify({'answer': result['answer'], 'chat_history': chat_history})


# if __name__ == '__main__':
#     host = socket.gethostbyname(socket.gethostname())
#     app.run(debug=True, host=host, port=5001)




