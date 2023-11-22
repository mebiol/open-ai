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

llm_name = "gpt-3.5-turbo"
print(llm_name)
loader = PyPDFLoader("data/wilee.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap = 150
    )
splits = text_splitter.split_documents(pages)

# Saving splits to a text file
with open('splits_output.txt', 'w', encoding='utf-8') as file:
    for split in splits:
        try:
            # Attempt to access common attributes or methods
            if hasattr(split, 'get_text'):
                text_content = split.get_text()
            elif hasattr(split, 'text'):
                text_content = split.text
            elif hasattr(split, 'extract_text'):
                text_content = split.extract_text()
            else:
                text_content = str(split)  # Fallback if no known method/attribute is found

            file.write(text_content + '\n\n')
        except Exception as e:
            print(f"Error processing split: {e}")



kept = []
for i in splits:
    kept.append(i)

def remove_metadata_from_string(text):
    # Split the text at 'metadata='
    parts = text.split("metadata=")
    
    # Return the first part (text before 'metadata='), if it exists
    return parts[0].strip() if parts else text

# Convert each element in 'kept' to string and remove metadata
cleaned_texts = [remove_metadata_from_string(str(item)) for item in kept]

# # Print the cleaned texts
# for clean in cleaned_texts:
#     print(clean.encode('latin1').decode('utf8'))

def decode_to_thai(text, encoding):
    try:
        return text.encode('latin1').decode(encoding)
    except UnicodeDecodeError:
        return "Decoding error with encoding: " + encoding

page_content = '59\n\x1e\x17\x12\x0b\x0e\x1c\x18\x1aJ\x14B9#8\x13\x03\x1a\x07\x0b\x18\x02\x1a\t\x07\x18;\x07O\x1f ...'  # truncated for brevity

# Try decoding with TIS-620
thai_text_tis620 = decode_to_thai(page_content, 'tis-620')

# Try decoding with cp874
thai_text_cp874 = decode_to_thai(page_content, 'cp874')

print("TIS-620 Decoded Text:\n", thai_text_tis620)
print("\nCP874 Decoded Text:\n", thai_text_cp874)

# index = 0
# for i in splits:
#      # Encode the string to bytes
#     bytes_text = str(i).encode('utf-8')
#     # Assuming 'i' is a string that you want to encode and then decode
#     # If you need to replace backslashes, do it here
#     modified_i = str(bytes_text).replace('\\\\', '\\')
#     # Decode the bytes back to string
#     decoded_text = modified_i.decode('utf-8')
    
#     print(decoded_text)

 
 
# print('',str.join(res))

 


# text = 'gr\\xc3\\xa9gory'
# # Convert the string representation of bytes to actual bytes
# bytes_text = bytes(splits, 'utf-8')
# # Now decode the bytes
# decoded_text = bytes_text.decode('utf-8')

# If you want to interpret the string as bytes encoded in latin1 and then decode
# print('gr\xc3\xa9gory'.encode('latin1').decode('utf8'))

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

llm = ChatOpenAI(model_name=llm_name, temperature=0.9)

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
#----------------------------------------------------------------------------------
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question and if qauetion is out of context just answer nautural in conversation:
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
 
# for i in range(5):
#     query = input('Enter ')
#     re = qa.run({f"query": query})
#     print(re)
#----------------------------------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
    # chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)

    #----------------------------------------------------------------------------------

qa_chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(),
        vectordb.as_retriever(search_kwargs={'k': 1}),
        return_source_documents=True
)

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




