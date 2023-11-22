import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()

def main():
    print("Chat with PDF 💬")
    # Replace with your PDF file path or use input() to get the file path from the user
    pdf_path = "data/wilee.pdf"  # or input("Enter the path to your PDF: ")
    store_name = os.path.basename(pdf_path)[:-4]
    if os.path.exists(f"{store_name}.pkl"):
            print('------------------------')
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
    else: 
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PdfReader(pdf_file)
        
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            print(f'Processing: {store_name}')
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
    query = input("Ask questions about your PDF file: ")

    # if query:
    docs = VectorStore.similarity_search(query=query, k=1)
    print(docs)
    llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(response)

if __name__ == '__main__':
    main()
