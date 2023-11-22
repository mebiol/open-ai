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
    pdf_path = "data/3_.pdf"  # or input("Enter the path to your PDF: ")
    store_name = os.path.basename(pdf_path)[:-4]
    if os.path.exists(f"{store_name}.pkl2"):
            print('------------------------')
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
    else: 
            loader = PyPDFLoader(f"data/{store_name}.pdf")
            pages = loader.load()
#            print(pages)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=pages)

            print(f'Processing: {store_name}')
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
    query = input("Ask questions about your PDF file: ")

    # if query:
    docs = VectorStore.similarity_search(query=query, k=3)
    print(len(docs))
    #     llm = OpenAI()
    #     chain = load_qa_chain(llm=llm, chain_type="stuff")
    #     with get_openai_callback() as cb:
    #         response = chain.run(input_documents=docs, question=query)
    #         print(response)

if __name__ == '__main__':
    main()
