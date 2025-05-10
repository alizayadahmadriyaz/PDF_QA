import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
import torch

# Load environment variables
load_dotenv()

MODEL_REPO_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256  # Prevents long outputs and input length errors
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    llm = load_local_model()
    # No memory: only current question is used
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

def extract_answer(full_response):
    """
    Extract only the answer from the model output.
    Assumes answer starts after 'Helpful Answer:'.
    """
    marker = "Helpful Answer:"
    if marker in full_response:
        return full_response.split(marker, 1)[1].strip()
    else:
        return full_response.strip()

def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    # Session state for the conversation chain only
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # User input box
    user_question = st.text_input("Ask a question about your documents:")
    

    # Handle user input and display answer
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation({
            "question": user_question,
            "chat_history": []
        })
        
        # Depending on your langchain version, the answer may be in response['answer'] or response['result']
        # If not, fallback to the last assistant message in response['chat_history']
        answer = ""
        if "answer" in response:
            answer = extract_answer(response["answer"])
        elif "result" in response:
            answer = extract_answer(response["result"])
        elif "chat_history" in response and response["chat_history"]:
            # Try to extract the latest assistant message
            for message in reversed(response["chat_history"]):
                if hasattr(message, "role") and message.role == "assistant":
                    answer = extract_answer(message.content)
                    break
                # Fallback for legacy message objects
                elif not hasattr(message, "role"):
                    idx = response["chat_history"].index(message)
                    if idx % 2 == 1:
                        answer = extract_answer(message.content)
                        break
        st.markdown(f"**User:** {user_question}")
        st.markdown(f"**Bot:** {answer}")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success("Documents processed! You can now ask questions.")

if __name__ == '__main__':
    import torch
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

    main()
