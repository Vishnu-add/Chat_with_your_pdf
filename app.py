import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import RetrievalQA
from streamlit_chat import message
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os

st.set_page_config(page_title="pdf-GPT", page_icon="📖", layout="wide")

@st.cache_resource
def get_model():
    device = torch.device('cpu')
    # device = torch.device('cuda:0')

    checkpoint = "LaMini-T5-738M"
    checkpoint = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map=device,
        torch_dtype = torch.float32,
        # offload_folder= "/model_ck"
    )
    return base_model,tokenizer

@st.cache_resource
def llm_pipeline():
    base_model,tokenizer = get_model()
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length = 512,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95,
        # device=device
    )

    local_llm = HuggingFacePipeline(pipeline = pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response=''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text

def clear_history():
    # Clear the conversation history
    # history["generated"] = []
    # history["past"] = []
    st.session_state["generated"] = []
    st.session_state["past"] = []

# Display conversation history using Streamlit messages
def display_conversation(history):
    # st.write(history)
    for i in range(len(history["generated"])):
        message(history["past"][i] , is_user=True, key= str(i) + "_user")
        if isinstance(history["generated"][i],str):
          message(history["generated"][i] , key= str(i))
        else:
          
          message(history["generated"][i][0] , key= str(i))
        #   sources_list = []
        #   for source in history["generated"][i][1]['source_documents']:
        #     # st.write(source.metadata['source'])
        #     sources_list.append(source.metadata['source'])
        #   message(str(set(sources_list)) , key="sources_"+str(i))


# function to display the PDF of a given file
@st.cache_data
def displayPDF(file,file_name):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="1000" type="application/pdf"></iframe>'
    # st.write()
    # pdf_display = f'<embed src="http://localhost:8900/{file_name}" width="700" height="1000" type="application/pdf"></embed>'
    # pdf_display = f'<iframe src="http://localhost:8900/{file_name}" width="700" height="900" type="application/pdf"></iframe>'


    # st.write(pdf_display)
    st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_resource
def data_ingestion(file_path,persist_directory):
    # for root, dirs, files in os.walk("docs"):
    #     for file in files:
    if file_path.endswith(".pdf"):
        print(file_path)
        loader = PDFMinerLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
        texts = text_splitter.split_documents(documents)
        # create embeddings 
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # create vector store
        db = Chroma.from_documents(texts, embeddings, persist_directory="uploaded/db")
        db.persist()
        db=None
  
def main():
    st.markdown("<h1 style='text-align:center; color: green;'>Chat with Your PDF 📑</h1>", unsafe_allow_html=True)
    # st.markdown("<h3 style='text-align:center; color: grey;'>Built by Vicky</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color: red;'>Upload your PDF</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("",type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "name" : uploaded_file.name,
            "type" : uploaded_file.type,
            "size" : uploaded_file.size
        }
        filepath = "uploaded/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1,1])
        with col1:
            # st.markdown("<h2 style='text-align:center; color:grey;'>PDF Details</h2>",unsafe_allow_html=True)
            # st.write(file_details)
            st.markdown("<h2 style='text-align:center; color: grey;'>PDF Preview</h2>", unsafe_allow_html=True)
            displayPDF(filepath,uploaded_file.name)
            # displayPDF(uploaded_file)
        with col2:
            with st.spinner("Embeddings are in process......."):
                ingested_data = data_ingestion(filepath,filepath)
            st.success('Embeddings are created Successfully!')
            st.markdown("<h2 style='text-align:center; color: grey;'>Chat Here</h2>", unsafe_allow_html=True)
            

            user_input = st.text_input(label="Message",key="input")
            # user_input = st.chat_input("",key="input")
            # For displaying text_input in the bottom
            # styl = f"""
            #         <style>
            #             .stTextInput {{
            #             position: fixed;
            #             bottom: 3rem;
            #             }}
            #         </style>
            #         """
            # st.markdown(styl, unsafe_allow_html=True)
            
            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey There!"]

            if st.button("Clear History",key="clear"):
                clear_history()

            # Search the database for a response based on user input and update session state
            if st.button("Get Answer",key="submit"):
                # st.write("Button Pressed")
            # if user_input:

                answer = process_answer({"query" : user_input})
                # answer = user_input
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)
                # st.write(st.session_state)
                # user_input = st.text_input(label="Message",key="input")

            # Display Conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

            



if __name__ == "__main__":
    main()
