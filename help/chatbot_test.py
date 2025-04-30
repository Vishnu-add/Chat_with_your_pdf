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

# device = torch.device('cpu')
device = torch.device('cuda:0')


checkpoint = "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype = torch.float32,
    # offload_folder= "/model_ck"
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer=tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95,
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

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i] , is_user=True, key= str(i) + "_user")
        message(history["generated"][i] , key= str(i))


def main():
    st.title("Chat with your pdf📚")
    with st.expander("About the App"):
        st.markdown(
        """
        This is a Generative AI powered Question and Answering app that responds to questions about your PDF file.
        """
        )

    user_input = st.text_input("",key="input")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey There!"]

    # Search the database for a response based on user input and update session state
    if user_input:
        answer = process_answer({"query" : user_input})
        st.session_state["past"].append(user_input)
        response = answer
        st.session_state["generated"].append(response)

    # Display Conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)
    


if __name__ == "__main__":
    main()