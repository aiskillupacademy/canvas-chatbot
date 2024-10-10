from langchain_core.prompts.chat import ChatPromptTemplate
import os
from langchain_groq import ChatGroq
import tempfile
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_google_vertexai import VertexAI
from langchain.output_parsers import PydanticOutputParser
import pymupdf4llm

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.title("Output")
st.sidebar.title("Canvas Chatbot")
llm = ChatGroq(model="llama3-70b-8192")
# llm = VertexAI(model_name="gemini-1.5-flash")

class Output(BaseModel):
    output: dict[str,str] = Field(description="dictionary containing the chat text and task output.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extraxt_doc_text(uploaded_files):
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        documents = []
        if file_extension == ".pdf":
            # loader = PyPDFLoader(temp_file_path)
            data = pymupdf4llm.to_markdown(temp_file_path)
            documents.append(Document(data))
        os.remove(temp_file_path)
    return documents

def create_rag_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 10)
    docs = text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embedding)
    retriever = db.as_retriever(search_kwargs={"k":4})
    # template = """Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Based on the question decide the length of answer and answer. Make it long wherever required. Give answer in markdown format.
    # Question: {question} 
    # Context: {context} 
    # Answer:"""
    # # template = inst + sys_prompt + template
    # prompt = ChatPromptTemplate.from_template(template)
    # rag_chain = (
    #             {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #             | prompt
    #             | llm
    #             | StrOutputParser()
    #             )
    return retriever

def get_output(query):
    parser = PydanticOutputParser(pydantic_object=Output)
    template = """
        You will be given a user query.
        You are a highly efficient assistant whose primary role is to help users complete tasks exactly as they specify. 
        Your response to any user input should be structured in a way that provides two outputs:

        1. 'chat_text': This is the part where you, as a friendly and professional assistant, respond to the user. 
        The response should acknowledge the completion of the task and express willingness to assist further. 
        For example:

        'Here is the result of your task. Feel free to ask for any changes or additional help.'
        'The task has been successfully completed! Let me know if there's anything else you'd like to do.'

        2. 'task_output': This part contains only the result or outcome of the specific task the user has asked you to perform, with no additional commentary. 
        It should be precise and directly reflect the task completion, without any extraneous information. 
        For instance, if the user asked to calculate a sum or write a sales email, only the final sum or email should be included here.

        Your responses should always be structured as a dictionary with these two keys: 'chat_text' and 'task_output'. 
        You should complete the user’s task efficiently and accurately while maintaining a helpful and friendly tone in your response.

        user query: {query}
        \n\n{format_instructions}\n\n
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | parser

    # query = "Write a sales email for rava ai (a marketing copilot for startups)"
    res = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})

    return res.output

uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
documents = []
if uploaded_files:
    text = extraxt_doc_text(uploaded_files)
    documents.extend(text)

task_output = ""
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar container for chat
with st.sidebar:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_input := st.chat_input("Ask a question"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        if uploaded_files:
            retriever = create_rag_chain(documents)
            context = retriever.invoke(user_input)
        with open('response.txt', 'r') as f:
            res = f.read()
        output = get_output(f"Context: \n\n {context}\n\n Use the context to answer the query. \n\n The previous response: \n\n {res} \n\n Use the above response history and context as reference (if required) to answer the below query. \n\n{user_input}")
        task_output = output['task_output']
        chat_text = output['chat_text']

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(chat_text)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": chat_text})

with open('response.txt', 'w') as f:
    f.write(task_output)
st.markdown(task_output)