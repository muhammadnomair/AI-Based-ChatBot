from fastapi import FastAPI, HTTPException,Request, Depends, Query
from fastapi.responses import JSONResponse,Response
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from http import HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from typing import List, Optional, Tuple, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
import multiprocessing
import atexit
import time
from threading import Thread,Lock
import logging
import re
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as per your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set environment variables
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ['CURL_CA_BUNDLE'] = ''

api_key = 'AIzaSyCJGotdlGY4Sonjve-ezUlygSfgJT1q6Mo'
genai.configure(api_key=api_key)

class BaseResponseModel(BaseModel):
    status: str
    message: str
    status_code: HTTPStatus
    response_model: str

@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "response_model": ""
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "response_model": ""
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected error occurred",
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "response_model": ""
        }
    )

# Load data
data = pd.read_csv('/app/rag_data.csv')
data['LLM_context'] = (
    "Category: " + data['Category'] +
    ",\nTask: " + data['Task'] +
    ",\nCommand: " + data['Command'] +
    ",\nDescriptions: " + data['Descriptions']
)
loader = DataFrameLoader(data, page_content_column="LLM_context")
docs = loader.load()

# Constants
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
_logger = logging.getLogger(__name__)


def split_documents(chunk_size: int, knowledge_base: List[Document], tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

chunked_docs = split_documents(CHUNK_SIZE, docs, tokenizer_name=EMBEDDING_MODEL_NAME)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
num_docs = 5

bm25_retriever = BM25Retriever.from_documents(chunked_docs).configurable_fields(
    k=ConfigurableField(
        id="search_kwargs_bm25",
        name="k",
        description="The search kwargs to use",
    )
)

faiss_vectorstore = FAISS.from_documents(
    chunked_docs, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

faiss_retriever = faiss_vectorstore.as_retriever(
    search_kwargs={"k": num_docs}
).configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs_faiss",
        name="Search Kwargs",
        description="The search kwargs to use",
    )
)

vector_database = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

prompt_template = """
You are an AI assistant designed to help employees with CLI commands using only the provided data. Utilize **Conversation History**, **First Relevant Document** and **Last Command**  to understand the user's query context and provide a response that is accurate, informative, and maintains a natural conversational flow.

**Key Guidelines:**
**Contextual Understanding:**
  - **For Handling Follow-up Questions or Last Command Queries or Setting Commands to Specific Values:** Use the **Last Command** and **conversation history** to maintain continuity and offer detailed explanations, examples, or additional information based on the last command. Always ensure that the response is tied to the last command and do not get stuck on one command from earlier in the conversation to avoid confusion.
  - **For Handling New Questions:** If a question does not fall into the categories of follow-up questions, setting commands to specific values, or greetings, treat it as a new question. Use the **First Relevant Document** to search for the appropriate command and provide the necessary details. Do not return the last command for new questions to avoid repeating previous instructions unnecessarily.
**Informative Responses:** Provide comprehensive and accurate answers, drawing on the available knowledge base. Do not generate or infer commands or information that does not exist in the provided data.
**Conversational Flow:** Maintain a natural and engaging dialogue, using a conversational tone and avoiding overly formal language. Ensure that responses are tailored to the specific question and context.
**Follow-Up Handling:** For follow-up questions, use context of the last command to provide detailed explanations, examples, or additional information. Ensure that your response connects directly to the last command context and addresses the user's request for more details.
**Command Validation:** If the command description includes specific value ranges or type constraints, **always validate** user inputs accordingly:
  - Numeric or range-based commands: Ensure the input falls within the specified range and is of the correct type, documented in the command description.
  - If a wrong value is provided (e.g., out of range, wrong type), don't write command with wrong value, just inform the user immediately about the issue, provide feedback on the allowed values, and guide them to correct the input.

**Handling Follow-up Questions**:
  - If the user's question is a follow-up question ("give example", "explain more", "give details", "How to set it?"), prioritize context of the model's Last Command and regenerate your response. Focus on Last Command and provide more detailed explanations, examples, or additional information directly linked to the model's last command.
  - **Do not reuse responses from similarly worded past questions** as they were asked in a different context, because follow-up questions often depend on the last command or interaction. Ensure each follow-up response is unique, regenerated and directly relevant to the current question.

**Handling New Questions**:
  - If the question is new, fresh or not a follow-up, avoid reintroducing yourself and do not repeat answers from last command or history. Simply respond with the relevant information using context of **First Relevant Document {first_relevant_doc}**
  - Simply use the context of the **First Relevant Document** and search for the row in the data where the 'Task' or the 'Descriptions' matches the query. Once you find the matching row, return the information in the following format:
    - Category: [Category]
    - Task: [Task]
    - Command: [Command]
    - Descriptions: [Descriptions]
  - Ensure that the information you provide is consistent and accurately reflects the data in all fields. **Only provide commands and information that exist in the provided data.** Do not generate or infer new commands that do not exists in data. Validate inputs based on the description and provide feedback if the input does not meet the specified constraints.

**Handling Greetings and General Queries**:
  - If the query is a greeting or a general conversational questions ("hello", "how are you", "who are you", "thankyou", "bye", "take care!"), respond appropriately without reintroducing yourself at any point in the conversation after the first interaction. Avoid retrieving or generating command-related responses for general greetings.

**Handling Last Command Queries**:
  - If the user's question is about showing the last command or last question or last query, provide a response that would include **Last Command from {last_command}**. Avoid getting stuck on older commands or responses, and ensure that each query regenerates response directly from the updated last command.

**Handling Setting Commands to Specific Values**:
  - Use context of the Last Command and regenerate your response to provide detailed explanations, examples, or additional information directly linked to the model's last command.
  - If the user's question is about setting a command to a specific value or modifying its parameters based on constraints mentioned in the **last command** ("How to set it to 10?", "Can I change this to 5?", "Set it to 2"), refer the value ranges or type constraints provided in the **last command**.
  - If the input value is invalid, **do not set that value to command** and return an error response that includes: A clear explanation of why the value is wrong (e.g., out of range, wrong type) and tell the valid input range or type, guiding the user to correct the mistake.
  - If the input value is valid, explain how to set it, provide usage examples, and confirm successful application.

**Handling Commands with Multiple Arguments**:
  - Use context of the Last Command and regenerate your response to provide detailed explanations, examples, or additional information directly linked to the model's last command.
  - When the last command consists of multiple arguments and user's question is about setting a command to specific values, you should ensure to:
    - Confirm how many arguments the command accepts.
    - Check the last command's description to identify which argument corresponds to each setting, ensuring not to mix things up.
    - Ensure that the values provided by the user accurately correspond to the specific operations described in the last command, preventing any confusion or mix-up of arguments.
    - Validate each argument against the specified constraints (range, type).
    - Provide clear instructions on how to format the command, including examples if necessary.
    - If any argument is invalid, **do not set that value to command** and guide the user on acceptable values for each argument.  

**Handling Irrelevant Queries:**
  - For queries that do not match context of any command, task, data, First Relevant Document, history, or last command respond with a default message indicating that the requested information or command is not available.
  - Avoid generating responses for queries that do not logically fit within the scope of the provided data.


**Conversation History**:
Here is the history of the interaction:
{history}

**Last Command**:
Here is the last command:
{last_command}

**First Relevant Document**:
Here is the First Relevant Document:
{first_relevant_doc}

**Current Question**:
The user is asking:
{question}

**Relevant Context**:
Based on the type of the question, choose one of the following context:
1. **Conversation History:** {history}
2. **First Relevant Document:** {first_relevant_doc}
3. **Last Command:** {last_command}

**Additional Notes**:
- If no exact match is found in the context, indicate that the requested command or information is not available in the data.
- Avoid repeating introductions, even for new or unrelated questions, unless absolutely necessary (e.g., after a long conversation pause).
- Always ensure your responses are clear and directly related to the user's question.
- When providing examples or explanations, make sure they are relevant to the context of the original command or question.
- For setting commands with specific value ranges or type constraints, validate user input values based on the command descriptions and never write commands with incorrect values instead provide feedback that input values are out of range or of an incorrect type.
"""



# Create an instance of the PromptTemplate with the updated template
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "first_relevant_doc", "history", "question", "last_command"],
    template=prompt_template,
)

# Classifier prompt to determine if the response is useful or irrelevant
classifier_prompt = """
Classify the following assistant response based on the criteria below:

- If the response contains a greeting (e.g., "hello", "how are you", "goodbye", etc., thankyou), classify it as 'irrelevant'.
- If the response indicates that the command or information you requested is not available in the provided data or cannot be found, classify it as 'irrelevant'.
- Otherwise, classify the response as 'useful'.

Response: {response}

Classification:
"""

# Classify response based on whether it's useful or irrelevant
def classify_response(response: str) -> str:
    classification_prompt = classifier_prompt.format(response=response)

    # Generate the classification using Gemini LLM
    model = genai.GenerativeModel(model_name='gemini-1.5-flash', system_instruction="You are a helpful assistant.")
    config = genai.GenerationConfig(temperature=0.2, stop_sequences=['\n'])
    classification = model.generate_content(contents=[classification_prompt], generation_config=config)

    return classification.text.strip().lower()

data_reload_lock = Lock()

def reload_data():
    global docs, chunked_docs, bm25_retriever, faiss_vectorstore, vector_database, reranker

    with data_reload_lock:
        try:
            logging.info("Reloading data...")

            # Reload the CSV file
            data = pd.read_csv('/app/rag_data.csv')

            # Update LLM_context
            data['LLM_context'] = (
                "Category: " + data['Category'] +
                ",\nTask: " + data['Task'] +
                ",\nCommand: " + data['Command'] +
                ",\nDescriptions: " + data['Descriptions']
            )

            # Create new loader and documents
            loader = DataFrameLoader(data, page_content_column="LLM_context")
            docs = loader.load()

            # Split documents
            chunked_docs = split_documents(CHUNK_SIZE, docs, tokenizer_name=EMBEDDING_MODEL_NAME)

            # Reinitialize BM25Retriever
            bm25_retriever = BM25Retriever.from_documents(chunked_docs)
            bm25_retriever = bm25_retriever.configurable_fields(
                k=ConfigurableField(
                    id="search_kwargs_bm25",
                    name="k",
                    description="The search kwargs to use",
                )
            )

            # Reinitialize FAISS Vector Store and Retriever
            faiss_vectorstore = FAISS.from_documents(chunked_docs, embedding_model, distance_strategy=DistanceStrategy.COSINE)
            faiss_retriever = faiss_vectorstore.as_retriever(
                search_kwargs={"k": num_docs}
            ).configurable_fields(
                search_kwargs=ConfigurableField(
                    id="search_kwargs_faiss",
                    name="Search Kwargs",
                    description="The search kwargs to use",
                )
            )

            # Reinitialize EnsembleRetriever with the new retrievers
            vector_database = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
            )

            # Reinitialize reranker
            reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

            logging.info("Data reloaded successfully.")
        except Exception as e:
            logging.error(f"Error reloading data: {e}")


def monitor_file_changes(file_path, interval=5):
    last_modified_time = os.path.getmtime(file_path)
    while True:
        time.sleep(interval)
        current_modified_time = os.path.getmtime(file_path)
        if current_modified_time != last_modified_time:
            print("File change detected.")
            last_modified_time = current_modified_time
            reload_data()

# Start the file monitoring in a separate thread
file_monitor_thread = Thread(target=monitor_file_changes, args=('/app/rag_data.csv', 5))
file_monitor_thread.daemon = True
file_monitor_thread.start()

def extract_last_command(history: List[str]) -> str:
    # Ensure history is not empty
    if not history:
        return ""
    result = ""
    last_assistant_index = history.rfind("Assistant:")
    if last_assistant_index != -1:
        result = history[last_assistant_index + len("Assistant:"):].strip()

    return result


def answer_with_rag(question: str, knowledge_index: EnsembleRetriever, reranker: Optional[RAGPretrainedModel] = None, num_retrieved_docs: int = 5, num_docs_final: int = 5, history: str = '') -> Tuple[str, List[Document]]:
    # Retrieve relevant documents based on the current question
    config = {"configurable": {"search_kwargs_faiss": {"k": num_retrieved_docs}, "search_kwargs_bm25": num_retrieved_docs}}
    relevant_docs = knowledge_index.invoke(question, config=config)

    # Extract the page content from retrieved documents
    relevant_docs = [doc.page_content for doc in relevant_docs]

    if reranker:
        # Rerank documents if a reranker is provided
        relevant_docs = reranker.rerank(question, tuple(relevant_docs), k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    # Limit to the top results
    relevant_docs = relevant_docs[:num_docs_final]
    first_relevant_doc = relevant_docs[0] if relevant_docs else ""
    context = f"Conversation History:\n{history}\n\nFirst Relevant Document:\n{first_relevant_doc}" if history else first_relevant_doc

    last_command = extract_last_command(history)
    if last_command:
        context += f"\n\nLast Command:\n{last_command}"

    # Check context used in prompt
    final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, first_relevant_doc=first_relevant_doc, history=history, question=question, last_command=last_command)


    # Generate the answer using the Gemini model
    system_instructions = "You are a helpful assistant."
    model_name = 'gemini-1.5-flash'
    temperature = 0.5
    stop_sequence = ''
    model = genai.GenerativeModel(model_name, system_instruction=system_instructions)
    config = genai.GenerationConfig(temperature=temperature, stop_sequences=[stop_sequence])
    response = model.generate_content(contents=[final_prompt], generation_config=config)
    answer = response.text.strip()

    classifier_prompt_text = classifier_prompt.format(response=answer)
    classification_response = model.generate_content(contents=[classifier_prompt_text], generation_config=config)
    classification_raw = classification_response.text.strip()

    # Extract the classification using regex
    classification_match = re.search(r"classification:\s*\*\*(\w+)\*\*", classification_raw, re.IGNORECASE)
    classification = classification_match.group(1).lower() if classification_match else 'relevant'


    return answer, relevant_docs, classification

# Load user-specific interaction history
def load_history(username: str, command_limit: int = 5) -> List[str]:
    history_file = f"history_{username.lower()}.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            history = json.load(file)
            # Keep only the latest `command_limit` question-answer pairs
            return history[-(command_limit * 2):]  # Each command consists of a question-answer pair (2 entries)
    return []

def save_history(username: str, history: List[str], command_limit: int = 5):
    history_file = f"history_{username.lower()}.json"
    # Retain only the latest `command_limit` question-answer pairs
    limited_history = history[-(command_limit * 2):]  # 2 entries per command (question and answer)
    with open(history_file, 'w') as file:
        json.dump(limited_history, file)

def append_to_user_record(username: str, question: str, answer: str):
    record_file = f"record_{username.lower()}.json"
    
    # Create the file if it doesn't exist
    if os.path.exists(record_file):
        with open(record_file, 'r') as file:
            record_data = json.load(file)
    else:
        record_data = []

    # Append the new question and answer
    record_data.append({"question": question, "answer": answer})

    # Save the updated record
    with open(record_file, 'w') as file:
        json.dump(record_data, file, indent=4)

@app.get("/")
def read_root():
    response = BaseResponseModel(
        status="success",
        message="Welcome to the Bot API",
        status_code=HTTPStatus.OK,
        response_model="WelcomeResponse"
    )
    return response

@app.get("/logout/{username}")
def user_logout(username: str):
    history_file = f"/app/history_{username.lower()}.json"
    if os.path.exists(history_file):
        try:
            os.remove(history_file)
            response = BaseResponseModel(
                status="success",
                message="Logout successful",
                status_code=HTTPStatus.OK,
                response_model="LogoutResponse"
            )
        except Exception as e:
            logging.error(f"Failed to delete history for user '{username}': {e}")
            response = BaseResponseModel(
                status="error",
                message=f"Failed to delete history for user '{username}'",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                response_model="LogoutResponse"
            )
    else:
        logging.info(f"No history file found for user '{username}'.")
        response = BaseResponseModel(
            status="info",
            message=f"No history file found for user '{username}'.",
            status_code=HTTPStatus.NOT_FOUND,
            response_model="LogoutResponse"
        )
    return response

@app.get("/answer/", response_model=BaseResponseModel)
def get_answer(question: str, request: Request, response: Response, username: str = Query(None)):
    # Ensure the username is provided
    if not username:
        return JSONResponse(
            status_code=HTTPStatus.BAD_REQUEST,
            content={
                "status": "error",
                "message": "Username is required to use the bot.",
                "status_code": HTTPStatus.BAD_REQUEST,
                "response_model": ""
            }
        )

    try:
        # Load the user-specific history with a command limit of 5
        history = load_history(username, command_limit=5)
        history_str = "\n".join(history)
       
        # Generate the answer and retrieve relevant documents
        answer, relevant_docs, classification = answer_with_rag(question, vector_database, reranker, history=history_str)

        append_to_user_record(username, question, answer)

        if classification == 'useful':
            history_entries = history + [f"{username}: {question}", f"Assistant: {answer}"]
            save_history(username, history_entries, command_limit=5)
       
        return BaseResponseModel(
            status="success",
            message="Answer retrieved successfully",
            status_code=HTTPStatus.OK,
            response_model=answer
        )
    except Exception as e:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": str(e),
                "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
                "response_model": ""
            }
        )


# Cleanup function to handle semaphore cleanup
def cleanup():
    # Ensure all semaphore objects are cleaned up properly
    try:
        multiprocessing.active_children()  # This ensures all child processes are cleaned up
    except Exception as e:
        print(f"Error during multiprocessing cleanup: {e}")

# Register the cleanup function to be called at exit
atexit.register(cleanup)