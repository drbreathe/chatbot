from dotenv import load_dotenv
import os
from operator import itemgetter
from typing import Dict, List, Optional
from langchain import hub
from openai import OpenAI
import pinecone
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (RunnableBranch, RunnableLambda,
                                       RunnableMap, RunnablePassthrough)
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
from pydantic import BaseModel
import humps
import json
from typing import Iterator

from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Dr.Liu [RAG]"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# load your credentials from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index = os.getenv("PINECONE_INDEX_NAME")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")

client = Client()

_DEFAULT_SUMMARIZER_TEMPLATE = """
1. Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
2. If most summary and new_lines are Chinese. You can only use traditional Chinese and refer to 中文範例 returning a new summary.

ENGLISH EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF ENGLISH EXAMPLE

中文範例
目前摘要：
人類詢問AI對人工智慧的看法。AI認為人工智慧是一種良好的力量。

新的對話內容：
人類：你為什麼認為人工智慧是一種良好的力量？
AI：因為人工智慧將幫助人類發揮他們的全部潛能。

新摘要：
人類詢問AI對人工智慧的看法。AI認為人工智慧是一種良好的力量，因為它將幫助人類發揮他們的全部潛能。
忠文範例結束

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""

SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

# Map file extensions with appropriate document handlers and their arguments
MEMORY_MAPPINGS = {
    "buffer": (ConversationBufferWindowMemory, {"memory_key":"chat_history"}),
    "summary": (ConversationSummaryMemory, {"memory_key":"chat_history","llm":OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1"),"prompt":SUMMARY_PROMPT, "return_messages":True}),
    "summary_buffer": (ConversationSummaryBufferMemory, {"memory_key":"chat_history", "llm":OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1"),"prompt":SUMMARY_PROMPT, "return_messages":True}),
}

def get_chatbot_memory(buffer_window: int, smry_bfr_tkn: int, memory_type: str = 'buffer'):
    if memory_type in MEMORY_MAPPINGS.keys():
        memory_class, memory_args = MEMORY_MAPPINGS[memory_type]

        additional_args = {"k":buffer_window, "max_token_limit":smry_bfr_tkn}
        all_args = {**additional_args, ** memory_args}
        memory_loader = memory_class(**all_args)

        return memory_loader
    
    raise ValueError(f"Memory type not supported: {memory_type}. Supported types are {MEMORY_MAPPINGS.keys()}")

def convert_source_documents(source_list: List):
    src_docs = "["
    for ind, document in enumerate(source_list):
        src_doc = (
            "{'pageContent' : "
            + "'"
            + document["page_content"]
            + "',  "
            + "'metadata' : "
            + "{'page' : "
            + document["page"]
            + ", "
            + "'source': "
            + document["source"]
            + "}"
            + "}"
        )
        if ind == len(source_list) - 1:
            src_docs += src_doc + "]"
        else:
            src_docs += src_doc + ","
    return src_docs

def get_retrieval_chain(prompt_hub: str, target_no_of_source_docs: int =1):
    CONDENSE_QUESTION_PROMPT = hub.pull(prompt_hub)

    # initialize Pinecone with credentials
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # initialize OpenAIEmbeddings and chat models with credentials
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_api_key
    )

    # initialize retrieval chain
    if os.getenv("PINECONE_NAMESPACE"):
        vectorstore = Pinecone.from_existing_index(
            index_name=pinecone_index,
            embedding=embeddings,
            text_key="text",
            namespace=pinecone_namespace,
        )
    else:
        vectorstore = Pinecone.from_existing_index(pinecone_index, embeddings)

    retriever = vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs={"k": target_no_of_source_docs, "score_threshold": 0.88}, )

    # User input
    class ChatHistory(BaseModel):
        question: str
        chat_history: Optional[List[Dict[str, str]]]


    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(chat_history=lambda x: x["chat_history"])
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(itemgetter("question")),
    )

    retriever_chain = RunnableMap(
        {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "context": _search_query | retriever,
        }
    ).with_types(input_type=ChatHistory)

    return retriever_chain

def get_qa_chain(prompt_hub: str):
    QA_PROMPT = hub.pull(prompt_hub)
    def nvidia_llama_request(retrieved_data):
        client = OpenAI(
            base_url = "https://integrate.api.nvidia.com/v1",
            api_key = nvidia_api_key
        )
        completion = client.chat.completions.create(
            model="meta/llama3-8b-instruct",
            messages=[{"role":"system","content":retrieved_data.messages[0].content,},{"role":"user","content":retrieved_data.messages[1].content}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=True
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response
    
    chain = QA_PROMPT | RunnableLambda(nvidia_llama_request) | StrOutputParser()
    return chain

class RagBot():
    def __init__(self, prompt_hub_qa, prompt_hub_condense,) -> None:
        self.qa_chain = get_qa_chain(prompt_hub_qa)
        self.retrival_chain = get_retrieval_chain(prompt_hub_condense)
        
    def chat(self, request: dict, history, evaluate = False):
        query = request['messages']

        input_data = {"question": query, "chat_history": history}
        retrieved_data = self.retrival_chain.invoke(input_data)

        query_vector = self.get_query_vector(query)
        similar_documents_scores = self.retrieve_similar_documents(query_vector, top_k=1)

        print(similar_documents_scores[0][1])
        url = None
        title = None

        if 'context' in retrieved_data and retrieved_data['context'] and similar_documents_scores[0][1] > 0.8:
            url = retrieved_data['context'][0].metadata['url']
            title = retrieved_data['context'][0].metadata['title']

        source_documents = [doc.to_json()["kwargs"] for doc in retrieved_data["context"]]
        camelized_source_documents = json.dumps(humps.camelize(source_documents))  # Convert dicts to json camel case

        def process_output(output_iterator: Iterator[str]):
            for chunk in output_iterator:
                yield chunk
            yield f"##SOURCE_DOCUMENTS##{camelized_source_documents}"

        stream_chain = self.qa_chain | process_output
        ai_response = stream_chain.invoke(retrieved_data)
        ai_response = ai_response.split("##SOURCE_DOCUMENTS##")[0]
        ai_response = ai_response.replace("\n", "<br>")
        if url and title:
            ai_response += "<br>" + f'<a href="{url}">根據您的描述，可以參考以下資料: {title}</a>'

        if evaluate:
            return ai_response, retrieved_data
        else:
            return ai_response

    def get_query_vector(self, text):
        if not hasattr(self, 'embeddings'):
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002", openai_api_key=openai_api_key
            )
        vector = self.embeddings.embed_query(text)
        return vector

    def retrieve_similar_documents(self, query_vector, top_k=1):
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

        # Get a reference to the specific index
        index = pinecone.Index(pinecone_index)

        # Execute the query
        query_vector = [query_vector]
        results = index.query(vector=query_vector, top_k=top_k)

        # Return the results containing document IDs and similarity scores
        return [(match['id'], match['score']) for match in results['matches']]
