from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langgraph.checkpoint.memory import MemorySaver
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.prebuilt import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

#
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"AIE1 - LangGraph - {uuid4().hex[0:8]}"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

#Instantiate a Simple Retrieval Chain using LCEL
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# # Load the document pertaining to a particular topic
docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()

# Split the dpocument into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350, chunk_overlap=50
)

chunked_documents = text_splitter.split_documents(docs)
#
# Instantiate the Embedding Model
embeddings = OpenAIEmbeddings()
# Create Index- Load document chunks into the vectorstore
faiss_vectorstore = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)
# Create a retriver
retriever = faiss_vectorstore.as_retriever()

#Generate a Rag Prompt
RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

#Instantiate the LLM
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

#Build LCEL RAG Chain
retrieval_augmented_generation_chain = (
       {"context": itemgetter("question") 
    | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)
#
retrieval_augmented_generation_chain

#Ask Query
retrieval_augmented_generation_chain.invoke({"question" : "What is Retrieval Augmented Generation?"})

#Tool Belt
tool_belt = [
    DuckDuckGoSearchRun(),
    ArxivQueryRun()
    
]

tool_executor = ToolExecutor(tool_belt)

#Instantiate Openai Function Calling
model = ChatOpenAI(temperature=0)
#
functions = [convert_to_openai_function(t) for t in tool_belt]
model = model.bind_functions(functions)

#Leverage LangGraph
class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]

def call_model(state):
  messages = state["messages"]
  response = model.invoke(messages)
  return {"messages" : [response]}

def call_tool(state):
    last_message = state["messages"][-1]

    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"])
    )

    # Track which tool is being used
    tool_name = action.tool
    print("Tool Used: ", tool_name)

    response = tool_executor.invoke(action)

    # Print the response structure before using it
    print("Tool response:", response)

    function_message = FunctionMessage(content=str(response), name=action.tool)

    return {"messages": [function_message]}

#Build the Workflow
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.nodes

#Set the Entry point
workflow.set_entry_point("agent")

#Build a Conditional Edge for Routing
def should_continue(state):
  last_message = state["messages"][-1]

  if "function_call" not in last_message.additional_kwargs:
    return "end"

  return "continue"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue" : "action",
        "end" : END
    }
)

#Finally connect the conditional edge to the agent node and action node
workflow.add_edge("action", "agent")
# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()
#Compile the workflow
app = workflow.compile(checkpointer=checkpointer)


#Invoke the LangGraph- Ask Query
final_state = app.invoke(
        {"messages": [HumanMessage(content="Who is elon musk")]},
        config={"configurable": {"thread_id": 42}}
    )
print(final_state)
