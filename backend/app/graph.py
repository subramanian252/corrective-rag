"""Corrective RAG graph adapted directly from notebooks/crag.ipynb."""

import os
import re
import time
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from pinecone import Pinecone, ServerlessSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

embeddings = OpenAIEmbeddings(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

pc = Pinecone(api_key=os.getenv("PINECONE_DB"))
if not pc.has_index("agenticrag"):
    print("Creating index...")
    pc.create_index(
        name="agenticrag",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        timeout=30,
    )

    while not pc.describe_index("agenticrag").status["ready"]:
        time.sleep(1)

    index = pc.Index("agenticrag")
    vector_store_pine = PineconeVectorStore(index=index, embedding=embeddings)
    loader = DirectoryLoader(
        str(PROJECT_ROOT / "backend" / "data"),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    vector_store_pine.add_documents(chunks)
else:
    print("Index already exists")
    index = pc.Index('agenticrag')
    vector_store_pine = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store_pine.as_retriever(search_kwargs={"k": 6})


llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)


class CRAGState(TypedDict):
    question: str
    context: list[Document]
    answer: object
    strips: list[str]
    kept_strips: list[str]
    refined_context: str
    good_docs: list[str]
    verdict: str
    web_search_results: list[dict]


def retriever_node(state: CRAGState):
    contexts = retriever.invoke(state["question"])
    return {"context": contexts}


class EvaluateDocs(BaseModel):
    score: float


evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "determine if this document is a good enough document for the question "
            "and give a score of how confident you are in your answer (0-1), the range "
            "can be float. for a completely irrelevant document give a score of 0 and "
            "for a completely relevant document give a score of 1",
        ),
        ("human", "the question: {question} and the document content: {content}"),
    ]
)
evaluator_chain = evaluator_prompt | llm.with_structured_output(EvaluateDocs)

UPPER_THRESHOLD = 0.8
LOWER_THRESHOLD = 0.3


def evaluator_node(state: CRAGState):
    good_docs = []
    contexts = [doc.page_content for doc in state["context"]]
    scored_docs = {}

    for content in contexts:
        score = evaluator_chain.invoke(
            {"question": state["question"], "content": content}
        ).score
        if score > LOWER_THRESHOLD:
            good_docs.append(content)
            scored_docs[content] = score

    for _, score in scored_docs.items():
        if score > UPPER_THRESHOLD:
            return {"good_docs": good_docs, "verdict": "correct"}

    if len(good_docs) == 0:
        return {"good_docs": [], "verdict": "incorrect"}
    return {"good_docs": good_docs, "verdict": "partial"}


def evaluator_router(state: CRAGState):
    if state["verdict"] == "correct":
        return "refine"
    return "search"


tavily_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
)


class RewriteQuery(BaseModel):
    query: str = Field(description="The rewritten query")


query_llm = llm.with_structured_output(RewriteQuery)


def search_node(state: CRAGState):
    rewrite_query = query_llm.invoke(
        [
            SystemMessage(
                content="Rewrite the user's question to be more specific and detailed "
                "for a web search, make sure it is detailed but also concise, just give "
                "the question nothing else is needed"
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    result = tavily_tool.invoke(rewrite_query.query)
    return {"web_search_results": result["results"]}


class FilterResult(BaseModel):
    keep_indexes: list[int]


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Select sentences that are useful for answering the question. Keep direct "
            "answers, important explanations, definitions, and useful examples. Drop "
            "only clearly irrelevant or noisy sentences.",
        ),
        ("human", "Question: {question}\n\nSentences:\n{sentences}"),
    ]
)
filter_chain = filter_prompt | llm.with_structured_output(FilterResult)


def split_sentences(text: str) -> list[str]:
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])|\n(?=[A-Z])", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 10]


def refine_node(state: CRAGState):
    context = list(state["good_docs"])
    if state.get("web_search_results"):
        for web_result in state["web_search_results"]:
            context.append(web_result["content"])

    sentences = split_sentences("\n".join(context))
    numbered_sentences = "\n".join(
        f"{i}: {sentence}" for i, sentence in enumerate(sentences)
    )
    result = filter_chain.invoke(
        {"question": state["question"], "sentences": numbered_sentences}
    )
    kept_sentences = [
        sentences[i] for i in result.keep_indexes if 0 <= i < len(sentences)
    ]
    return {
        "refined_context": "\n".join(kept_sentences),
        "strips": sentences,
        "kept_strips": kept_sentences,
    }


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the context, if you don't know the answer, say 'I don't know'.",
        ),
        ("human", "the question: {question} and the context: {context}"),
    ]
)
llm_chain = prompt | llm


def generate_node(state: CRAGState):
    answer = llm_chain.invoke(
        {"question": state["question"], "context": state["refined_context"]}
    )
    return {"answer": answer}


graph = StateGraph(CRAGState)
graph.add_node("retriever", retriever_node)
graph.add_node("evaluator", evaluator_node)
graph.add_node("refine", refine_node)
graph.add_node("search", search_node)
graph.add_node("generate", generate_node)
graph.set_entry_point("retriever")
graph.add_edge("retriever", "evaluator")
graph.add_conditional_edges(
    "evaluator", evaluator_router, {"refine": "refine", "search": "search"}
)
graph.add_edge("search", "refine")
graph.add_edge("refine", "generate")
graph.add_edge("generate", END)

app = graph.compile()
