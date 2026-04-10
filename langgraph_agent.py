import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    options = ["apple", "tesla", "both", "none"]
    router_prompt = f"""
You are a routing classifier for a financial RAG system.
Classify the user question into one of exactly these labels: {options}.

Rules:
1) "apple" if it only asks about Apple.
2) "tesla" if it only asks about Tesla.
3) "both" if it compares or asks both companies.
4) "none" if it does not ask Apple/Tesla financial content.

Return ONLY valid JSON and nothing else:
{{"datasource":"apple|tesla|both|none"}}

User Question: {question}
"""

    try:
        response = llm.invoke(router_prompt)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        res_json = json.loads(content)
        target = str(res_json.get("datasource", "none")).lower().strip()
        if target not in options:
            target = "none"
    except Exception as e:
        print(colored(f"[Warning] Error parsing router output: {e}. Using rule fallback.", "yellow"))
        q = question.lower()
        has_apple = "apple" in q
        has_tesla = "tesla" in q
        if has_apple and has_tesla:
            target = "both"
        elif has_apple:
            target = "apple"
        elif has_tesla:
            target = "tesla"
        else:
            target = "none"

    print(colored(f"[Router] Routing to: {target}", "cyan"))

    docs_content = ""
    targets_to_search = []
    if target == "both":
        targets_to_search = list(FILES.keys())
    elif target in FILES:
        targets_to_search = [target]
    
    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            source_name = t.capitalize()
            docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("--- GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a strict binary relevance grader.
Decide whether retrieved context is sufficient and relevant to answer the user question.
If relevant enough, output yes.
If missing, weak, or off-topic, output no.
Return ONLY one word: yes or no."""
    
    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved document context: \n\n {documents} \n\n User question: {question}")
    ]
    
    response = llm.invoke(msg)
    content = response.content.strip().lower()
    
    grade = "yes" if content == "yes" else "no"
    print(f"   Relevance Grade: {grade}")
    return {"needs_rewrite": grade}

@retry_logic
def generate_node(state: AgentState):
    print(colored("--- GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst.\n"
                   "Use ONLY the provided context.\n"
                   "Final answer must be in English.\n"
                   "Always distinguish 2024, 2023, and 2022 columns.\n"
                   "If the exact requested data is not in context, reply exactly: I don't know.\n"
                   "Always cite sources in brackets such as [Source: Apple] or [Source: Tesla].\n"
                   "Do not hallucinate.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    msg = [ 
        HumanMessage(content=f"The previous search for '{question}' yielded irrelevant results.\n"
                             f"Rewrite it into a specific financial query using accounting terms "
                             f"(for example: Total net sales, Research and development expenses, "
                             f"Cost of sales - Services, Capital expenditures, Gross margin).\n"
                             f"Preserve company names and year constraints.\n"
                             f"Output ONLY the rewritten question text.")
    ]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2: 
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- RUNNING LEGACY AGENT (ReAct) ---", "magenta"))
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools.retriever import create_retriever_tool
    from langchain.tools.render import render_text_description

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(create_retriever_tool(
            retriever, 
            f"search_{key}_financials", 
            f"Searches {key.capitalize()}'s financial data."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    template = """You are a careful financial analysis agent that MUST follow the ReAct loop.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Rules:
1) Final Answer must be English only.
2) Distinguish 2024, 2023, and 2022 columns carefully.
3) If exact requested 2024 value cannot be found from observations, reply exactly: I don't know.
4) Never fabricate numbers.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"