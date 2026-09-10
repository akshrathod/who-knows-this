"""
multi_agent.py
--------------
LangGraph multi-agent pipeline.
Wires all agents into a stateful graph with conditional routing.

Flow:
  START -> supervisor -> extractor -> supervisor -> graph_agent
        -> supervisor -> neo4j_execute -> supervisor -> END
                                       -> graph_agent (self-healing retry)
"""

import json
import time
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from ingestion.ingester import ingest
from security.pii_gateway import redact
from agents.extractor import extract
from agents.graph_agent import generate_cypher
from agents.supervisor import decide, MAX_RETRIES
from graph.neo4j_loader import execute_batch, get_driver, setup_constraints

load_dotenv()

# Agent State

class AgentState(TypedDict):
    doc_id          : str
    raw_text        : str
    sanitized_text  : str
    pii_audit       : dict
    profile         : dict
    cypher_queries  : list
    executed        : bool
    neo4j_error     : str
    failed_queries  : list
    retry_count     : int
    status          : str
    latency_log     : dict
    token_log       : dict


def initial_state(doc: dict) -> AgentState:
    """Build a fresh state object from an ingested document dict."""
    return AgentState(
        doc_id         = doc["doc_id"],
        raw_text       = doc["text"],
        sanitized_text = "",
        pii_audit      = {},
        profile        = {},
        cypher_queries = [],
        executed       = False,
        neo4j_error    = "",
        failed_queries = [],
        retry_count    = 0,
        status         = "ingested",
        latency_log    = {},
        token_log      = {}
    )

# Node Functions

def pii_node(state: AgentState) -> AgentState:
    """Run PII redaction on raw text. Always runs first before any LLM call."""
    print(f"  [PII Gateway] Redacting {state['doc_id'][:50]}")
    t = time.time()

    result = redact(state["raw_text"], doc_id=state["doc_id"])

    state["sanitized_text"] = result["sanitized_text"]
    state["pii_audit"]      = result["audit"]
    state["status"]         = "sanitized"
    state["latency_log"]["pii"] = round(time.time() - t, 2)

    print(f"    Redacted {result['audit']['items_redacted']} items in {state['latency_log']['pii']}s")
    return state


def supervisor_node(state: AgentState) -> AgentState:
    """Read current state and write routing decision into status."""
    decision = decide(state)
    state["status"] = decision
    print(f"  [Supervisor] -> {decision}")
    return state


def extractor_node(state: AgentState) -> AgentState:
    """Extract structured researcher profile from sanitized text."""
    print(f"  [Extractor] Extracting profile")
    t = time.time()

    result = extract(state["sanitized_text"], doc_id=state["doc_id"])

    state["profile"]   = result["profile"]
    state["status"]    = "extracted"
    state["latency_log"]["extract"] = round(time.time() - t, 2)
    state["token_log"]["extract"]   = {
        "input" : result["input_tokens"],
        "output": result["output_tokens"]
    }

    print(f"    Extracted {len(result['profile'].get('researchers', []))} researchers in {state['latency_log']['extract']}s")
    return state


def graph_agent_node(state: AgentState) -> AgentState:
    """Generate Cypher queries from extracted profile. Handles retry with error context."""
    is_retry = bool(state.get("neo4j_error"))

    if is_retry:
        print(f"  [Graph Agent] Retry {state['retry_count']} - fixing failed queries")
    else:
        print(f"  [Graph Agent] Generating Cypher queries")

    t = time.time()

    result = generate_cypher(
        profile        = state["profile"],
        doc_id         = state["doc_id"],
        error          = state.get("neo4j_error"),
        failed_queries = state.get("failed_queries")
    )

    state["cypher_queries"]  = result["queries_list"]
    state["neo4j_error"]     = ""
    state["failed_queries"]  = []
    state["executed"]        = False
    state["status"]          = "cypher_ready"
    state["latency_log"]["graph"] = round(time.time() - t, 2)
    state["token_log"]["graph"]   = {
        "input" : result["input_tokens"],
        "output": result["output_tokens"]
    }

    print(f"    Generated {len(result['queries_list'])} queries in {state['latency_log']['graph']}s")
    return state


def neo4j_node(state: AgentState) -> AgentState:
    """Execute Cypher queries against Neo4j. Captures errors for self-healing."""
    print(f"  [Neo4j] Executing {len(state['cypher_queries'])} queries")
    t = time.time()

    result = execute_batch(state["cypher_queries"], doc_id=state["doc_id"])

    state["executed"] = True
    state["latency_log"]["neo4j"] = round(time.time() - t, 2)

    if result["success"]:
        state["status"]      = "complete"
        state["neo4j_error"] = ""
        state["failed_queries"] = []
        print(f"    All {result['success_count']} queries succeeded in {state['latency_log']['neo4j']}s")
    else:
        state["neo4j_error"]    = result["errors"][0] if result["errors"] else "Unknown error"
        state["failed_queries"] = result["failed_queries"]
        state["retry_count"]    += 1
        state["executed"]       = False
        state["status"]         = "error"
        print(f"    Failed. Error: {state['neo4j_error'][:80]}")
        print(f"    Retry count: {state['retry_count']}/{MAX_RETRIES}")

    return state

# Routing Logic

def route(state: AgentState) -> str:
    """
    Conditional edge function for LangGraph.
    Reads state status and returns the next node name.
    """
    status = state.get("status")

    if status == "extract":
        return "extractor"
    elif status == "graph":
        return "graph_agent"
    elif status in ("execute", "cypher_ready"):
        return "neo4j"
    elif status == "retry" and state.get("retry_count", 0) < MAX_RETRIES:
        return "graph_agent"
    else:
        return END

# Graph Construction

def build_graph() -> StateGraph:
    """Wire all nodes and edges into a LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("pii",        pii_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("extractor",  extractor_node)
    graph.add_node("graph_agent",graph_agent_node)
    graph.add_node("neo4j",      neo4j_node)

    graph.set_entry_point("pii")

    graph.add_edge("pii",        "supervisor")
    graph.add_edge("extractor",  "supervisor")
    graph.add_edge("graph_agent","supervisor")
    graph.add_edge("neo4j",      "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route,
        {
            "extractor" : "extractor",
            "graph_agent": "graph_agent",
            "neo4j"     : "neo4j",
            END         : END
        }
    )

    return graph.compile()

# Pipeline Runner

def run_pipeline(docs: list[dict]) -> list[dict]:
    """
    Run the full multi-agent pipeline across a list of documents.
    Returns a list of final states for benchmarking and evaluation.
    """
    driver = get_driver()
    setup_constraints(driver)
    driver.close()

    pipeline = build_graph()
    print(pipeline.get_graph().draw_mermaid())
    results  = []
    total_start = time.time()

    print(f"\nRunning multi-agent pipeline on {len(docs)} documents\n")
    print("=" * 60)

    for i, doc in enumerate(docs):
        print(f"\n[{i+1}/{len(docs)}] {doc['doc_id'][:60]}")
        doc_start = time.time()

        state = initial_state(doc)

        try:
            final_state = pipeline.invoke(state)
            final_state["latency_log"]["total"] = round(time.time() - doc_start, 2)
            results.append(final_state)

            total_tokens = sum(
                v.get("input", 0) + v.get("output", 0)
                for v in final_state["token_log"].values()
                if isinstance(v, dict)
            )

            print(f"  Status  : {final_state['status']}")
            print(f"  Latency : {final_state['latency_log']['total']}s")
            print(f"  Tokens  : {total_tokens}")
            print(f"  Retries : {final_state['retry_count']}")

        except Exception as e:
            print(f"  Pipeline error: {e}")
            results.append({"doc_id": doc["doc_id"], "status": "pipeline_error", "error": str(e)})

    total_time = round(time.time() - total_start, 2)

    print(f"\n{'=' * 60}")
    print(f"PIPELINE SUMMARY")
    print(f"{'=' * 60}")

    fully_complete = sum(
        1 for r in results
        if r.get("status") != "pipeline_error" and not r.get("neo4j_error")
    )
    partial_complete = sum(
        1 for r in results
        if r.get("status") == "end" and r.get("neo4j_error") and r.get("retry_count", 0) >= MAX_RETRIES
    )
    pipeline_errors = sum(
        1 for r in results
        if r.get("status") == "pipeline_error"
    )

    avg_latency = round(
        sum(r.get("latency_log", {}).get("total", 0) for r in results) / len(results), 2
    ) if results else 0

    print(f"Documents  : {len(docs)}")
    print(f"Fully completed   : {fully_complete}/{len(docs)}")
    print(f"Partial (retries exhausted): {partial_complete}/{len(docs)}")
    print(f"Pipeline errors   : {pipeline_errors}/{len(docs)}")
    print(f"Avg latency: {avg_latency}s per document")
    print(f"Total time : {total_time}s")

    output_path = Path("data/pipeline_results.json")
    with open(output_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "raw_text"} for r in results],
            f, indent=2
        )
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    docs = ingest("s3", limit=6)
    run_pipeline(docs)
