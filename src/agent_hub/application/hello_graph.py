"""M1 — hello-graph.

The smallest possible LangGraph: typed state, two nodes, two edges, compile.
Exists purely to prove the framework wiring end-to-end (state flows, nodes
return partial updates that merge, the compiled app is invokable) before any
LLM or tool enters the picture. Reached via GET /api/agent/ping.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class HelloState(TypedDict):
    message: str
    steps: list[str]


def _greet(state: HelloState) -> dict:
    # Partial update — LangGraph merges it into the running state.
    return {"message": f"hello, {state['message']}", "steps": ["greet"]}


def _shout(state: HelloState) -> dict:
    return {"message": state["message"].upper(), "steps": state["steps"] + ["shout"]}


def build_hello_graph():
    g = StateGraph(HelloState)
    g.add_node("greet", _greet)
    g.add_node("shout", _shout)
    g.add_edge(START, "greet")
    g.add_edge("greet", "shout")
    g.add_edge("shout", END)
    return g.compile()


hello_app = build_hello_graph()
