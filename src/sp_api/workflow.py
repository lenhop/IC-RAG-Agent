"""LangGraph workflow for Seller Operations Agent."""
from typing import Any, Dict, List, TypedDict

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    StateGraph = None
    END = None


class SellerAgentState(TypedDict):
    query: str
    session_id: str
    intent: str
    selected_tools: List[str]
    agent_result: str
    formatted_response: str


def _build_workflow(agent):
    """Build workflow with agent injected via closure."""
    def classify_intent(state: SellerAgentState) -> Dict[str, Any]:
        intent = agent._classify_intent(state["query"])
        return {"intent": intent}

    def select_tools(state: SellerAgentState) -> Dict[str, Any]:
        intent = state.get("intent", "query")
        tools = agent.list_tools()
        names = [t["name"] for t in tools]
        if intent == "query":
            selected = [n for n in names if n in ("product_catalog", "inventory_summary", "list_orders", "order_details", "financials")]
        elif intent == "action":
            selected = [n for n in names if n in ("create_shipment",)]
        elif intent == "report":
            selected = [n for n in names if n in ("request_report",)]
        else:
            selected = names[:5]
        return {"selected_tools": selected if selected else names[:3]}

    def execute_react_loop(state: SellerAgentState) -> Dict[str, Any]:
        result = agent.query(state["query"], state["session_id"])
        return {"agent_result": result}

    def format_response(state: SellerAgentState) -> Dict[str, Any]:
        return {"formatted_response": state.get("agent_result", "")}

    def store_memory(state: SellerAgentState) -> Dict[str, Any]:
        return {}

    if StateGraph is None:
        return None
    workflow = StateGraph(SellerAgentState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("execute_react_loop", execute_react_loop)
    workflow.add_node("format_response", format_response)
    workflow.add_node("store_memory", store_memory)
    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "select_tools")
    workflow.add_edge("select_tools", "execute_react_loop")
    workflow.add_edge("execute_react_loop", "format_response")
    workflow.add_edge("format_response", "store_memory")
    workflow.add_edge("store_memory", END)
    return workflow.compile()


def create_app(agent):
    """Create compiled LangGraph app from agent."""
    w = _build_workflow(agent)
    return w
