from langgraph.graph import END, START, StateGraph

from Linki.graph.nodes import (
    chat_responder_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    final_node,
    intent_route_fn,
    intent_router_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from Linki.graph.state import LinkiGraphState


def build_entry_workflow():
    """Build an intent-routing graph that determines whether the user input is
    conversational or requires the full task workflow.
    """

    graph = StateGraph(LinkiGraphState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("chat_responder", chat_responder_node)

    graph.add_edge(START, "intent_router")

    graph.add_conditional_edges(
        "intent_router",
        intent_route_fn,
        {
            "chat_responder": "chat_responder",
            # When routed to planner, hand execution over to the main workflow.
            "planner": END,
        },
    )

    graph.add_edge("chat_responder", END)

    return graph.compile()


def build_workflow():
    graph = StateGraph(LinkiGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {
            "final": "final",
            "planner": "planner",
        },
    )
    graph.add_edge("final", END)
    return graph.compile()


def build_complex_workflow():
    graph = StateGraph(LinkiGraphState)

    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")

    graph.add_conditional_edges(
        "context_monitor",
        context_monitor_route,
        {
            "context_compressor": "context_compressor",
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )

    graph.add_conditional_edges(
        "context_compressor",
        context_compressor_route,
        {
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )

    # Verification must also pass through the context monitor.
    graph.add_edge("verifier", "context_monitor")

    graph.add_edge("final", END)

    return graph.compile()
