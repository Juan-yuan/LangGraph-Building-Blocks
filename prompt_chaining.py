import sys
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from z_langgraph_basic import show_graph


class State(TypedDict):
    topic: str
    outline: str
    draft: str
    paper: str


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
)


def get_outline(state: State) -> State:
    prompt = (
        f'Write an outline for an analysis report on the topic "{state["topic"]}". '
        "Include these main sections: historical background, current situation, "
        "cause analysis, solutions, and trend assessment."
    )
    state["outline"] = model.invoke(prompt).content
    print("outline  " + "*" * 80)
    print(state["outline"])
    return state


def get_draft(state: State) -> State:
    prompt = (
        "Based on the [Outline], write a complete analysis report. "
        "The language should be fluent and the logic clear. "
        "Follow the outline closely and write about 2000 words.\n\n"
        f"[Outline]\n{state['outline']}"
    )
    state["draft"] = model.invoke(prompt).content
    print("draft  " + "*" * 80)
    print(state["draft"])
    return state


def get_paper(state: State) -> State:
    prompt = (
        "Polish the [Analysis Report] below. Check grammar, improve word choice, "
        "and adjust sentence structure. The writing should be natural, fluent, "
        "logically clear, vivid, and concise. Avoid a stiff, formulaic AI style, "
        "and keep the original meaning unchanged.\n\n"
        f"[Analysis Report]\n{state['draft']}"
    )
    state["paper"] = model.invoke(prompt).content
    print("paper  " + "*" * 80)
    print(state["paper"])
    return state


def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("get_outline", get_outline)
    graph_builder.add_node("get_draft", get_draft)
    graph_builder.add_node("get_paper", get_paper)
    graph_builder.add_edge(START, "get_outline")
    graph_builder.add_edge("get_outline", "get_draft")
    graph_builder.add_edge("get_draft", "get_paper")
    graph_builder.add_edge("get_paper", END)
    return graph_builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    show_graph.show_graph_in_code(graph, "graph.jpg")

    state: State = {"topic": "Report on the drug crisis in Mexico"}
    result = graph.invoke(state)
    print("final  " + "*" * 80)
    print(result)
