from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
import json


class State(TypedDict):
    topic: str
    article: str
    feedback: str
    qualified: str
    count: int


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/"
)


def generate(state):
    if state.get("feedback"):
        prompt = """Write an argumentative article based on the provided topic.
        Make sure the article is logically rigorous and persuasive.

        The topic is: """ + state["topic"] + """

        You also need to consider the following revision suggestions:
        """ + state["feedback"]
    else:
        # prompt = """Write an argumentative article based on the provided topic.
        # Make sure the article is logically rigorous and persuasive.
        # The topic is: """ + state["topic"]

        prompt = """You are an elementary school student who has no writing skills.
        You are now required to write an argumentative article.
        The topic is: """ + state["topic"]

    result = model.invoke(prompt)

    state["count"] += 1
    state["article"] = result.content

    print("generate  " + "*" * 80)
    print(state["article"])

    return state


def evaluate(state):
    prompt = """Evaluate whether the [Argumentative Article] effectively supports
    the [Topic], and whether the argument is logically rigorous and persuasive.

    If the article does not meet the requirements, provide specific revision
    suggestions.

    Output the result strictly according to the [Specified Format].
    The value of "Qualified" in the [Specified Format] must only be "Yes" or "No".
    Do not output any characters other than those specified in the [Specified Format].

    [Specified Format]
    {"Qualified":"", "Revision Suggestions":""}

    [Topic]
    """ + state["topic"] + """

    [Argumentative Article]
    """ + state["article"]

    result = model.invoke(prompt)

    print("evaluate  " + "*" * 80)
    print(result.content)

    resultJson = json.loads(result.content)

    state["qualified"] = resultJson["Qualified"]
    state["feedback"] = resultJson["Revision Suggestions"]

    return state


def judgement(state):
    if state["count"] >= 5:
        return "accept"
    else:
        if state["qualified"] == "Yes":
            return "accept"
        else:
            return "reject"


def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("generate", generate)
    graphBuilder.add_node("evaluate", evaluate)

    graphBuilder.add_edge(START, "generate")
    graphBuilder.add_edge("generate", "evaluate")

    graphBuilder.add_conditional_edges(
        "evaluate",
        judgement,
        {
            "accept": END,
            "reject": "generate"
        }
    )

    graph = graphBuilder.compile()

    return graph


if __name__ == "__main__":
    graph = buildGraph()

    from z_langgraph_basic import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {
        "topic": "The Japanese economy will rise again in the future",
        "count": 0
    }

    result = graph.invoke(state)

    print("final  " + "*" * 80)
    print(result)