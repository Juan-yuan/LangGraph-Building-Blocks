from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model


class State(TypedDict):
    topic: str
    aspect: str
    faction: str
    debate: str


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/"
)


def getFaction(state):
    prompt = """You are a master of Chinese traditional philosophy. 
You believe that the viewpoint " """ + state["topic"] + """ " is """ + state["aspect"] + """.
Among the three schools of thought ["Confucianism", "Legalism", "Taoism"], 
which school would you most likely agree with?
Only output the name of the school. Do not output any other characters."""

    state["faction"] = model.invoke(prompt).content

    return state


def selectFaction(state):
    if state["faction"] == "Confucianism":
        return "Confucian"
    elif state["faction"] == "Legalism":
        return "Legalists"
    elif state["faction"] == "Taoism":
        return "Taoism"

    return None


def getDebateFromConfucian(state):
    prompt = """You are a master of Chinese traditional philosophy.
You believe that the viewpoint " """ + state["topic"] + """ " is """ + state["aspect"] + """.
This is a Confucian viewpoint. Please use Confucian philosophy to provide a detailed argument on this topic."""

    state["debate"] = model.invoke(prompt).content

    return state


def getDebateFromLegalists(state):
    prompt = """You are a master of Chinese traditional philosophy.
You believe that the viewpoint " """ + state["topic"] + """ " is """ + state["aspect"] + """.
This is a Legalist viewpoint. Please use Legalist philosophy to provide a detailed argument on this topic."""

    state["debate"] = model.invoke(prompt).content

    return state


def getDebateFromTaoism(state):
    prompt = """You are a master of Chinese traditional philosophy.
You believe that the viewpoint " """ + state["topic"] + """ " is """ + state["aspect"] + """.
This is a Taoist viewpoint. Please use Taoist philosophy to provide a detailed argument on this topic."""

    state["debate"] = model.invoke(prompt).content

    return state


def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getFaction", getFaction)
    graphBuilder.add_node("getDebateFromConfucian", getDebateFromConfucian)
    graphBuilder.add_node("getDebateFromLegalists", getDebateFromLegalists)
    graphBuilder.add_node("getDebateFromTaoism", getDebateFromTaoism)

    graphBuilder.add_edge(START, "getFaction")

    graphBuilder.add_conditional_edges(
        "getFaction",
        selectFaction,
        {
            "Confucian": "getDebateFromConfucian",
            "Legalists": "getDebateFromLegalists",
            "Taoism": "getDebateFromTaoism"
        }
    )

    graph = graphBuilder.compile()

    return graph


if __name__ == "__main__":
    graph = buildGraph()

    from z_langgraph_basic import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {
        "topic": "The highest form of goodness is like water",
        "aspect": "correct"
    }

    result = graph.invoke(state)
    print(result)