from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict
from typing import Annotated
import random
from langchain.chat_models import init_chat_model


def updateReceiveDate(left, right):
    return max(left, right)


class State(TypedDict):
    sendDate: int
    transTime: int
    receiveDate: Annotated[int, updateReceiveDate]
    returnMessage: str


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/"
)


def getSendDate(state):
    sendDate = random.randint(1, 25)
    state["sendDate"] = sendDate
    return state


def getTransTime(state):
    transTime = random.randint(3, 5)
    state["transTime"] = transTime
    return state


def sendMessage(state):
    prompt = (
        "You are Wang Zhongqi, a salesperson at Jiangnan Machinery Factory. "
        "Customer Chairman Zhang has ordered a batch of machinery from our factory "
        "and hopes to receive the goods on September "
        + str(state["receiveDate"])
        + ". The earliest shipping date from the factory is September "
        + str(state["sendDate"])
        + ", and the transportation time is "
        + str(state["transTime"])
        + " days. Please send an appropriate response to the customer."
    )

    state["returnMessage"] = model.invoke(prompt).content
    return state


def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getSendDate", getSendDate)
    graphBuilder.add_node("getTransTime", getTransTime)
    graphBuilder.add_node("sendMessage", sendMessage)

    graphBuilder.add_edge(START, "getSendDate")
    graphBuilder.add_edge(START, "getTransTime")
    graphBuilder.add_edge(
        ["getSendDate", "getTransTime"],
        "sendMessage"
    )

    graph = graphBuilder.compile()

    return graph


if __name__ == "__main__":
    graph = buildGraph()

    from z_langgraph_basic import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {
        "receiveDate": 5
    }

    result = graph.invoke(state)

    print(result)