from langgraph.graph import StateGraph, START
from langgraph.types import Send
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import BaseModel, Field
import operator
from langchain.chat_models import init_chat_model


class Section(BaseModel):
    num: int = Field(description="Chapter number")
    name: str = Field(
        description='Chapter title, using the format "【Chapter 3】 The Mist Returns"'
    )
    description: str = Field(
        description='Chapter description. It starts with the chapter title, '
                    'followed by the chapter content outline. '
                    'The chapter title should use the format '
                    '"【Chapter 3】 The Mist Returns" and appear on a separate line.'
    )


class Sections(BaseModel):
    sections: list[Section] = Field(
        description="The individual chapters of the novel"
    )


class State(TypedDict):
    storyLine: str
    sections: list[Section]
    completedSections: Annotated[list, operator.add]
    novel: str


class WorkerState(TypedDict):
    section: Section
    completedSections: Annotated[list, operator.add]


model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/"
)


def getWholeStory(state):
    prompt = (
        "You are a famous novelist writing an exciting detective novel. "
        "First, provide a story outline of approximately 1,000 Chinese characters."
    )

    state["storyLine"] = model.invoke(prompt).content

    return state


def orchestrate(state):
    planner = model.with_structured_output(Sections)

    result = planner.invoke(
        """You are a famous novelist writing an exciting detective novel.
        Based on the following [Story Outline], divide the novel into 10 chapters
        and provide the plot development for each chapter.
        Approximately 1,000 Chinese characters.

        [Story Outline]
        """ + state["storyLine"]
    )

    state["sections"] = result.sections

    for section in state["sections"]:
        print(str(section.name + "        " + section.description))

    return state


def work(state: WorkerState):
    prompt = """You are a famous novelist writing an exciting detective novel.
    Based on the [Chapter Title] and [Chapter Overview] provided below,
    write one complete chapter.

    The chapter should start with the chapter title, followed by the chapter content.
    The chapter title should use the format "【Chapter 3】 The Mist Returns"
    and appear on a separate line.

    The chapter number is """ + str(state["section"].num) + """.
    The chapter should be approximately 1,500 Chinese characters long.

    [Chapter Title]
    """ + state["section"].name + """

    [Chapter Overview]
    """ + state["section"].description

    result = model.invoke(prompt)

    return {
        "num": state["section"].num,
        "name": state["section"].name,
        "description": state["section"].description,
        "completedSections": [
            {
                "num": state["section"].num,
                "content": result.content
            }
        ]
    }


def synthesizer(state):
    completedSections = sorted(
        state["completedSections"],
        key=lambda completeSection: completeSection["num"]
    )

    novel = "\n\n".join(
        [completeSection["content"] for completeSection in completedSections]
    )

    state["novel"] = novel

    return state


def assignWorkers(state: State):
    return [
        Send("work", {"section": section})
        for section in state["sections"]
    ]


def buildGraph():
    graphBuilder = StateGraph(State)

    graphBuilder.add_node("getWholeStory", getWholeStory)
    graphBuilder.add_node("orchestrate", orchestrate)
    graphBuilder.add_node("work", work)
    graphBuilder.add_node("synthesizer", synthesizer)

    graphBuilder.add_edge(START, "getWholeStory")
    graphBuilder.add_edge("getWholeStory", "orchestrate")
    graphBuilder.add_conditional_edges(
        "orchestrate",
        assignWorkers,
        ["work"]
    )
    graphBuilder.add_edge("work", "synthesizer")

    graph = graphBuilder.compile()

    return graph


if __name__ == "__main__":
    graph = buildGraph()

    from z_langgraph_basic import showGraph

    showGraph.showGraphInCode(graph, "graph.jpg")

    state: State = {}

    result = graph.invoke(state)

    print(result["novel"])