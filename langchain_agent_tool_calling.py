from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


def getTrainSchedule(queryDate, start, end):
    """Query train schedules based on the specified date, departure city, and destination city."""
    print("Getting train schedules")

    result = [
        ["D81", "12:24", "14:30", "Beijing West Railway Station", "540"],
        ["K4427", "15:38", "21:30", "Beijing Railway Station", "220"]
    ]

    resultStr = (
        "There are " + str(len(result)) +
        " trains available from " + start +
        " to " + end +
        " on " + queryDate + ":\n"
    )

    for res in result:
        resultStr += (
            res[0] + " train: Departure time: " + res[1] +
            ", Arrival time: " + res[2] +
            ", Departure station: " + res[3] +
            ", Ticket price: " + res[4] + "\n"
        )

    return resultStr


def getAvailableHotel(queryDate, location):
    """Query available hotels based on the specified date and city."""
    print("Getting available hotels at the destination")

    result = [
        ["Regent Hotel", "Five-star", "King Room", "1200"],
        ["Regent Grand Hotel", "Two-star", "Standard Room", "300"]
    ]

    resultStr = (
        "There are " + str(len(result)) +
        " hotels available for booking in " + location +
        " on " + queryDate + ":\n"
    )

    for res in result:
        resultStr += (
            res[0] + ", Rating: " + res[1] +
            ", Room type: " + res[2] +
            ", Price: " + res[3] + "\n"
        )

    return resultStr


toolList = [getTrainSchedule, getAvailableHotel]


model = init_chat_model(
    model="qwen2.5:7b",
    # model="qwen3.5:9b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
    reasoning=False
)


agent = create_agent(
    model=model,
    tools=toolList,
    system_prompt="",
)


if __name__ == "__main__":
    prompt = (
        "You are a reliable personal AI assistant. "
        "I want to travel from Beijing to Qingdao on September 15. "
        "Please help me arrange the train and hotel. "
        "The trip should be fast and comfortable."
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    print(result["messages"][-1].content)