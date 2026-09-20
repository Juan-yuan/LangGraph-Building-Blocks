import os
from pathlib import Path


def load_project_env() -> None:
    env_path = Path(__file__).resolve().parent / "evaluation" / ".env"
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value

    os.environ["LANGFUSE_SECRET_KEY"] = values["LANGFUSE_INIT_PROJECT_SECRET_KEY"]
    os.environ["LANGFUSE_PUBLIC_KEY"] = values["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"]
    os.environ["LANGFUSE_BASE_URL"] = values.get("NEXTAUTH_URL", "http://localhost:3000")


load_project_env()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langfuse import get_client
from langfuse.langchain import CallbackHandler

model = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    base_url="http://localhost:11434/",
    reasoning=False,
)

agent = create_agent(
    model=model,
    system_prompt="You are a capable AI assistant.",
)


if __name__ == "__main__":
    langfuse = get_client()
    langfuse_handler = CallbackHandler()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is 1 + 1?",
                }
            ]
        },
        config={"callbacks": [langfuse_handler]},
    )
    print(result["messages"][-1].content)
    langfuse.flush()
    langfuse.shutdown()
