from langchain.agents import create_agent
from dotenv import load_dotenv
import os
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()
client = TypeSafeClient(api_key=os.getenv("TYPESAFE_API_KEY"))
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="gpt-4o",
            criteria="Direct lookups, extraction, and localized changes.",
        ),
        "powerful": ModelChoice(
            model="gpt-4o-mini",
            criteria="Architecture and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task.",
)

agent = create_agent("openai:gpt-4o", middleware=[router])

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Prove that there are infinitely many prime numbers.",
            }
        ]
    }
)
print(result["model_route"])