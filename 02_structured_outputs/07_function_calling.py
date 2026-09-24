"""
Topic: Tool / function calling - how models decide when to use tools

The model never runs your code. You describe tools (name, description,
JSON-schema parameters). For each request the model decides whether to:
  - answer directly, or
  - return one or more "tool calls" (tool name + JSON arguments).
Your code executes the call and sends the result back; the model then
writes the final answer. The tool *description* is what drives the decision.

Part 1 shows only the decision for several queries.
Part 2 runs the full request -> tool -> result -> answer loop.

Run:  python 02_structured_outputs/07_function_calling.py
"""

import json
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL, banner, show

# ---------------------------------------------------------------------------
# Our "backend": plain Python functions with fake data
# ---------------------------------------------------------------------------
ORDERS = {
    "A1001": {"status": "shipped", "carrier": "BlueDart", "eta": "2026-09-26"},
    "A1002": {"status": "processing", "carrier": None, "eta": "2026-09-30"},
}


def get_order_status(order_id: str) -> dict:
    order = ORDERS.get(order_id.upper())
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {"order_id": order_id.upper(), **order}


def get_weather(city: str, unit: str = "celsius") -> dict:
    fake = {"kochi": 29, "bengaluru": 23, "london": 14}
    temp_c = fake.get(city.lower(), 25)
    temp = temp_c if unit == "celsius" else round(temp_c * 9 / 5 + 32)
    return {"city": city, "temperature": temp, "unit": unit, "conditions": "partly cloudy"}


AVAILABLE_FUNCTIONS = {"get_order_status": get_order_status, "get_weather": get_weather}

# ---------------------------------------------------------------------------
# Tool definitions the model sees (OpenAI format, which LiteLLM uses for all providers)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up the current shipping status and ETA of a customer's order. "
                "Use this whenever the user asks where their order is or when it will arrive."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order id, e.g. A1001"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
]

SYSTEM = "You are a helpful assistant for an online store. Use tools when they help."


# ---------------------------------------------------------------------------
# Part 1: what does the model decide to do?
# ---------------------------------------------------------------------------
def show_decisions() -> None:
    banner("Part 1: tool-use decisions")
    queries = [
        "What is the capital of France?",                  # no tool needed
        "Where is my order A1001?",                         # one tool
        "What's the weather in Kochi and London?",          # same tool twice (parallel)
        "Is order A1002 shipped, and is it raining in Bengaluru?",  # two different tools
        "Where is my order?",                               # missing argument -> should ask
    ]
    for query in queries:
        response = litellm.completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
            tools=TOOLS,
            tool_choice="auto",  # let the model decide ("none" / "required" force it)
            temperature=0,
        )
        message = response.choices[0].message
        print(f"\nUser: {query}")
        if message.tool_calls:
            for call in message.tool_calls:
                print(f"  -> TOOL CALL {call.function.name}({call.function.arguments})")
        else:
            print(f"  -> ANSWERS DIRECTLY: {(message.content or '').strip()[:100]}")


# ---------------------------------------------------------------------------
# Part 2: the full loop
# ---------------------------------------------------------------------------
def run_tool_call(call) -> str:
    """Execute one tool call safely and return a JSON string for the model."""
    function = AVAILABLE_FUNCTIONS.get(call.function.name)
    if function is None:
        return json.dumps({"error": f"Unknown tool {call.function.name}"})
    try:
        args = json.loads(call.function.arguments)  # arguments arrive as a JSON *string*
        result = function(**args)
    except (json.JSONDecodeError, TypeError) as e:
        # Send errors back to the model instead of crashing; it can often recover.
        result = {"error": f"Bad arguments: {e}"}
    return json.dumps(result)


def agent_loop(user_message: str, max_steps: int = 5) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}]

    for step in range(1, max_steps + 1):
        response = litellm.completion(model=MODEL, messages=messages, tools=TOOLS, temperature=0)
        message = response.choices[0].message
        messages.append(message)  # keep the assistant's tool-call turn in history

        if not message.tool_calls:
            return message.content  # model is done

        for call in message.tool_calls:
            result = run_tool_call(call)
            print(f"  step {step}: {call.function.name}({call.function.arguments}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,  # links this result to the call it answers
                "name": call.function.name,
                "content": result,
            })

    return "Stopped: too many tool steps."  # always cap the loop in production


if __name__ == "__main__":
    show_decisions()

    banner("Part 2: full tool-calling loop")
    for question in [
        "Hi! Can you check order A1001 and A1002 and tell me which arrives first?",
        "What's the status of order Z9999?",
        "It's hot here in Kochi, right? Tell me in fahrenheit.",
    ]:
        print(f"\nUser: {question}")
        show("Final answer", agent_loop(question))

# Exercises:
# 1. Change get_order_status's description to just "Order tool." Re-run Part 1.
#    How do the decisions change? (Descriptions are prompts.)
# 2. Add a `cancel_order(order_id)` tool. Require the loop to ask the user for
#    confirmation (input()) before executing it - a human-in-the-loop guard.
# 3. Set tool_choice="required" for "What is the capital of France?". What happens?
