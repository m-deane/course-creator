"""
Recipe: Complete ART Training Loop
====================================

Minimal training loop for an ART agent with MCP tools.
Shows the full cycle: rollout → score → train → checkpoint.

Usage:
    Modify the MCP_SERVER_URL and MODEL_NAME, then run:
    python 04_training_loop.py
"""

import numpy as np


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-3B"
MCP_SERVER_URL = "http://localhost:8000"
GROUP_SIZE = 4        # completions per prompt
NUM_STEPS = 50        # training steps
LEARNING_RATE = 2e-5


# ============================================================
# Step 1: Define the rollout function
# ============================================================

def rollout(model, scenario: str, tools: list[dict]) -> dict:
    """
    Run one agent episode.

    The agent receives a question, uses tools to query the database,
    and produces a final answer. Every message is recorded.

    Parameters
    ----------
    model : object
        The ART model client (sends requests to vLLM backend)
    scenario : str
        The natural language question for the agent
    tools : list[dict]
        Available MCP tools (discovered automatically)

    Returns
    -------
    dict with "messages" (trajectory) and "final_answer"
    """
    messages = [
        {"role": "system", "content": "You are a SQL expert. Use the available tools to answer questions about the database. Always explore the schema before writing queries."},
        {"role": "user", "content": scenario},
    ]

    max_turns = 8
    for _ in range(max_turns):
        response = model.chat(messages=messages, tools=tools)

        if response.tool_calls:
            # Agent wants to use a tool
            messages.append(response.to_dict())
            for tool_call in response.tool_calls:
                # Execute tool call against MCP server
                result = execute_tool(tool_call, MCP_SERVER_URL)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # Agent produced final answer
            messages.append(response.to_dict())
            break

    return {
        "messages": messages,
        "final_answer": messages[-1].get("content", ""),
    }


def execute_tool(tool_call, server_url: str) -> str:
    """Execute a tool call against the MCP server."""
    import json
    import urllib.request

    payload = json.dumps({
        "tool": tool_call.function.name,
        "arguments": json.loads(tool_call.function.arguments),
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/call",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["result"]
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# Step 2: Score trajectories with RULER
# ============================================================

def score_trajectories(scenario: str, trajectories: list[dict]) -> list[float]:
    """
    Score a group of trajectories using RULER (LLM-as-a-judge).

    In production, this calls the judge model. Here we show the interface.
    """
    import json
    from openai import OpenAI

    client = OpenAI()

    traj_texts = []
    for i, traj in enumerate(trajectories):
        text = f"Trajectory {i+1}:\n"
        for msg in traj["messages"]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            text += f"  [{role}] {content}\n"
        traj_texts.append(text)

    prompt = (
        f"Task: {scenario}\n\n"
        + "\n---\n".join(traj_texts)
        + f"\n\nScore each trajectory 0.0-1.0. Return JSON array of {len(trajectories)} scores."
    )

    response = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "Score agent trajectories by correctness and efficiency. Return ONLY a JSON array."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    scores = json.loads(response.choices[0].message.content.strip())
    return [max(0.0, min(1.0, float(s))) for s in scores]


# ============================================================
# Step 3: The training loop
# ============================================================

def train(model, scenarios: list[str], tools: list[dict]):
    """
    Full GRPO training loop.

    For each step:
    1. Pick a scenario
    2. Run GROUP_SIZE rollouts
    3. Score with RULER
    4. Train with GRPO (model.train handles the weight update)
    5. New checkpoint loads automatically
    """
    for step in range(NUM_STEPS):
        # Pick a random scenario
        scenario = scenarios[step % len(scenarios)]

        # Generate multiple trajectories
        trajectories = [rollout(model, scenario, tools) for _ in range(GROUP_SIZE)]

        # Score with RULER
        rewards = score_trajectories(scenario, trajectories)

        # Compute advantages
        r = np.array(rewards)
        advantages = ((r - r.mean()) / (r.std() + 1e-8)).tolist()

        # Update model weights via GRPO
        # ART handles: loss computation, gradient update, LoRA checkpoint swap
        model.train(
            trajectories=[t["messages"] for t in trajectories],
            advantages=advantages,
        )

        # Log progress
        print(
            f"Step {step+1:3d}/{NUM_STEPS} | "
            f"Scenario: {scenario[:40]}... | "
            f"Rewards: {[f'{r:.2f}' for r in rewards]} | "
            f"Best: {max(rewards):.2f}"
        )


if __name__ == "__main__":
    print("ART Training Loop Recipe")
    print("=" * 50)
    print()
    print("To run this recipe:")
    print("1. Start the MCP server:  python mcp_database_server.py --create-sample")
    print("2. Start ART backend:     art serve --model Qwen/Qwen2.5-3B")
    print("3. Run this script:       python 04_training_loop.py")
    print()
    print("See Module 05 and 06 for complete walkthroughs.")
