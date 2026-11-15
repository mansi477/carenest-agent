"""Message helpers. The key idea: trim history only at user-turn boundaries."""
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


def message_text(msg: AnyMessage) -> str:
    """Plain text of a message. Anthropic returns content as a list of blocks; OpenAI as a string."""
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def last_turns(messages: list[AnyMessage], n: int) -> list[AnyMessage]:
    """Keep the last n user turns, including everything the assistant and tools did in them.

    Slicing by raw message count can cut between an AI tool call and its ToolMessage,
    which providers reject. Starting the slice at a HumanMessage never does.
    """
    starts = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(starts) <= n:
        return list(messages)
    return list(messages[starts[-n]:])


def plain_history(messages: list[AnyMessage], n: int) -> list[AnyMessage]:
    """Last n turns as plain user/assistant text only (no tool calls). Safe for tool-free nodes."""
    out: list[AnyMessage] = []
    for m in last_turns(messages, n):
        text = message_text(m).strip()
        if isinstance(m, HumanMessage):
            out.append(HumanMessage(content=text))
        elif isinstance(m, AIMessage) and text and not m.tool_calls:
            out.append(AIMessage(content=text))
    return out


def render_transcript(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            continue
        text = message_text(m).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def latest_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return message_text(m)
    return ""
