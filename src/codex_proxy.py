"""
Local proxy: OpenAI Chat Completion API → Codex Responses API.
Lets litellm hit localhost with standard format, proxy translates to chatgpt.com.
"""
import json
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

CODEX_AUTH_PATH = os.path.expanduser("~/.codex/auth.json")
CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_VERSION = "0.130.0"

app = FastAPI()


def read_codex_credentials():
    with open(CODEX_AUTH_PATH) as f:
        data = json.load(f)
    token = data.get("tokens", {}).get("access_token")
    account_id = data.get("tokens", {}).get("account_id")
    if not token:
        raise RuntimeError("No Codex OAuth token — run `codex login`")
    return token, account_id


_call_id_map = {}

def _to_fc_id(call_id):
    """Convert litellm call_xxx ID to Codex fc_xxx format."""
    if call_id.startswith("fc_"):
        return call_id
    if call_id in _call_id_map:
        return _call_id_map[call_id]
    fc_id = "fc_" + call_id.replace("call_", "")
    _call_id_map[call_id] = fc_id
    return fc_id


def chat_messages_to_responses_input(messages, tools=None):
    """Convert Chat Completion messages to Responses API input format."""
    input_items = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue
        elif role == "user":
            input_items.append({"role": "user", "content": content})
        elif role == "assistant":
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    fc_id = _to_fc_id(tc.get("id", ""))
                    input_items.append({
                        "type": "function_call",
                        "id": fc_id,
                        "call_id": fc_id,
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })
            if content:
                input_items.append({"role": "assistant", "content": content})
        elif role == "tool":
            fc_id = _to_fc_id(msg.get("tool_call_id", ""))
            input_items.append({
                "type": "function_call_output",
                "call_id": fc_id,
                "output": content,
            })
    return input_items


def responses_output_to_chat_completion(response_data, model):
    """Convert Responses API output to Chat Completion format."""
    output = response_data.get("output", [])

    content = None
    tool_calls = []

    for item in output:
        item_type = item.get("type", "")
        if item_type == "message":
            # Extract text content
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content = (content or "") + part.get("text", "")
        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id", item.get("id", "")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = response_data.get("usage", {})

    return {
        "id": response_data.get("id", "proxy-response"),
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def convert_tools_to_responses_format(tools):
    """Convert OpenAI Chat Completion tools to Responses API tools."""
    if not tools:
        return []
    responses_tools = []
    for t in tools:
        if t.get("type") == "function":
            func = t["function"]
            responses_tools.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "strict": False,
            })
    return responses_tools


def _patch_additional_properties(schema):
    """Recursively add additionalProperties: false to all object schemas."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    for key in ("properties", "items", "anyOf", "oneOf", "allOf"):
        val = schema.get(key)
        if isinstance(val, dict):
            for v in val.values():
                _patch_additional_properties(v)
        elif isinstance(val, list):
            for v in val:
                _patch_additional_properties(v)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    token, account_id = read_codex_credentials()

    messages = body.get("messages", [])
    tools = body.get("tools")
    model = body.get("model", "gpt-5.5")

    # Extract system message as instructions
    instructions = None
    for msg in messages:
        if msg.get("role") == "system":
            instructions = msg.get("content", "")
            break

    # Build Responses API request
    input_items = chat_messages_to_responses_input(messages, tools)
    responses_tools = convert_tools_to_responses_format(tools)

    responses_body = {
        "model": model,
        "input": input_items,
        "store": False,
        "stream": True,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"effort": body.get("reasoning_effort", "high"), "summary": "auto"},
    }
    responses_body["instructions"] = instructions or "You are a helpful assistant."
    if responses_tools:
        responses_body["tools"] = responses_tools

    # Pass through response_format for structured output
    response_format = body.get("response_format")
    if response_format and isinstance(response_format, dict):
        rf_type = response_format.get("type", "")
        if rf_type == "json_schema" and "json_schema" in response_format:
            js = response_format["json_schema"]
            schema = js.get("schema", {})
            # Codex requires additionalProperties: false at every object level
            _patch_additional_properties(schema)
            fmt = {
                "type": "json_schema",
                "name": js.get("name", "structured_output"),
                "strict": True,
                "schema": schema,
            }
            responses_body["text"] = {"format": fmt}
        elif rf_type == "json_object":
            responses_body["text"] = {"format": {"type": "json_object"}}
        print(f"[proxy] response_format applied: {rf_type}", flush=True)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "chatgpt-account-id": account_id or "",
        "OpenAI-Beta": "responses=experimental",
        "originator": "codex_sdk_ts",
        "User-Agent": f"codex_sdk_ts/{CODEX_VERSION}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", CODEX_RESPONSES_URL, json=responses_body, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return JSONResponse(
                    status_code=resp.status_code,
                    content={"error": {"message": body.decode()[:500], "type": "proxy_error"}},
                )

            # Collect output items from streaming events
            output_items = []
            usage = {}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    evt = json.loads(payload)
                    if evt.get("type") == "response.output_item.done":
                        output_items.append(evt.get("item", {}))
                    elif evt.get("type") == "response.completed":
                        usage = evt.get("response", {}).get("usage", {})
                except json.JSONDecodeError:
                    continue

    response_data = {"output": output_items, "usage": usage}
    chat_response = responses_output_to_chat_completion(response_data, model)
    # Log response content for debugging structured output issues
    msg = chat_response.get("choices", [{}])[0].get("message", {})
    print(f"[proxy] response content: {json.dumps(msg.get('content', ''))[:300]}", flush=True)
    return JSONResponse(content=chat_response)


@app.get("/v1/models")
async def list_models():
    return JSONResponse(content={
        "data": [{"id": "gpt-5.5", "object": "model"}],
        "object": "list",
    })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    print(f"Codex proxy starting on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
