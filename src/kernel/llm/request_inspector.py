"""LLM 请求体调试检视器。

提供一个基于 FastAPI + SSE 的 Web 界面，实时记录并展示每次
OpenAI 兼容 API 的完整请求体，方便调试 payload 结构。

除原始 JSON 外，本模块还会将请求体转换为结构化渲染模型，
用于以对话式布局展示 role、工具调用、工具结果与 Markdown 文本。

使用方式：
    # 在 HTTP 服务器启动后调用一次
    from src.kernel.llm.request_inspector import get_inspector
    get_inspector().mount(fastapi_app)

    # 在发起请求前调用
    from src.kernel.llm.request_inspector import capture
    capture("chat.completions.create", params)

WebUI 地址：http://<host>:<port>/_inspector/
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse


_OVERVIEW_FIELDS: list[tuple[str, str]] = [
    ("stream", "流式"),
    ("temperature", "temperature"),
    ("max_tokens", "max_tokens"),
    ("top_p", "top_p"),
    ("presence_penalty", "presence_penalty"),
    ("frequency_penalty", "frequency_penalty"),
    ("seed", "seed"),
    ("tool_choice", "tool_choice"),
    ("parallel_tool_calls", "parallel_tool_calls"),
    ("response_format", "response_format"),
]

_ROLE_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "tool_result": "Tool Result",
}


def _json_dumps(value: Any) -> str:
    """将值安全序列化为 JSON 字符串。"""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _format_scalar(value: Any) -> str:
    """将任意标量格式化为便于展示的字符串。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return _json_dumps(value)
    return str(value)


def _build_overview(
    api_name: str,
    model: str,
    params: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    """构造顶层请求摘要。"""
    messages = _get_inspector_messages(params)
    tools = params.get("tools", [])
    overview = [
        {"label": "API", "value": api_name},
        {"label": "提供商", "value": _format_scalar(metadata.get("api_provider", "-"))},
        {"label": "模型", "value": model or "-"},
        {
            "label": "消息数",
            "value": str(len(messages) if isinstance(messages, list) else 0),
        },
        {"label": "工具数", "value": str(len(tools) if isinstance(tools, list) else 0)},
    ]

    estimated_input_tokens = metadata.get("estimated_input_tokens")
    if estimated_input_tokens is not None:
        overview.append(
            {
                "label": "预估输入 Tokens",
                "value": _format_scalar(estimated_input_tokens),
            }
        )

    request_name = metadata.get("request_name")
    if request_name:
        overview.append({"label": "请求名称", "value": _format_scalar(request_name)})

    for key, label in _OVERVIEW_FIELDS:
        if key in params:
            overview.append({"label": label, "value": _format_scalar(params[key])})

    reserved_keys = {"messages", "tools", "model", "system"}
    reserved_keys.update(name for name, _ in _OVERVIEW_FIELDS)
    for key in sorted(key for key in params.keys() if key not in reserved_keys):
        overview.append({"label": key, "value": _format_scalar(params[key])})

    return overview


def _schema_type_text(schema: dict[str, Any]) -> str:
    """读取 JSON Schema 的类型信息。"""
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        return " | ".join(str(item) for item in schema_type)
    if "anyOf" in schema:
        return "anyOf"
    if "oneOf" in schema:
        return "oneOf"
    if "allOf" in schema:
        return "allOf"
    return "unknown"


def _build_tools_view(params: dict[str, Any]) -> list[dict[str, Any]]:
    """将 tools schema 转为更适合前端展示的结构。"""
    tools = params.get("tools", [])
    if not isinstance(tools, list):
        return []

    rendered_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            rendered_tools.append(
                {
                    "index": index,
                    "name": f"tool_{index}",
                    "kind": "unknown",
                    "description": "",
                    "required": [],
                    "properties": [],
                    "raw_json": _json_dumps(tool),
                }
            )
            continue

        function_obj = tool.get("function")
        if isinstance(function_obj, dict):
            function_schema: dict[str, Any] = function_obj
        else:
            function_schema = tool
        parameters_obj = function_schema.get("parameters")
        if not isinstance(parameters_obj, dict):
            parameters_obj = function_schema.get("input_schema")
        parameters: dict[str, Any] = (
            parameters_obj if isinstance(parameters_obj, dict) else {}
        )
        properties_obj = parameters.get("properties")
        properties: dict[str, Any] = (
            properties_obj if isinstance(properties_obj, dict) else {}
        )
        required_obj = parameters.get("required")
        required: list[Any] = required_obj if isinstance(required_obj, list) else []
        rendered_tools.append(
            {
                "index": index,
                "name": str(function_schema.get("name", f"tool_{index}")),
                "kind": str(tool.get("type", "function")),
                "description": str(function_schema.get("description", "")),
                "required": [str(item) for item in required],
                "properties": [
                    {
                        "name": str(prop_name),
                        "type": (
                            _schema_type_text(prop_schema)
                            if isinstance(prop_schema, dict)
                            else "unknown"
                        ),
                        "description": (
                            str(prop_schema.get("description", ""))
                            if isinstance(prop_schema, dict)
                            else ""
                        ),
                        "required": prop_name in required,
                    }
                    for prop_name, prop_schema in properties.items()
                ],
                "raw_json": _json_dumps(tool),
            }
        )
    return rendered_tools


def _make_block(block_type: str, **kwargs: Any) -> dict[str, Any]:
    """构造统一的消息块结构。"""
    block = {"type": block_type}
    block.update(kwargs)
    return block


def _render_unknown_content(content: Any, label: str = "未知内容") -> dict[str, Any]:
    """为未知结构构造回退块。"""
    return _make_block("unknown", label=label, text=_json_dumps(content))


def _render_content_list(content: list[Any]) -> list[dict[str, Any]]:
    """将 OpenAI content 数组转换为消息块。"""
    blocks: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(_make_block("markdown", text=item))
            continue

        if not isinstance(item, dict):
            blocks.append(_render_unknown_content(item))
            continue

        item_type = str(item.get("type", "unknown"))
        if item_type in {"text", "input_text", "output_text"}:
            blocks.append(_make_block("markdown", text=str(item.get("text", ""))))
            continue

        if item_type in {"image_url", "input_image"}:
            image_url_obj = item.get("image_url")
            image_url: dict[str, Any] = (
                image_url_obj if isinstance(image_url_obj, dict) else {}
            )
            url = image_url.get("url") or item.get("url") or ""
            blocks.append(
                _make_block(
                    "media",
                    media_type="image",
                    title="图片内容",
                    text="请求中包含图片输入。",
                    meta=_format_scalar(url)[:120],
                )
            )
            continue

        if item_type == "image":
            source_obj = item.get("source")
            source = source_obj if isinstance(source_obj, dict) else {}
            meta = source.get("media_type") or source.get("type") or "image"
            blocks.append(
                _make_block(
                    "media",
                    media_type="image",
                    title="图片内容",
                    text="请求中包含图片输入。",
                    meta=_format_scalar(meta),
                )
            )
            continue

        if item_type in {"audio", "input_audio", "output_audio"}:
            blocks.append(
                _make_block(
                    "media",
                    media_type="audio",
                    title="音频内容",
                    text="请求中包含音频输入。",
                    meta="音频数据已省略",
                )
            )
            continue

        if item_type == "refusal":
            blocks.append(
                _make_block(
                    "markdown", text=str(item.get("refusal", "")), label="拒绝说明"
                )
            )
            continue

        if item_type == "thinking":
            blocks.append(
                _make_block(
                    "markdown",
                    text=str(item.get("thinking", "")),
                    label="Reasoning",
                )
            )
            continue

        if item_type == "tool_use":
            blocks.append(
                _make_block(
                    "tool_call",
                    call_id=str(item.get("id", "") or ""),
                    name=str(item.get("name", "unknown_tool")),
                    arguments_text=_json_dumps(item.get("input", {})),
                )
            )
            continue

        if item_type == "tool_result":
            blocks.append(
                _make_block(
                    "tool_result",
                    name=str(item.get("tool_name", "")),
                    call_id=str(item.get("tool_use_id", "") or ""),
                    text=_format_scalar(item.get("content", "")),
                )
            )
            continue

        blocks.append(
            _render_unknown_content(item, label=f"未知 content 类型: {item_type}")
        )
    return blocks


def _get_inspector_messages(params: dict[str, Any]) -> list[Any]:
    """获取用于检视器展示的消息列表，兼容 Anthropic 的 system 字段。"""
    rendered_messages: list[Any] = []

    system = params.get("system")
    if isinstance(system, str):
        rendered_messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        rendered_messages.append({"role": "system", "content": system})

    messages = params.get("messages", [])
    if isinstance(messages, list):
        rendered_messages.extend(messages)

    return rendered_messages


def _build_message_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """将单条 message 转换为前端可渲染的块列表。"""
    blocks: list[dict[str, Any]] = []
    role = str(message.get("role", "unknown"))
    content = message.get("content")

    if role == "tool":
        tool_name = str(message.get("name", ""))
        blocks.append(
            _make_block(
                "tool_result",
                name=tool_name,
                call_id=str(message.get("tool_call_id", "") or ""),
                text=_format_scalar(content if content is not None else ""),
            )
        )
    elif isinstance(content, str):
        blocks.append(_make_block("markdown", text=content))
    elif isinstance(content, list):
        blocks.extend(_render_content_list(content))
    elif content is None:
        blocks.append(_make_block("empty", text="此消息没有文本内容。"))
    else:
        blocks.append(_render_unknown_content(content))

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        blocks.append(
            _make_block("markdown", text=reasoning_content, label="Reasoning")
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                blocks.append(_render_unknown_content(tool_call, label="未知工具调用"))
                continue
            function_block_obj = tool_call.get("function")
            function_block: dict[str, Any] = (
                function_block_obj if isinstance(function_block_obj, dict) else {}
            )
            name = str(
                function_block.get("name", tool_call.get("name", "unknown_tool"))
            )
            arguments = function_block.get("arguments", tool_call.get("args", {}))
            arguments_text = (
                arguments if isinstance(arguments, str) else _json_dumps(arguments)
            )
            blocks.append(
                _make_block(
                    "tool_call",
                    call_id=str(tool_call.get("id", "") or ""),
                    name=name,
                    arguments_text=arguments_text,
                )
            )

    if not blocks:
        blocks.append(_make_block("empty", text="此消息没有可展示内容。"))

    return blocks


def _build_messages_view(params: dict[str, Any]) -> list[dict[str, Any]]:
    """将请求中的 messages 转换为卡片化展示模型。"""
    messages = _get_inspector_messages(params)
    if not messages:
        return []

    rendered_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            rendered_messages.append(
                {
                    "index": index,
                    "role": "unknown",
                    "label": "Unknown",
                    "meta": "非标准 message 结构",
                    "blocks": [_render_unknown_content(message)],
                }
            )
            continue

        role = str(message.get("role", "unknown"))
        meta_parts: list[str] = []
        if isinstance(message.get("name"), str) and message["name"]:
            meta_parts.append(f"name: {message['name']}")
        if isinstance(message.get("tool_call_id"), str) and message["tool_call_id"]:
            meta_parts.append(f"tool_call_id: {message['tool_call_id']}")
        if isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            meta_parts.append(f"tools: {len(message['tool_calls'])}")

        rendered_messages.append(
            {
                "index": index,
                "role": role,
                "label": _ROLE_LABELS.get(role, role.title()),
                "meta": " · ".join(meta_parts),
                "blocks": _build_message_blocks(message),
            }
        )

    return rendered_messages


def build_render_view(
    api_name: str,
    model: str,
    params: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造详情页使用的结构化渲染模型。"""
    effective_metadata = metadata or {}
    return {
        "overview": _build_overview(api_name, model, params, effective_metadata),
        "tools": _build_tools_view(params),
        "messages": _build_messages_view(params),
    }


@dataclass
class CapturedRequest:
    """单条捕获记录。"""

    id: int
    ts: float
    api_name: str
    model: str
    params: dict[str, Any]
    metadata: dict[str, Any]

    def to_summary(self) -> dict[str, Any]:
        """返回列表展示用的摘要。"""
        msg_count = len(self.params.get("messages", []))
        tool_count = len(self.params.get("tools", []))
        return {
            "id": self.id,
            "ts": self.ts,
            "ts_str": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "api_name": self.api_name,
            "model": self.model,
            "api_provider": str(self.metadata.get("api_provider", "-")),
            "estimated_input_tokens": self.metadata.get("estimated_input_tokens"),
            "msg_count": msg_count,
            "tool_count": tool_count,
        }

    def to_full(self) -> dict[str, Any]:
        """返回完整记录与结构化渲染模型。"""
        summary = self.to_summary()
        summary["params"] = self.params
        summary["metadata"] = self.metadata
        summary["rendered"] = build_render_view(
            self.api_name, self.model, self.params, self.metadata
        )
        return summary


class RequestInspector:
    """LLM 请求体检视器，保存最近 N 条请求并通过 Web 界面展示。"""

    def __init__(self, max_records: int = 200) -> None:
        self._max_records = max_records
        self._records: deque[CapturedRequest] = deque(maxlen=max_records)
        self._counter: int = 0
        self._subscribers: list[asyncio.Queue[CapturedRequest | None]] = []
        self._mounted: bool = False

    def capture(
        self,
        api_name: str,
        params: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """捕获一条请求体，存入内存并推送给所有 SSE 订阅者。"""
        self._counter += 1
        try:
            copied = json.loads(json.dumps(params, default=str))
        except Exception:
            copied = {"_raw": str(params)}
        effective_metadata = metadata.copy() if isinstance(metadata, dict) else {}

        model = str(copied.get("model", "—"))
        record = CapturedRequest(
            id=self._counter,
            ts=time.time(),
            api_name=api_name,
            model=model,
            params=copied,
            metadata=effective_metadata,
        )
        self._records.append(record)

        for queue in list(self._subscribers):
            try:
                queue.put_nowait(record)
            except asyncio.QueueFull:
                pass

    def import_json(self, payload: Any) -> list[CapturedRequest]:
        """导入外部 JSON，并转换为可渲染的捕获记录。"""
        imported: list[CapturedRequest] = []
        for api_name, params, metadata in _normalize_import_payload(payload):
            next_id = self._counter + 1
            self.capture(api_name, params, metadata)
            record = self._records[-1]
            if record.id != next_id:
                raise RuntimeError("导入记录 ID 不连续")
            imported.append(record)
        return imported

    def mount(self, app: Any, prefix: str = "/_inspector") -> None:
        """将 WebUI 路由挂载到 FastAPI 应用。"""
        if self._mounted:
            return
        self._mounted = True
        router = self._build_router()
        app.include_router(router, prefix=prefix)

    def _build_router(self) -> APIRouter:
        """构建 FastAPI router，包含 WebUI、REST 与 SSE 端点。"""
        router = APIRouter()

        @router.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def webui() -> HTMLResponse:  # type: ignore[return-value]
            return HTMLResponse(_WEBUI_HTML)

        @router.get("/api/requests")
        async def list_requests() -> JSONResponse:
            return JSONResponse([record.to_summary() for record in self._records])

        @router.get("/api/requests/{req_id}")
        async def get_request(req_id: int) -> JSONResponse:
            for record in reversed(self._records):
                if record.id == req_id:
                    return JSONResponse(record.to_full())
            return JSONResponse({"error": "not found"}, status_code=404)

        @router.delete("/api/requests")
        async def clear_requests() -> JSONResponse:
            self._records.clear()
            return JSONResponse({"ok": True})

        @router.post("/api/import")
        async def import_requests(payload: Any = Body(...)) -> JSONResponse:
            try:
                imported = self.import_json(payload)
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return JSONResponse(
                {
                    "ok": True,
                    "count": len(imported),
                    "items": [record.to_summary() for record in imported],
                }
            )

        @router.get("/api/analytics")
        async def get_analytics() -> JSONResponse:
            """返回综合统计指标。"""
            try:
                from src.kernel.llm.stats import get_llm_stats_collector
                collector = get_llm_stats_collector()
                summary = await collector.get_summary()
                by_model = await collector.get_by_model()
                by_request = await collector.get_by_request_name()
                by_stream = await collector.get_by_stream()
                return JSONResponse({
                    "summary": summary,
                    "by_model": by_model,
                    "by_request_name": by_request,
                    "by_stream": by_stream,
                })
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        @router.get("/api/stream", include_in_schema=False)
        async def sse_stream() -> StreamingResponse:
            queue: asyncio.Queue[CapturedRequest | None] = asyncio.Queue(maxsize=50)
            self._subscribers.append(queue)

            async def generate() -> AsyncIterator[str]:
                try:
                    snapshot = [record.to_summary() for record in self._records]
                    yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
                    while True:
                        try:
                            record = await asyncio.wait_for(queue.get(), timeout=25)
                        except asyncio.TimeoutError:
                            yield ": heartbeat\n\n"
                            continue
                        if record is None:
                            break
                        yield f"event: new\ndata: {json.dumps(record.to_summary())}\n\n"
                finally:
                    try:
                        self._subscribers.remove(queue)
                    except ValueError:
                        pass

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        @router.get("/analytics", response_class=HTMLResponse, include_in_schema=False)
        async def analytics_page() -> HTMLResponse:
            return HTMLResponse(_ANALYTICS_HTML)

        return router


_inspector: RequestInspector | None = None


def get_inspector() -> RequestInspector:
    """获取全局 RequestInspector 单例。"""
    global _inspector
    if _inspector is None:
        _inspector = RequestInspector()
    return _inspector


def capture(
    api_name: str,
    params: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """捕获一条 OpenAI 请求体。"""
    get_inspector().capture(api_name, params, metadata)


def _normalize_import_payload(
    payload: Any,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """将导入 JSON 归一化为 capture 所需的记录列表。"""
    entries = _normalize_import_entries(payload)
    if not entries:
        raise ValueError("导入 JSON 为空，无法渲染")
    return entries


def _normalize_import_entries(
    payload: Any,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """递归解析导入 JSON，支持单条、列表和包装对象。"""
    if isinstance(payload, list):
        entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for item in payload:
            entries.extend(_normalize_import_entries(item))
        return entries

    if not isinstance(payload, dict):
        raise ValueError(
            "导入 JSON 必须是对象、对象数组，或包含 requests/params 的对象"
        )

    requests_payload = payload.get("requests")
    if requests_payload is not None:
        if not isinstance(requests_payload, list):
            raise ValueError("requests 字段必须是数组")
        entries: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for item in requests_payload:
            entries.extend(_normalize_import_entries(item))
        return entries

    metadata_raw = payload.get("metadata")
    metadata = metadata_raw.copy() if isinstance(metadata_raw, dict) else {}
    metadata.setdefault("imported", True)

    api_name = str(payload.get("api_name") or payload.get("api") or "imported.request")

    if "params" in payload:
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("params 字段必须是对象")
        if "source" in payload and "import_source" not in metadata:
            metadata["import_source"] = str(payload.get("source"))
        return [(api_name, params, metadata)]

    if any(key in payload for key in ("messages", "tools", "model", "input")):
        params = payload.copy()
        params.pop("metadata", None)
        return [(api_name, params, metadata)]

    raise ValueError(
        "无法识别导入 JSON 结构；请提供原始请求体、包含 params 的对象，或 requests 数组"
    )


_WEBUI_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM 请求检视器</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f4efe7;
    --bg-strong: #ebe2d5;
    --panel: rgba(255, 252, 246, 0.9);
    --panel-strong: #fffaf3;
    --border: rgba(70, 53, 32, 0.12);
    --shadow: 0 18px 48px rgba(78, 60, 37, 0.08);
    --text: #2f261d;
    --muted: #7a6856;
    --accent: #0e7490;
    --accent-soft: rgba(14, 116, 144, 0.12);
    --success: #1f7a4d;
    --danger: #aa3a2a;
    --system: #5e3ebc;
    --user: #0f766e;
    --assistant: #b45309;
    --tool: #2563eb;
    --code-bg: #221c16;
    --code-text: #f8efe2;
  }
  html, body { height: 100%; }
  body {
    height: 100vh;
    min-height: 100vh;
    background:
      radial-gradient(circle at top left, rgba(14, 116, 144, 0.16), transparent 28%),
      radial-gradient(circle at top right, rgba(180, 83, 9, 0.14), transparent 32%),
      linear-gradient(180deg, var(--bg) 0%, #f8f3eb 100%);
    color: var(--text);
    font-family: 'Segoe UI', 'PingFang SC', 'Noto Sans SC', sans-serif;
    font-size: 14px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    padding: 18px 22px;
    display: flex;
    gap: 16px;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 250, 243, 0.78);
    backdrop-filter: blur(18px);
  }
  .brand { display: flex; align-items: center; gap: 14px; min-width: 0; flex: 1; }
  .brand-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(14,116,144,0.95), rgba(180,83,9,0.88));
    color: #fff;
    display: grid;
    place-items: center;
    font-weight: 700;
    letter-spacing: 0.04em;
    box-shadow: 0 10px 24px rgba(14, 116, 144, 0.18);
  }
  .brand-copy h1 { font-size: 19px; font-weight: 700; }
  .brand-copy p { color: var(--muted); margin-top: 4px; font-size: 12px; }
  .status-group { display: flex; align-items: center; gap: 10px; }
  #status-dot {
    width: 10px; height: 10px; border-radius: 50%; background: #baa995;
    box-shadow: 0 0 0 6px rgba(122, 104, 86, 0.08);
  }
  #status-dot.live { background: var(--success); box-shadow: 0 0 0 6px rgba(31, 122, 77, 0.14); }
  .badge {
    min-width: 38px;
    text-align: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 700;
  }
  .toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
  .toolbar-group { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button {
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.72);
    color: var(--text);
    padding: 9px 14px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 13px;
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
  }
  button:hover {
    transform: translateY(-1px);
    border-color: rgba(14, 116, 144, 0.28);
    box-shadow: 0 10px 20px rgba(78, 60, 37, 0.08);
  }
  button.active { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.danger { color: var(--danger); }
  input[type=text] {
    width: 220px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.72);
    color: var(--text);
    border-radius: 12px;
    padding: 10px 14px;
    outline: none;
  }
  input[type=text]:focus { border-color: rgba(14, 116, 144, 0.45); }
  textarea {
    width: 100%;
    min-height: 220px;
    resize: vertical;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.8);
    color: var(--text);
    border-radius: 16px;
    padding: 14px;
    outline: none;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.6;
  }
  textarea:focus { border-color: rgba(14, 116, 144, 0.45); }
  .main { display: flex; flex: 1; min-height: 0; overflow: hidden; }
  .import-panel {
    display: none;
    padding: 16px 18px;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 250, 243, 0.82);
    backdrop-filter: blur(12px);
    flex-direction: column;
    gap: 12px;
  }
  .import-panel.open { display: flex; }
  .import-row {
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
  }
  .import-hint { color: var(--muted); font-size: 12px; }
  .import-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .status-text { color: var(--muted); font-size: 12px; min-height: 18px; }
  .status-text.error { color: var(--danger); }
  .status-text.success { color: var(--success); }
  .list-panel {
    width: 360px;
    flex-shrink: 0;
    border-right: 1px solid var(--border);
    background: rgba(248, 242, 233, 0.68);
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .list-scroll { padding: 14px; overflow-y: auto; min-height: 0; }
  .req-item {
    padding: 14px;
    border: 1px solid transparent;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 10px 28px rgba(78, 60, 37, 0.04);
    cursor: pointer;
    margin-bottom: 12px;
    transition: transform .15s ease, border-color .15s ease, background .15s ease;
  }
  .req-item:hover { transform: translateY(-1px); border-color: rgba(14, 116, 144, 0.18); }
  .req-item.active {
    border-color: rgba(14, 116, 144, 0.35);
    background: linear-gradient(180deg, rgba(14, 116, 144, 0.08), rgba(255, 255, 255, 0.8));
  }
  .row1 { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
  .api-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(14, 116, 144, 0.1);
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
  }
  .ts { color: var(--muted); font-size: 12px; }
  .model { margin-top: 9px; font-weight: 700; word-break: break-word; }
  .meta { margin-top: 6px; color: var(--muted); font-size: 12px; }
  .meta-row { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
  .mini-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 999px;
    background: rgba(94, 62, 188, 0.08);
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
  }
  .detail-panel { flex: 1; min-width: 0; display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .detail-toolbar {
    padding: 14px 18px;
    display: flex;
    gap: 10px;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 250, 243, 0.72);
    backdrop-filter: blur(12px);
    flex-wrap: wrap;
  }
  .detail-title { flex: 1; min-width: 240px; color: var(--muted); }
  .detail-body { flex: 1; min-height: 0; overflow: auto; padding: 22px; }
  .empty-tip {
    padding: 36px;
    border: 1px dashed rgba(122, 104, 86, 0.24);
    border-radius: 22px;
    text-align: center;
    color: var(--muted);
    background: rgba(255, 255, 255, 0.5);
  }
  .section { margin-bottom: 18px; }
  .section-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }
  .section-head h2 { font-size: 15px; }
  .section-hint { color: var(--muted); font-size: 12px; }
  .overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
  }
  .overview-item {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: var(--panel);
    padding: 14px;
    box-shadow: var(--shadow);
  }
  .overview-item .label { color: var(--muted); font-size: 12px; }
  .overview-item .value {
    margin-top: 8px;
    font-weight: 700;
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'Segoe UI', 'PingFang SC', sans-serif;
  }
  details.tool-card {
    border: 1px solid var(--border);
    border-radius: 18px;
    background: var(--panel);
    margin-bottom: 12px;
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  details.tool-card > summary {
    list-style: none;
    cursor: pointer;
    padding: 14px 16px;
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
  }
  details.tool-card > summary::-webkit-details-marker { display: none; }
  .tool-name { font-weight: 700; }
  .tool-kind { color: var(--muted); font-size: 12px; }
  .tool-body { padding: 0 16px 16px; }
  .tool-desc { color: var(--muted); margin-bottom: 12px; }
  .tool-props {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
    margin-bottom: 12px;
  }
  .tool-prop {
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.62);
  }
  .tool-prop .name { font-weight: 700; }
  .tool-prop .meta { margin-top: 6px; }
  .message-stack { display: flex; flex-direction: column; gap: 14px; }
  .message-card {
    border-radius: 22px;
    border: 1px solid var(--border);
    background: var(--panel);
    box-shadow: var(--shadow);
    overflow: hidden;
  }
  .message-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.5);
    flex-wrap: wrap;
  }
  .message-role {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
  }
  .message-role .dot { width: 10px; height: 10px; border-radius: 50%; }
  .role-system .dot { background: var(--system); }
  .role-user .dot { background: var(--user); }
  .role-assistant .dot { background: var(--assistant); }
  .role-tool .dot { background: var(--tool); }
  .role-unknown .dot { background: var(--muted); }
  .message-meta { color: var(--muted); font-size: 12px; }
  .message-body { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .message-card.cache-hit-candidate {
    background: linear-gradient(180deg, rgba(221, 247, 227, 0.92), rgba(248, 255, 249, 0.94));
    border-color: rgba(58, 138, 83, 0.22);
  }
  .message-card.cache-miss-candidate {
    background: linear-gradient(180deg, rgba(254, 236, 232, 0.92), rgba(255, 248, 246, 0.95));
    border-color: rgba(181, 78, 58, 0.18);
  }
  .cache-probe-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.01em;
  }
  .cache-probe-chip.hit {
    color: #215b34;
    background: rgba(132, 204, 151, 0.2);
    border: 1px solid rgba(58, 138, 83, 0.2);
  }
  .cache-probe-chip.miss {
    color: #8a3428;
    background: rgba(244, 114, 102, 0.14);
    border: 1px solid rgba(181, 78, 58, 0.16);
  }
  .block {
    border: 1px solid rgba(70, 53, 32, 0.08);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.62);
    padding: 14px;
  }
  .block-label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .markdown-body { color: var(--text); line-height: 1.7; word-break: break-word; }
  .markdown-body h1, .markdown-body h2, .markdown-body h3, .markdown-body h4, .markdown-body h5, .markdown-body h6 {
    margin: 14px 0 8px;
    line-height: 1.3;
  }
  .markdown-body p + p { margin-top: 10px; }
  .markdown-body ul, .markdown-body ol { padding-left: 20px; margin: 8px 0; }
  .markdown-body code {
    background: rgba(14, 116, 144, 0.12);
    padding: 2px 6px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
  }
  .markdown-body pre {
    background: var(--code-bg);
    color: var(--code-text);
    padding: 14px;
    border-radius: 14px;
    overflow: auto;
    margin: 10px 0;
  }
  .markdown-body pre code { background: transparent; padding: 0; color: inherit; }
  .tool-call { border-left: 4px solid rgba(14, 116, 144, 0.5); }
  .tool-result { border-left: 4px solid rgba(37, 99, 235, 0.45); }
  .media-block { border-left: 4px solid rgba(180, 83, 9, 0.45); }
  .unknown-block { border-left: 4px solid rgba(170, 58, 42, 0.42); }
  .code-panel {
    background: var(--code-bg);
    color: var(--code-text);
    border-radius: 14px;
    padding: 14px;
    overflow: auto;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .json-key { color: #8ed7ff; }
  .json-str { color: #9fe3a7; }
  .json-num { color: #f7b56d; }
  .json-bool { color: #f3d070; }
  .json-null { color: #d0bfa8; }
  .new-flash { animation: flash .6s ease; }
  @keyframes flash {
    0%,100% { box-shadow: 0 10px 28px rgba(78, 60, 37, 0.04); }
    50% { box-shadow: 0 18px 32px rgba(14, 116, 144, 0.18); }
  }
  @media (max-width: 980px) {
    header { flex-direction: column; align-items: stretch; }
    .toolbar { justify-content: stretch; }
    .main { flex-direction: column; }
    .list-panel { width: 100%; border-right: 0; border-bottom: 1px solid var(--border); max-height: 42vh; }
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <div class="brand-mark">LLM</div>
    <div class="brand-copy">
      <h1>LLM 请求检视器</h1>
      <p>默认展示结构化对话视图，保留原始 JSON 便于精确调试。</p>
    </div>
  </div>
  <div class="status-group">
    <span id="status-dot"></span>
    <span class="badge" id="total-badge">0</span>
  </div>
  <div class="toolbar">
    <div class="toolbar-group">
      <input type="text" id="filter-input" placeholder="过滤 model / api…">
      <a href="/_inspector/analytics"><button type="button">统计页</button></a>
      <button id="import-toggle-btn">导入 JSON</button>
      <button id="auto-scroll-btn" class="active" title="自动滚动到最新">跟随最新</button>
    </div>
    <div class="toolbar-group">
      <button id="pretty-btn" class="active">渲染视图</button>
      <button id="json-btn">JSON 视图</button>
      <button id="copy-btn">复制 JSON</button>
      <button id="clear-btn" class="danger">清空</button>
    </div>
  </div>
</header>
<div class="main">
  <div class="list-panel">
    <div class="import-panel" id="import-panel">
      <div class="import-row">
        <div>
          <strong>导入外部请求 JSON</strong>
          <div class="import-hint">支持原始请求体、包含 params 的对象、对象数组，或带 requests 数组的导出文件。</div>
        </div>
        <div class="import-actions">
          <input type="file" id="import-file-input" accept=".json,application/json" multiple hidden>
          <button id="import-file-btn">选择文件</button>
          <button id="import-submit-btn">导入文本</button>
        </div>
      </div>
      <textarea id="import-textarea" placeholder='可直接粘贴 JSON，例如：
{
  "api_name": "chat.completions.create",
  "params": {
    "model": "gpt-4.1",
    "messages": [{"role": "user", "content": "hello"}]
  },
  "metadata": {"api_provider": "OpenAI"}
}'></textarea>
      <div class="status-text" id="import-status"></div>
    </div>
    <div class="list-scroll" id="list-scroll">
      <div class="empty-tip" id="empty-tip">等待请求捕获…</div>
    </div>
  </div>
  <div class="detail-panel">
    <div class="detail-toolbar">
      <span class="detail-title" id="detail-title">选择左侧记录查看结构化对话视图</span>
    </div>
    <div class="detail-body" id="detail-body">
      <div class="empty-tip">左侧选择一条请求后，这里会显示摘要、工具定义和消息渲染。</div>
    </div>
  </div>
</div>
<script>
const listScroll = document.getElementById('list-scroll');
const detailBody = document.getElementById('detail-body');
const detailTitle = document.getElementById('detail-title');
const totalBadge = document.getElementById('total-badge');
const filterInput = document.getElementById('filter-input');
const statusDot = document.getElementById('status-dot');
const autoScrollBtn = document.getElementById('auto-scroll-btn');
const prettyBtn = document.getElementById('pretty-btn');
const jsonBtn = document.getElementById('json-btn');
const importToggleBtn = document.getElementById('import-toggle-btn');
const importPanel = document.getElementById('import-panel');
const importTextarea = document.getElementById('import-textarea');
const importStatus = document.getElementById('import-status');
const importFileInput = document.getElementById('import-file-input');
const importFileBtn = document.getElementById('import-file-btn');
const importSubmitBtn = document.getElementById('import-submit-btn');

let requests = [];
let activeId = null;
let autoScroll = true;
let viewMode = 'pretty';
let fullCache = {};
const CACHE_PROBE_WINDOW = 8;

function connectSSE() {
  const es = new EventSource('/_inspector/api/stream');
  statusDot.classList.remove('live');
  es.addEventListener('snapshot', event => {
    statusDot.classList.add('live');
    requests = JSON.parse(event.data);
    renderList();
  });
  es.addEventListener('new', event => {
    const record = JSON.parse(event.data);
    requests.push(record);
    appendItem(record, true);
    totalBadge.textContent = requests.length;
    if (autoScroll) listScroll.scrollTop = listScroll.scrollHeight;
  });
  es.onerror = () => {
    statusDot.classList.remove('live');
    es.close();
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

async function refreshRequests() {
  const response = await fetch('/_inspector/api/requests');
  requests = await response.json();
  renderList();
}

function filterText() {
  return filterInput.value.trim().toLowerCase();
}

function matchFilter(record) {
  const f = filterText();
  if (!f) return true;
  return String(record.model || '').toLowerCase().includes(f) || String(record.api_name || '').toLowerCase().includes(f);
}

function renderList() {
  listScroll.innerHTML = '';
  const filtered = requests.filter(matchFilter);
  if (filtered.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-tip';
    empty.textContent = requests.length ? '无匹配记录' : '等待请求捕获…';
    listScroll.appendChild(empty);
    totalBadge.textContent = requests.length;
    return;
  }
  filtered.forEach(record => appendItem(record, false));
  totalBadge.textContent = requests.length;
  if (autoScroll) listScroll.scrollTop = listScroll.scrollHeight;
}

function appendItem(record, flash) {
  if (!matchFilter(record)) return;
  const item = document.createElement('div');
  item.className = 'req-item' + (record.id === activeId ? ' active' : '') + (flash ? ' new-flash' : '');
  item.dataset.id = record.id;
  const provider = record.api_provider ? escHtml(record.api_provider) : '-';
  const tokens = record.estimated_input_tokens == null ? '-' : escHtml(record.estimated_input_tokens);
  item.innerHTML = `<div class="row1"><span class="api-chip">${escHtml(record.api_name)}</span><span class="ts">${escHtml(record.ts_str)}</span></div>
    <div class="model">${escHtml(record.model)}</div>
    <div class="meta">${record.msg_count} 条消息 · ${record.tool_count} 个工具</div>
    <div class="meta-row"><span class="mini-chip">Provider · ${provider}</span><span class="mini-chip">Tokens · ${tokens}</span></div>`;
  item.addEventListener('click', () => selectItem(record.id));
  listScroll.appendChild(item);
}

filterInput.addEventListener('input', renderList);

async function selectItem(id) {
  activeId = id;
  document.querySelectorAll('.req-item').forEach(el => el.classList.toggle('active', +el.dataset.id === id));
  detailTitle.textContent = '加载中…';
  await ensureRequestDetails([id, ...getRecentProbeIds(id)]);
  renderActiveDetail();
}

function getRecentProbeIds(id) {
  const currentIndex = requests.findIndex(item => item.id === id);
  if (currentIndex <= 0) {
    return [];
  }
  return requests.slice(Math.max(0, currentIndex - CACHE_PROBE_WINDOW), currentIndex).map(item => item.id);
}

async function ensureRequestDetails(ids) {
  const pending = ids.filter(reqId => reqId != null && !fullCache[reqId]).map(async reqId => {
    const res = await fetch(`/_inspector/api/requests/${reqId}`);
    fullCache[reqId] = await res.json();
  });
  if (pending.length) {
    await Promise.all(pending);
  }
}

function renderActiveDetail() {
  if (activeId == null || !fullCache[activeId]) {
    detailBody.innerHTML = '<div class="empty-tip">左侧选择一条请求后，这里会显示详情。</div>';
    detailTitle.textContent = '选择左侧记录查看结构化对话视图';
    return;
  }
  const data = fullCache[activeId];
  const record = requests.find(item => item.id === activeId);
  detailTitle.textContent = record ? `#${activeId} · ${record.api_name} · ${record.ts_str}` : `#${activeId}`;
  if (viewMode === 'json') {
    renderJsonDetail(data.params);
    return;
  }
  renderPrettyDetail(data.rendered || {});
}

function renderPrettyDetail(rendered) {
  detailBody.innerHTML = '';
  const fragment = document.createDocumentFragment();
  fragment.appendChild(renderOverviewSection(rendered.overview || []));
  fragment.appendChild(renderToolsSection(rendered.tools || []));
  fragment.appendChild(renderMessagesSection(rendered.messages || [], buildCacheProbeStates(rendered.messages || [])));
  detailBody.appendChild(fragment);
}

function renderOverviewSection(items) {
  const section = document.createElement('section');
  section.className = 'section';
  section.innerHTML = '<div class="section-head"><h2>请求摘要</h2><span class="section-hint">顶层参数与统计</span></div>';
  const grid = document.createElement('div');
  grid.className = 'overview-grid';
  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'empty-tip';
    empty.textContent = '没有可展示的顶层参数。';
    section.appendChild(empty);
    return section;
  }
  items.forEach(item => {
    const card = document.createElement('div');
    card.className = 'overview-item';
    card.innerHTML = `<div class="label">${escHtml(item.label || '')}</div><div class="value">${escHtml(item.value || '')}</div>`;
    grid.appendChild(card);
  });
  section.appendChild(grid);
  return section;
}

function renderToolsSection(tools) {
  const section = document.createElement('section');
  section.className = 'section';
  section.innerHTML = '<div class="section-head"><h2>Tools 定义</h2><span class="section-hint">函数签名、参数要求与原始 schema</span></div>';
  if (!tools.length) {
    section.innerHTML += '<div class="empty-tip">本次请求没有携带 tools 定义。</div>';
    return section;
  }
  tools.forEach(tool => {
    const details = document.createElement('details');
    details.className = 'tool-card';
    details.open = tool.index === 0;
    const propertyHtml = (tool.properties || []).map(prop => {
      const requiredLabel = prop.required ? '必填' : '可选';
      return `<div class="tool-prop"><div class="name">${escHtml(prop.name || '')}</div><div class="meta">${escHtml(prop.type || '')} · ${requiredLabel}</div><div class="meta">${escHtml(prop.description || '无描述')}</div></div>`;
    }).join('');
    details.innerHTML = `<summary><div><div class="tool-name">${escHtml(tool.name || '')}</div><div class="tool-kind">${escHtml(tool.kind || '')}</div></div><div class="section-hint">${(tool.required || []).length} 个必填参数</div></summary>
      <div class="tool-body">
        <div class="tool-desc">${escHtml(tool.description || '无描述')}</div>
        ${(tool.properties || []).length ? `<div class="tool-props">${propertyHtml}</div>` : '<div class="empty-tip">没有显式参数定义。</div>'}
        <div class="code-panel">${syntaxHighlight(tool.raw_json || '{}')}</div>
      </div>`;
    section.appendChild(details);
  });
  return section;
}

function renderMessagesSection(messages, cacheProbeStates) {
  const section = document.createElement('section');
  section.className = 'section';
  section.innerHTML = `<div class="section-head"><h2>Messages 对话流</h2><span class="section-hint">按 role 拆分板块，渲染 Markdown、工具调用与结果。绿色表示最近 ${CACHE_PROBE_WINDOW} 条同类请求中存在相同前缀，红色表示本地模拟未命中或前缀在此断开。</span></div>`;
  if (!messages.length) {
    section.innerHTML += '<div class="empty-tip">本次请求没有 messages，或并非 chat 类型请求。</div>';
    return section;
  }
  const stack = document.createElement('div');
  stack.className = 'message-stack';
  messages.forEach((message, index) => stack.appendChild(renderMessageCard(message, cacheProbeStates[index] || null)));
  section.appendChild(stack);
  return section;
}

function renderMessageCard(message, cacheProbeState) {
  const card = document.createElement('article');
  const role = normalizeRoleClass(message.role || 'unknown');
  const probeClass = cacheProbeState && cacheProbeState.hit ? ' cache-hit-candidate' : (cacheProbeState ? ' cache-miss-candidate' : '');
  card.className = `message-card role-${role}${probeClass}`;
  const metaText = message.meta ? escHtml(message.meta) : '&nbsp;';
  const probeChip = cacheProbeState
    ? `<span class="cache-probe-chip ${cacheProbeState.hit ? 'hit' : 'miss'}">${escHtml(cacheProbeState.label || '')}</span>`
    : '';
  card.innerHTML = `<div class="message-head"><div class="message-role"><span class="dot"></span><span>${escHtml(message.label || role)}</span></div><div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;justify-content:flex-end">${probeChip}<span class="message-meta">${metaText}</span></div></div>`;
  const body = document.createElement('div');
  body.className = 'message-body';
  (message.blocks || []).forEach(block => body.appendChild(renderBlock(block)));
  card.appendChild(body);
  return card;
}

function buildCacheProbeStates(messages) {
  if (activeId == null || !messages.length) {
    return [];
  }

  const currentData = fullCache[activeId] || {};
  const currentRecord = requests.find(item => item.id === activeId) || {};
  const currentGroup = getCacheProbeGroupKey(currentRecord, currentData);
  const currentFingerprints = messages.map(message => fingerprintMessageForCacheProbe(message));
  const candidates = getRecentProbeIds(activeId)
    .map(reqId => ({ record: requests.find(item => item.id === reqId) || {}, data: fullCache[reqId] || null }))
    .filter(entry => entry.data && getCacheProbeGroupKey(entry.record, entry.data) === currentGroup)
    .map(entry => (entry.data.rendered && Array.isArray(entry.data.rendered.messages) ? entry.data.rendered.messages : []))
    .map(candidateMessages => candidateMessages.map(message => fingerprintMessageForCacheProbe(message)));

  return currentFingerprints.map((fingerprint, index) => {
    let hit = false;
    let prefixBreak = false;
    for (const candidate of candidates) {
      if (index > 0 && hasMatchingPrefix(candidate, currentFingerprints, index - 1) && !hasMatchingPrefix(candidate, currentFingerprints, index)) {
        prefixBreak = true;
      }
      if (hasMatchingPrefix(candidate, currentFingerprints, index)) {
        hit = true;
        break;
      }
    }

    if (hit) {
      return { hit: true, label: '命中候选' };
    }
    if (prefixBreak) {
      return { hit: false, label: '前缀在此断开' };
    }
    return { hit: false, label: candidates.length ? '最近样本未命中' : '无对照样本' };
  });
}

function hasMatchingPrefix(candidateFingerprints, currentFingerprints, endIndex) {
  if (!Array.isArray(candidateFingerprints) || candidateFingerprints.length <= endIndex) {
    return false;
  }
  for (let index = 0; index <= endIndex; index += 1) {
    if (candidateFingerprints[index] !== currentFingerprints[index]) {
      return false;
    }
  }
  return true;
}

function getCacheProbeGroupKey(record, data) {
  const metadata = data && data.metadata ? data.metadata : {};
  const provider = metadata.api_provider || record.api_provider || '-';
  const requestName = metadata.request_name || '';
  return [record.api_name || '', provider, record.model || '', requestName].join('|');
}

function fingerprintMessageForCacheProbe(message) {
  const blocks = Array.isArray(message.blocks) ? message.blocks.map(block => normalizeBlockForCacheProbe(block)) : [];
  return JSON.stringify({ role: message.role || 'unknown', blocks });
}

function normalizeBlockForCacheProbe(block) {
  if (!block || typeof block !== 'object') {
    return block;
  }
  return {
    type: block.type || 'unknown',
    text: block.text || '',
    label: block.label || '',
    call_id: block.call_id || '',
    name: block.name || '',
    arguments_text: block.arguments_text || '',
    title: block.title || '',
    meta: block.meta || '',
    media_type: block.media_type || '',
  };
}

function renderBlock(block) {
  const container = document.createElement('div');
  const type = block.type || 'unknown';
  container.className = 'block';
  if (block.label) {
    const label = document.createElement('div');
    label.className = 'block-label';
    label.textContent = block.label;
    container.appendChild(label);
  }
  if (type === 'markdown') {
    const body = document.createElement('div');
    body.className = 'markdown-body';
    body.innerHTML = renderMarkdown(block.text || '');
    container.appendChild(body);
    return container;
  }
  if (type === 'tool_call') {
    container.classList.add('tool-call');
    container.innerHTML += `<div class="block-label">工具调用 ${block.call_id ? '· ' + escHtml(block.call_id) : ''}</div><div><strong>${escHtml(block.name || '')}</strong></div><div class="code-panel">${syntaxHighlight(block.arguments_text || '{}')}</div>`;
    return container;
  }
  if (type === 'tool_result') {
    container.classList.add('tool-result');
    container.innerHTML += `<div class="block-label">工具结果${block.call_id ? ' · ' + escHtml(block.call_id) : ''}${block.name ? ' · ' + escHtml(block.name) : ''}</div><div class="markdown-body">${renderMarkdown(block.text || '')}</div>`;
    return container;
  }
  if (type === 'media') {
    container.classList.add('media-block');
    container.innerHTML += `<div><strong>${escHtml(block.title || '媒体内容')}</strong></div><div class="message-meta">${escHtml(block.meta || '')}</div><div class="markdown-body">${renderMarkdown(block.text || '')}</div>`;
    return container;
  }
  if (type === 'empty') {
    container.innerHTML += `<div class="message-meta">${escHtml(block.text || '')}</div>`;
    return container;
  }
  container.classList.add('unknown-block');
  container.innerHTML += `<div class="block-label">${escHtml(block.label || '未知内容')}</div><div class="code-panel">${syntaxHighlight(block.text || '{}')}</div>`;
  return container;
}

function renderJsonDetail(params) {
  detailBody.innerHTML = '';
  const section = document.createElement('section');
  section.className = 'section';
  section.innerHTML = '<div class="section-head"><h2>原始 JSON</h2><span class="section-hint">保留完整调试视图</span></div>';
  const code = document.createElement('div');
  code.className = 'code-panel';
  code.innerHTML = syntaxHighlight(JSON.stringify(params, null, 2));
  section.appendChild(code);
  detailBody.appendChild(section);
}

function normalizeRoleClass(role) {
  return ['system', 'user', 'assistant', 'tool'].includes(role) ? role : 'unknown';
}

autoScrollBtn.addEventListener('click', () => {
  autoScroll = !autoScroll;
  autoScrollBtn.classList.toggle('active', autoScroll);
  autoScrollBtn.textContent = autoScroll ? '跟随最新' : '暂停跟随';
});

prettyBtn.addEventListener('click', () => {
  viewMode = 'pretty';
  prettyBtn.classList.add('active');
  jsonBtn.classList.remove('active');
  renderActiveDetail();
});

jsonBtn.addEventListener('click', () => {
  viewMode = 'json';
  jsonBtn.classList.add('active');
  prettyBtn.classList.remove('active');
  renderActiveDetail();
});

document.getElementById('clear-btn').addEventListener('click', async () => {
  if (!confirm('确定清空所有记录？')) return;
  await fetch('/_inspector/api/requests', { method: 'DELETE' });
  requests = [];
  fullCache = {};
  activeId = null;
  renderList();
  renderActiveDetail();
  totalBadge.textContent = '0';
});

document.getElementById('copy-btn').addEventListener('click', async () => {
  if (activeId == null || !fullCache[activeId]) return;
  await navigator.clipboard.writeText(JSON.stringify(fullCache[activeId].params, null, 2));
});

function setImportStatus(message, kind = '') {
  importStatus.textContent = message;
  importStatus.className = 'status-text' + (kind ? ` ${kind}` : '');
}

async function importJsonPayload(payload) {
  setImportStatus('正在导入…');
  const response = await fetch('/_inspector/api/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || '导入失败');
  }
  await refreshRequests();
  fullCache = {};
  const latest = (result.items || []).at(-1);
  if (latest) {
    await selectItem(latest.id);
  } else {
    renderActiveDetail();
  }
  const count = Number(result.count || 0);
  setImportStatus(`已导入 ${count} 条记录`, 'success');
}

importToggleBtn.addEventListener('click', () => {
  const isOpen = importPanel.classList.toggle('open');
  importToggleBtn.classList.toggle('active', isOpen);
  if (isOpen) {
    importTextarea.focus();
  }
});

importSubmitBtn.addEventListener('click', async () => {
  const text = importTextarea.value.trim();
  if (!text) {
    setImportStatus('请输入或粘贴 JSON 内容', 'error');
    return;
  }
  try {
    await importJsonPayload(JSON.parse(text));
  } catch (error) {
    setImportStatus(error.message || '导入失败', 'error');
  }
});

importFileBtn.addEventListener('click', () => importFileInput.click());

importFileInput.addEventListener('change', async event => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  try {
    const payloads = [];
    for (const file of files) {
      const text = await file.text();
      payloads.push(JSON.parse(text));
    }
    await importJsonPayload(payloads.length === 1 ? payloads[0] : payloads);
  } catch (error) {
    setImportStatus(error.message || '文件导入失败', 'error');
  } finally {
    importFileInput.value = '';
  }
});

function escHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderInlineMarkdown(text) {
  let rendered = escHtml(text);
  rendered = rendered.replace(/`([^`]+)`/g, '<code>$1</code>');
  rendered = rendered.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  rendered = rendered.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return rendered;
}

function renderParagraph(lines) {
  if (!lines.length) return '';
  return `<p>${lines.map(renderInlineMarkdown).join('<br>')}</p>`;
}

function renderMarkdown(text) {
  const source = String(text || '');
  const codeBlocks = [];
  const placeholderPrefix = '__CODE_BLOCK__';
  const withoutCode = source.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, language = '', code = '') => {
    const index = codeBlocks.length;
    codeBlocks.push(`<pre><code>${escHtml(code.replace(/\n$/, ''))}</code></pre>`);
    return `${placeholderPrefix}${index}__`;
  });
  const lines = withoutCode.split(/\r?\n/);
  const parts = [];
  let paragraph = [];
  let listType = null;
  let listItems = [];

  function flushParagraph() {
    if (paragraph.length) {
      parts.push(renderParagraph(paragraph));
      paragraph = [];
    }
  }

  function flushList() {
    if (listType && listItems.length) {
      parts.push(`<${listType}>${listItems.map(item => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</${listType}>`);
    }
    listType = null;
    listItems = [];
  }

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const placeholderMatch = trimmed.match(new RegExp(`^${placeholderPrefix}(\\d+)__$`));
    if (placeholderMatch) {
      flushParagraph();
      flushList();
      parts.push(codeBlocks[Number(placeholderMatch[1])] || '');
      return;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 6);
      parts.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }

    const unordered = trimmed.match(/^[-*+]\s+(.+)$/);
    if (unordered) {
      flushParagraph();
      if (listType && listType !== 'ul') flushList();
      listType = 'ul';
      listItems.push(unordered[1]);
      return;
    }

    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);
    if (ordered) {
      flushParagraph();
      if (listType && listType !== 'ol') flushList();
      listType = 'ol';
      listItems.push(ordered[1]);
      return;
    }

    flushList();
    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();

  if (!parts.length) {
    return `<p>${renderInlineMarkdown(source).replace(/\n/g, '<br>')}</p>`;
  }
  return parts.join('');
}

function syntaxHighlight(json) {
  const normalized = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
  return escHtml(normalized).replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(?:\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, match => {
    let cls = 'json-num';
    if (/^"/.test(match)) cls = /:$/.test(match) ? 'json-key' : 'json-str';
    else if (/true|false/.test(match)) cls = 'json-bool';
    else if (/null/.test(match)) cls = 'json-null';
    return `<span class="${cls}">${match}</span>`;
  });
}
</script>
</body>
</html>"""


_ANALYTICS_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM 统计仪表盘</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  :root{--bg:#f4efe7;--panel:rgba(255,252,246,0.92);--border:rgba(70,53,32,0.12);--text:#2f261d;--muted:#7a6856;--accent:#0e7490;--accent-soft:rgba(14,116,144,0.14);--success:#1f7a4d;--danger:#aa3a2a;--amber:#b45309;--rose:#c2416c;--ink:#3b3228}
  body{background:radial-gradient(circle at top left,rgba(14,116,144,0.14),transparent 28%),radial-gradient(circle at top right,rgba(180,83,9,0.12),transparent 32%),linear-gradient(180deg,var(--bg) 0%,#f8f3eb 100%);color:var(--text);font-family:'Segoe UI','PingFang SC',sans-serif;font-size:14px;padding:24px;min-height:100vh}
  header{margin-bottom:24px;display:flex;justify-content:space-between;align-items:center}
  h1{font-size:22px;font-weight:700}
  .nav{display:flex;gap:12px}
  .nav a{padding:8px 16px;border-radius:12px;border:1px solid var(--border);text-decoration:none;color:var(--text);font-size:13px}
  .nav a:hover{background:var(--panel);border-color:var(--accent)}
  .kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:24px}
  .kpi{border:1px solid var(--border);border-radius:18px;background:var(--panel);padding:18px;box-shadow:0 12px 32px rgba(78,60,37,0.06)}
  .kpi .kpi-label{color:var(--muted);font-size:12px}
  .kpi .kpi-value{margin-top:10px;font-size:26px;font-weight:700}
  .kpi .kpi-sub{color:var(--muted);font-size:11px;margin-top:4px}
  .chart-grid{display:grid;grid-template-columns:1.15fr 1fr;gap:18px;margin-bottom:18px}
  .chart-card{background:var(--panel);border:1px solid var(--border);border-radius:22px;padding:20px;box-shadow:0 12px 32px rgba(78,60,37,0.06);overflow:hidden}
  .chart-card h2{font-size:16px;margin-bottom:6px}
  .chart-subtitle{color:var(--muted);font-size:12px;margin-bottom:16px}
  .chart-stack{display:grid;gap:18px}
  .ring-layout{display:grid;grid-template-columns:160px 1fr;gap:20px;align-items:center}
  .ring{width:160px;height:160px;border-radius:50%;display:grid;place-items:center;background:conic-gradient(var(--accent) 0deg, var(--accent) 180deg, rgba(0,0,0,0.08) 180deg 360deg);position:relative;margin:auto}
  .ring::after{content:"";position:absolute;inset:18px;border-radius:50%;background:linear-gradient(180deg,#fffaf2 0%,#fbf6ee 100%);box-shadow:inset 0 1px 0 rgba(255,255,255,0.8)}
  .ring-center{position:relative;z-index:1;text-align:center}
  .ring-value{font-size:28px;font-weight:700;line-height:1}
  .ring-label{font-size:12px;color:var(--muted);margin-top:6px}
  .legend{display:grid;gap:10px}
  .legend-item{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center}
  .swatch{width:10px;height:10px;border-radius:999px}
  .legend-name{font-size:13px;color:var(--ink)}
  .legend-value{font-size:12px;color:var(--muted)}
  .metric-rail{height:12px;border-radius:999px;background:rgba(60,50,40,0.08);overflow:hidden;display:flex}
  .metric-fill{height:100%}
  .chart-list{display:grid;gap:12px}
  .chart-row{display:grid;grid-template-columns:minmax(0,160px) 1fr auto;gap:12px;align-items:center}
  .chart-label{font-size:12px;color:var(--ink);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .chart-value{font-size:12px;color:var(--muted)}
  .bar-track{height:14px;border-radius:999px;background:rgba(60,50,40,0.08);overflow:hidden;position:relative}
  .bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent),#2aa1b7)}
  .bar-fill.alt{background:linear-gradient(90deg,#d97706,#f59e0b)}
  .bar-fill.rose{background:linear-gradient(90deg,#be185d,#ec4899)}
  .two-up{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  .section2{background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:20px;margin-bottom:18px;box-shadow:0 12px 32px rgba(78,60,37,0.06)}
  .section2 h2{font-size:16px;margin-bottom:14px}
  table{width:100%;border-collapse:collapse}
  th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--border);font-size:13px}
  th{color:var(--muted);font-weight:600;font-size:11px}
  .tag{padding:3px 8px;border-radius:999px;font-size:11px;font-weight:600}
  .tag-ok{background:rgba(31,122,77,0.12);color:var(--success)}
  .tag-warn{background:rgba(180,83,9,0.12);color:#b45309}
  .empty{text-align:center;color:var(--muted);padding:48px}
  .refresh-bar{display:flex;gap:10px;align-items:center;margin-bottom:18px}
  .refresh-bar button{padding:8px 14px;border:1px solid var(--border);border-radius:10px;background:var(--panel);cursor:pointer;font-size:12px}
  .refresh-bar button:hover{border-color:var(--accent)}
  .mono{font-family:'JetBrains Mono','Consolas',monospace;font-size:12px}
  @media (max-width: 980px){.chart-grid,.two-up,.ring-layout{grid-template-columns:1fr}.chart-row{grid-template-columns:minmax(0,110px) 1fr auto}}
</style>
</head>
<body>
<header>
  <h1>LLM 消耗统计仪表盘</h1>
  <div class="nav">
    <a href="/_inspector/">请求检视器</a>
    <a href="/_inspector/analytics" style="background:var(--accent);color:#fff;border-color:var(--accent)">统计</a>
  </div>
</header>
<div class="refresh-bar">
  <button onclick="loadData()">刷新数据</button>
  <span id="last-update" style="color:var(--muted);font-size:12px"></span>
</div>
<div class="kpi-grid" id="kpi-grid"></div>
<div class="chart-grid">
  <div class="chart-card">
    <h2>缓存与 Token 结构</h2>
    <div class="chart-subtitle">快速查看整体命中率与输入输出比例</div>
    <div class="chart-stack">
      <div class="ring-layout">
        <div class="ring" id="cache-ring">
          <div class="ring-center">
            <div class="ring-value" id="cache-ring-value">0.0%</div>
            <div class="ring-label">全局缓存命中</div>
          </div>
        </div>
        <div class="legend" id="cache-legend"></div>
      </div>
      <div>
        <div class="chart-subtitle" style="margin-bottom:10px">Token 构成</div>
        <div class="metric-rail" id="token-composition"></div>
        <div class="legend" id="token-legend" style="margin-top:12px"></div>
      </div>
    </div>
  </div>
  <div class="chart-card">
    <h2>流量分布</h2>
    <div class="chart-subtitle">按模型和请求名称查看占比</div>
    <div class="two-up">
      <div>
        <div class="chart-subtitle" style="margin-bottom:10px">模型请求占比</div>
        <div class="chart-list" id="model-share-chart"></div>
      </div>
      <div>
        <div class="chart-subtitle" style="margin-bottom:10px">请求名称占比</div>
        <div class="chart-list" id="request-share-chart"></div>
      </div>
    </div>
  </div>
</div>
<div class="chart-card" style="margin-bottom:18px">
  <h2>聊天流比例视图</h2>
  <div class="chart-subtitle">观察最活跃聊天流的请求量与缓存命中差异</div>
  <div class="chart-list" id="stream-share-chart"></div>
</div>
<div class="section2"><h2>按模型统计</h2><div id="by-model"></div></div>
<div class="section2"><h2>按请求名称统计</h2><div id="by-request"></div></div>
<div class="section2"><h2>按聊天流统计（含缓存命中率）</h2><div id="by-stream"></div></div>
<script>
async function loadData() {
  try {
    const resp = await fetch('/_inspector/api/analytics');
    const data = await resp.json();
    renderKPIs(data.summary || {});
    renderVisualOverview(data.summary || {}, data.by_model || [], data.by_request_name || [], data.by_stream || []);
    renderTable('by-model', data.by_model || [], ['model_name','api_provider','total_requests','total_tokens','total_cost','avg_latency']);
    renderTable('by-request', data.by_request_name || [], ['request_name','total_requests','total_tokens','total_cost','avg_latency']);
    renderStreamTable('by-stream', data.by_stream || []);
    document.getElementById('last-update').textContent = '最后更新: ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}
function renderKPIs(s) {
  const grid = document.getElementById('kpi-grid');
  const f = (n) => n != null ? Number(n).toLocaleString() : '-';
  const p = (n) => n != null ? (Number(n)*100).toFixed(1)+'%' : '-';
  const c = (n) => n != null ? '$'+Number(n).toFixed(4) : '-';
  const cacheObserved = Number(s.total_cache_hit_tokens || 0) + Number(s.total_cache_miss_tokens || 0) > 0;
  const items = [
    {l:'总请求数',v:f(s.total_requests)},
    {l:'成功率',v:p(s.success_rate),sub:(s.success_count||0)+' / '+(s.error_count||0)},
    {l:'总 Token',v:f(s.total_tokens),sub:'in:'+f(s.total_prompt_tokens)+' out:'+f(s.total_completion_tokens)},
    {l:'缓存命中率',v:p(s.cache_hit_rate),sub:cacheObserved ? 'hit:'+f(s.total_cache_hit_tokens)+' miss:'+f(s.total_cache_miss_tokens) : ((s.total_requests||0) > 0 ? '当前记录未返回缓存指标' : 'hit:0 miss:0')},
    {l:'总成本',v:c(s.total_cost)},
    {l:'平均延迟',v:s.avg_latency != null ? Number(s.avg_latency).toFixed(2)+'s' : '-'},
  ];
  grid.innerHTML = items.map(i=>`<div class="kpi"><div class="kpi-label">${i.l}</div><div class="kpi-value">${i.v}</div>${i.sub?`<div class="kpi-sub">${i.sub}</div>`:''}</div>`).join('');
}
function clampPercent(value) {
  const num = Number(value || 0);
  return Math.max(0, Math.min(100, num));
}
function renderVisualOverview(summary, byModel, byRequest, byStream) {
  renderCacheChart(summary);
  renderTokenComposition(summary);
  renderShareChart('model-share-chart', byModel, 'model_name', 'total_requests', '请求', 'accent');
  renderShareChart('request-share-chart', byRequest, 'request_name', 'total_requests', '请求', 'rose');
  renderStreamShareChart(byStream);
}
function renderCacheChart(summary) {
  const ring = document.getElementById('cache-ring');
  const ringValue = document.getElementById('cache-ring-value');
  const legend = document.getElementById('cache-legend');
  const hit = Number(summary.total_cache_hit_tokens || 0);
  const miss = Number(summary.total_cache_miss_tokens || 0);
  const total = hit + miss;
  const hitRate = total > 0 ? hit / total : 0;
  const angle = clampPercent(hitRate * 100) * 3.6;
  ring.style.background = `conic-gradient(var(--accent) 0deg ${angle}deg, rgba(60,50,40,0.09) ${angle}deg 360deg)`;
  ringValue.textContent = total > 0 ? `${(hitRate * 100).toFixed(1)}%` : ((summary.total_requests || 0) > 0 ? 'N/A' : '0.0%');
  legend.innerHTML = [
    {label:'缓存命中',value:hit,color:'var(--accent)'},
    {label:'缓存未命中',value:miss,color:'var(--amber)'},
    {label:'指标状态',value:total > 0 ? '已采集' : ((summary.total_requests || 0) > 0 ? '未返回' : '暂无数据'),color:'rgba(60,50,40,0.32)'}
  ].map(item => `
    <div class="legend-item">
      <span class="swatch" style="background:${item.color}"></span>
      <span class="legend-name">${item.label}</span>
      <span class="legend-value">${typeof item.value === 'number' ? Number(item.value).toLocaleString() : item.value}</span>
    </div>
  `).join('');
}
function renderTokenComposition(summary) {
  const track = document.getElementById('token-composition');
  const legend = document.getElementById('token-legend');
  const prompt = Number(summary.total_prompt_tokens || 0);
  const completion = Number(summary.total_completion_tokens || 0);
  const total = prompt + completion;
  const promptWidth = total > 0 ? (prompt / total) * 100 : 0;
  const completionWidth = total > 0 ? (completion / total) * 100 : 0;
  track.innerHTML = `
    <div class="metric-fill" style="width:${promptWidth}%;background:linear-gradient(90deg,var(--accent),#2aa1b7)"></div>
    <div class="metric-fill" style="width:${completionWidth}%;background:linear-gradient(90deg,#c2416c,#ec4899)"></div>
  `;
  legend.innerHTML = [
    {label:'Prompt Tokens',value:prompt,color:'var(--accent)'},
    {label:'Completion Tokens',value:completion,color:'var(--rose)'}
  ].map(item => `
    <div class="legend-item">
      <span class="swatch" style="background:${item.color}"></span>
      <span class="legend-name">${item.label}</span>
      <span class="legend-value">${Number(item.value).toLocaleString()}</span>
    </div>
  `).join('');
}
function renderShareChart(id, rows, labelKey, valueKey, unitLabel, tone) {
  const container = document.getElementById(id);
  if (!rows.length) {
    container.innerHTML = '<div class="empty">暂无数据</div>';
    return;
  }
  const items = rows.slice(0, 6);
  const maxValue = Math.max(...items.map(row => Number(row[valueKey] || 0)), 1);
  const fillClass = tone === 'rose' ? 'bar-fill rose' : tone === 'accent' ? 'bar-fill' : 'bar-fill alt';
  container.innerHTML = items.map(row => {
    const value = Number(row[valueKey] || 0);
    const width = (value / maxValue) * 100;
    const label = String(row[labelKey] || '(未命名)');
    return `
      <div class="chart-row">
        <div class="chart-label" title="${label}">${label}</div>
        <div class="bar-track"><div class="${fillClass}" style="width:${width}%"></div></div>
        <div class="chart-value">${value.toLocaleString()} ${unitLabel}</div>
      </div>
    `;
  }).join('');
}
function renderStreamShareChart(rows) {
  const container = document.getElementById('stream-share-chart');
  if (!rows.length) {
    container.innerHTML = '<div class="empty">暂无按聊天流的数据</div>';
    return;
  }
  const items = rows.slice(0, 8);
  const maxRequests = Math.max(...items.map(row => Number(row.total_requests || 0)), 1);
  container.innerHTML = items.map(row => {
    const requests = Number(row.total_requests || 0);
    const width = (requests / maxRequests) * 100;
    const hitRate = Number(row.cache_hit_rate || 0);
    const label = String(row.stream_id || '-').slice(0, 24);
    return `
      <div class="chart-row">
        <div class="chart-label mono" title="${row.stream_id || '-'}">${label}</div>
        <div class="bar-track"><div class="bar-fill alt" style="width:${width}%"></div></div>
        <div class="chart-value">${requests.toLocaleString()} 请求 / ${(hitRate * 100).toFixed(1)}%</div>
      </div>
    `;
  }).join('');
}
function renderTable(id, rows, cols) {
  const container = document.getElementById(id);
  if (!rows.length) { container.innerHTML = '<div class="empty">暂无数据</div>'; return; }
  const labels = cols.map(c=>c.replace(/_/g,' ').replace(/\b\w/g,l=>l.toUpperCase()));
  const fmt = (v,k) => {
    if (k === 'total_cost') return v != null ? '$'+Number(v).toFixed(4) : '-';
    if (k === 'avg_latency') return v != null ? Number(v).toFixed(2)+'s' : '-';
    if (k && k.includes('token')) return v != null ? Number(v).toLocaleString() : '-';
    return v != null ? String(v) : '-';
  };
  container.innerHTML = `<table><thead><tr>${labels.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${fmt(r[c],c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function renderStreamTable(id, rows) {
  const container = document.getElementById(id);
  if (!rows.length) { container.innerHTML = '<div class="empty">暂无按聊天流的数据</div>'; return; }
  container.innerHTML = `<table><thead><tr><th>Stream ID</th><th>请求数</th><th>Prompt</th><th>Completion</th><th>缓存命中</th><th>缓存未命中</th><th>命中率</th><th>成本</th></tr></thead><tbody>${rows.map(r=>`<tr>
    <td class="mono">${String(r.stream_id||'-').slice(0,24)}</td>
    <td>${r.total_requests}</td>
    <td>${Number(r.total_prompt_tokens).toLocaleString()}</td>
    <td>${Number(r.total_completion_tokens).toLocaleString()}</td>
    <td>${Number(r.total_cache_hit).toLocaleString()}</td>
    <td>${Number(r.total_cache_miss).toLocaleString()}</td>
    <td><span class="tag ${r.cache_hit_rate>0.5?'tag-ok':'tag-warn'}">${(r.cache_hit_rate*100).toFixed(1)}%</span></td>
    <td>$${Number(r.total_cost).toFixed(4)}</td>
  </tr>`).join('')}</tbody></table>`;
}
loadData();
setInterval(loadData, 30000);
</script>
</body>
</html>"""