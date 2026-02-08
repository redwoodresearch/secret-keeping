import asyncio
import threading
import time
import uuid
from typing import Any

import pydantic
import uvicorn
from fastapi import FastAPI, HTTPException

from src import ChatMessage, MessageRole, Prompt
from src.model_organisms import BasicModel


class _Message(pydantic.BaseModel):
    role: str
    content: str


class ChatCompletionRequest(pydantic.BaseModel):
    model: str
    messages: list[_Message]
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    n: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    # Accepted but ignored — model organisms are chat-only
    tools: Any | None = None
    tool_choice: Any | None = None


class _Choice(pydantic.BaseModel):
    index: int
    message: _Message
    finish_reason: str = "stop"


class _Usage(pydantic.BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(pydantic.BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: list[_Choice]
    usage: _Usage = _Usage()


class _ModelEntry(pydantic.BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "organism"


class _ModelListResponse(pydantic.BaseModel):
    object: str = "list"
    data: list[_ModelEntry]


_ROLE_MAP = {
    "system": MessageRole.system,
    "user": MessageRole.user,
    "assistant": MessageRole.assistant,
}


def _build_app(models: dict[str, BasicModel]) -> FastAPI:
    app = FastAPI(title="Model Organism API")

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        model = models.get(request.model)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found. Available: {list(models.keys())}",
            )

        # Convert OpenAI messages → internal Prompt.
        # Trailing assistant messages are kept as-is so the underlying model
        # treats them as a prefill (Anthropic-style). The response contains
        # only the continuation, matching standard OpenAI API behaviour.
        chat_messages = []
        for m in request.messages:
            role = _ROLE_MAP.get(m.role)
            if role is None:
                continue
            chat_messages.append(ChatMessage(role=role, content=m.content))

        prompt = Prompt(messages=chat_messages)
        completion = await model(prompt)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                _Choice(
                    index=0,
                    message=_Message(role="assistant", content=completion),
                )
            ],
        )

    @app.get("/v1/models")
    async def list_models() -> _ModelListResponse:
        return _ModelListResponse(
            data=[_ModelEntry(id=name) for name in models],
        )

    return app


def launch_server(
    models: dict[str, BasicModel],
    host: str = "0.0.0.0",
    port: int = 8193,
) -> None:
    """Start the OpenAI-compatible server (blocking)."""
    app = _build_app(models)
    uvicorn.run(app, host=host, port=port)


class BackgroundServer:
    """Handle returned by ``launch_server_background`` for lifecycle control."""

    def __init__(self, thread: threading.Thread, shutdown_event: threading.Event):
        self.thread = thread
        self.shutdown_event = shutdown_event

    def shutdown(self) -> None:
        self.shutdown_event.set()
        self.thread.join(timeout=5)


def launch_server_background(
    models: dict[str, BasicModel],
    host: str = "0.0.0.0",
    port: int = 8193,
) -> BackgroundServer:
    """Start the server in a background thread. Returns a ``BackgroundServer``."""
    app = _build_app(models)
    shutdown_event = threading.Event()

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    time.sleep(1)

    def _watch_shutdown():
        shutdown_event.wait()
        server.should_exit = True

    watcher = threading.Thread(target=_watch_shutdown, daemon=True)
    watcher.start()

    return BackgroundServer(thread=thread, shutdown_event=shutdown_event)
