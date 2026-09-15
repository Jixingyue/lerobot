#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""共享的 Qwen-VL 客户端。

流水线在模块之间使用单个共享的 VLM。vLLM 在可用时是首选（高吞吐量、JSON 引导解码）；
transformers 是回退方案。``stub`` 后端用于单元测试，因此夹具永远不会调用真实模型。

客户端只提供一个方法 :meth:`VlmClient.generate_json`，它：

- 接受 OpenAI/HF 风格的多模态消息列表，
- 向服务器请求 JSON 输出，
- 透明地批处理请求，
- 并在 JSON 解析失败时使用内联更正消息重新提示一次，然后再抛出。
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .config import VlmConfig


class VlmClient(Protocol):
    """每个后端必须实现的协议。"""

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        """为每个消息列表生成一个 JSON 解码响应。"""


@dataclass
class StubVlmClient:
    """单元测试中使用的确定性存根。

    测试传递一个可调用对象，该对象将*最后一条用户消息文本*（或者如果为空，
    则是完整消息列表）映射到 JSON 可序列化响应。
    """

    responder: Callable[[Sequence[dict[str, Any]]], Any]

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        return [self.responder(list(messages)) for messages in messages_batch]


def _strip_to_json(text: str) -> Any:
    text = text.strip()
    # 去除 <think>...</think> 块（Qwen3 Thinking 风格）
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>", start) + len("</think>")
        text = (text[:start] + text[end:]).strip()
    # 从聊天调优的骨干网络中去除 ```json ... ``` 围栏
    if text.startswith("```"):
        first = text.find("\n")
        last = text.rfind("```")
        if first != -1 and last != -1 and last > first:
            text = text[first + 1 : last].strip()
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    # 回退到提取第一个平衡的 {...} 块。
    obj_text = _extract_first_json_object(text)
    if obj_text is None:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(obj_text)


def _extract_first_json_object(text: str) -> str | None:
    """返回第一个平衡的 ``{...}`` 子串，忽略字符串字面量中的花括号。
    如果未找到平衡块，则返回 ``None``。"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        # 注意：``escape`` 在这里始终为 False——上面的 ``if escape`` 分支已经处理并重置了它。
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


@dataclass
class _GenericTextClient:
    """将任何文本生成可调用对象包装在 JSON 模式 + 一次重试语义中。"""

    generate_text: Callable[[Sequence[Sequence[dict[str, Any]]], int, float], list[str]]
    config: VlmConfig

    def generate_json(
        self,
        messages_batch: Sequence[Sequence[dict[str, Any]]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[Any]:
        max_tok = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        raw = self.generate_text(messages_batch, max_tok, temp)
        out: list[Any] = []
        for messages, text in zip(messages_batch, raw, strict=True):
            try:
                out.append(_strip_to_json(text))
                continue
            except (ValueError, json.JSONDecodeError):
                pass
            retry = list(messages) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. "
                        "Reply with strictly valid JSON, no prose, no fences."
                    ),
                },
            ]
            retry_text = self.generate_text([retry], max_tok, temp)[0]
            try:
                out.append(_strip_to_json(retry_text))
            except (ValueError, json.JSONDecodeError):
                # 重试后：记录预览并返回 None 而不是崩溃整个流水线。
                # 模块将 None 视为"跳过"。
                preview = retry_text.strip().replace("\n", " ")[:200]
                print(
                    f"[vlm] WARNING: failed to parse JSON after retry; preview: {preview!r}",
                    flush=True,
                )
                out.append(None)
        return out


def make_vlm_client(config: VlmConfig) -> VlmClient:
    """构建共享的 VLM 客户端。

    目前只支持 ``openai`` 后端。交付的工作流是
    Hugging Face Jobs（``lerobot-annotate --job.target=<flavor>``）：它在
    ``vllm/vllm-openai`` 镜像内启动一个 vLLM 服务器，流水线通过
    兼容 OpenAI 的 API 与之通信（``--vlm.backend=openai``，可选地通过
    ``auto_serve`` / ``serve_command`` 自动生成服务器）。以前的进程内
    ``vllm`` / ``transformers`` 后端已被移除，以将支持面保持在
    HF Jobs 路径上。

    对于 ``stub``，直接使用响应器可调用对象构造 :class:`StubVlmClient`；
    这里拒绝它以使意外误用变得明显。
    """
    if config.backend == "openai":
        return _make_openai_client(config)
    if config.backend == "stub":
        raise ValueError(
            "Use StubVlmClient(...) directly for the stub backend; make_vlm_client builds real clients."
        )
    if config.backend in {"vllm", "transformers"}:
        raise ValueError(
            f"backend={config.backend!r} (in-process local model) is not supported for now — "
            "only backend='openai' (the Hugging Face Jobs flow) is. Run the pipeline with "
            "`lerobot-annotate --job.target=<flavor>`, which serves the model with vLLM in the "
            "vllm/vllm-openai image and talks to it over the OpenAI-compatible API."
        )
    raise ValueError(f"Unknown VLM backend: {config.backend!r}")


def _make_openai_client(config: VlmConfig) -> VlmClient:
    """与任何兼容 OpenAI 的服务器通信的后端。

    兼容 ``vllm serve``、``transformers serve``、
    ``ktransformers serve`` 和托管端点。默认情况下，服务器
    应该已经在运行。设置 ``auto_serve=True`` 以使此客户端
    生成一个（默认：``transformers serve``），等待直到就绪，
    并在进程退出时关闭它。

    图像块 ``{"type":"image", "image":<PIL.Image>}`` 会被
    自动转换为 ``image_url`` 数据 URL。视频块
    ``{"type":"video", "video":[<PIL>...]}`` 在支持的地方
    作为多帧 ``video_url`` 项转发。
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "openai package is required for backend='openai'. Install with `pip install openai`."
        ) from exc

    api_base = config.api_base
    api_key = config.api_key
    auto_serve = config.auto_serve
    api_bases: list[str] = [api_base]

    print(
        f"[lerobot-annotate] backend=openai model={config.model_id} "
        f"api_base={api_base} auto_serve={auto_serve}",
        flush=True,
    )
    if auto_serve:
        if config.parallel_servers > 1:
            print(
                f"[lerobot-annotate] spawning {config.parallel_servers} parallel servers",
                flush=True,
            )
            api_bases = _spawn_parallel_inference_servers(config)
        elif _server_is_up(api_base):
            print(f"[lerobot-annotate] reusing server already up at {api_base}", flush=True)
        else:
            print("[lerobot-annotate] no server reachable; spawning one", flush=True)
            api_base = _spawn_inference_server(config)
            api_bases = [api_base]
            print(f"[lerobot-annotate] server ready at {api_base}", flush=True)

    clients = [OpenAI(base_url=base, api_key=api_key) for base in api_bases]
    # 并行模式的轮询计数器
    rr_counter = {"i": 0}

    # ``mm_processor_kwargs`` 是 vllm 特有的额外参数；transformers serve
    # 会以 HTTP 422 拒绝它。仅当通过环境变量（例如用于 vllm 的
    # ``LEROBOT_OPENAI_SEND_MM_KWARGS=1``）显式选择加入时才发送它。
    send_mm_kwargs = os.environ.get("LEROBOT_OPENAI_SEND_MM_KWARGS", "").lower() in {"1", "true", "yes"}

    rr_lock = threading.Lock()

    def _one_call(messages: Sequence[dict[str, Any]], max_tok: int, temp: float) -> str:
        api_messages, mm_kwargs = _to_openai_messages(messages)
        kwargs: dict[str, Any] = {
            "model": config.model_id,
            "messages": api_messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if config.reasoning_effort:
            kwargs["reasoning_effort"] = config.reasoning_effort
        extra_body: dict[str, Any] = {}
        if send_mm_kwargs and mm_kwargs:
            extra_body["mm_processor_kwargs"] = {**mm_kwargs, "do_sample_frames": True}
        if config.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = config.chat_template_kwargs
        if extra_body:
            kwargs["extra_body"] = extra_body
        with rr_lock:
            chosen = clients[rr_counter["i"] % len(clients)]
            rr_counter["i"] += 1
        response = chosen.chat.completions.create(**kwargs)
        # 某些兼容 OpenAI 的服务器可能返回没有消息的选择
        # （安全过滤器，或在发射内容之前花费整个预算的"思考"模型）。
        # 将其视为空回复，以便 JSON 重试路径处理它而不是崩溃运行。
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        return (message.content if message is not None else None) or ""

    def _gen(batch: Sequence[Sequence[dict[str, Any]]], max_tok: int, temp: float) -> list[str]:
        if len(batch) <= 1 or config.client_concurrency <= 1:
            return [_one_call(messages, max_tok, temp) for messages in batch]
        # 并行扇出——vllm 在服务器端对这些进行批处理。
        max_workers = min(config.client_concurrency, len(batch))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one_call, messages, max_tok, temp) for messages in batch]
            return [f.result() for f in futures]

    return _GenericTextClient(_gen, config)


def _bind_serve_port(cmd: str, port: int) -> str:
    """将 serve 命令绑定到 ``port``：如果存在则替换 ``{port}`` 占位符，
    否则当命令省略时追加 ``--port``（保留显式的 ``--port`` 不变）。
    单服务器和并行服务器路径共享此函数，因此 serve_command 永远不会
    带着字面的 ``{port}`` 到达服务器。"""
    if "{port}" in cmd:
        return cmd.replace("{port}", str(port))
    if "--port" not in cmd:
        return f"{cmd} --port {port}"
    return cmd


def _spawn_parallel_inference_servers(config: VlmConfig) -> list[str]:
    """生成 ``config.parallel_servers`` 个独立的 vllm 副本。

    每个副本：
    - 通过 ``CUDA_VISIBLE_DEVICES`` 固定到单个 GPU
    - 在 ``serve_port + i`` 上监听
    - 通过与单服务器路径相同的 atexit 钩子关闭

    返回客户端应该轮询的 ``api_base`` URL 列表。
    """
    n = config.parallel_servers
    api_bases: list[str] = []
    procs: list[subprocess.Popen] = []
    ready_events: list[threading.Event] = []
    # 多个就绪信号——uvicorn 自己的横幅在 ``--uvicorn-log-level warning`` 下被抑制，
    # 因此我们也接受 vllm 自己的
    # "Starting vLLM API server" 行和路由列表行。下面的 HTTP 探测是最终回退。
    ready_markers = (
        "Uvicorn running",
        "Application startup complete",
        "Starting vLLM API server",
        "Available routes are",
    )
    # 所有服务器流线程使用单个锁，以便来自不同服务器的多字节字符
    # 不会交错并撕裂 UTF-8 序列。
    print_lock = threading.Lock()

    base_cmd = config.serve_command or (
        f"vllm serve {shlex.quote(config.model_id)} "
        f"--tensor-parallel-size 1 "
        f"--max-model-len {config.max_model_len or 32768} "
        f"--uvicorn-log-level warning"
    )

    num_gpus = config.num_gpus if config.num_gpus > 0 else n
    for i in range(n):
        port = config.serve_port + i
        gpu = i % num_gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = _bind_serve_port(base_cmd, port)
        api_base = f"http://localhost:{port}/v1"
        api_bases.append(api_base)
        print(f"[server-{i}] launching on GPU {gpu} port {port}: {cmd}", flush=True)
        proc = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        procs.append(proc)
        ready = threading.Event()
        ready_events.append(ready)

        def _stream(idx: int, p: subprocess.Popen, ev: threading.Event) -> None:
            # 读取整行并在共享 print_lock 下原子地发出每一行，
            # 以便来自 N 个服务器的输出保持可读。
            assert p.stdout is not None
            for line in iter(p.stdout.readline, ""):
                with print_lock:
                    sys.stdout.write(f"[server-{idx}] {line}")
                    if not line.endswith(("\n", "\r")):
                        sys.stdout.write("\n")
                    sys.stdout.flush()
                if any(m in line for m in ready_markers):
                    ev.set()

        threading.Thread(target=_stream, args=(i, proc, ready), daemon=True).start()

        def _probe(idx: int, base: str, ev: threading.Event, p: subprocess.Popen) -> None:
            while not ev.is_set() and p.poll() is None:
                if _server_is_up(base):
                    print(f"[server-{idx}] ready (http probe)", flush=True)
                    ev.set()
                    return
                time.sleep(2)

        threading.Thread(target=_probe, args=(i, api_base, ready, proc), daemon=True).start()

    def _shutdown() -> None:
        for i, p in enumerate(procs):
            if p.poll() is None:
                print(f"[server-{i}] stopping pid={p.pid}", flush=True)
                p.send_signal(signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)

    atexit.register(_shutdown)

    deadline = time.monotonic() + config.serve_ready_timeout_s
    while any(not ev.is_set() for ev in ready_events) and time.monotonic() < deadline:
        for i, p in enumerate(procs):
            if p.poll() is not None:
                raise RuntimeError(
                    f"[server-{i}] inference server exited unexpectedly with rc={p.returncode}"
                )
        time.sleep(2)
    if any(not ev.is_set() for ev in ready_events):
        raise RuntimeError(f"[server] not all replicas became ready within {config.serve_ready_timeout_s}s")
    print(f"[lerobot-annotate] all {n} servers ready: {api_bases}", flush=True)
    return api_bases


def _server_is_up(api_base: str) -> bool:
    """如果 ``api_base/models`` 在 2 秒内响应 200，则返回 True。"""
    url = api_base.rstrip("/") + "/models"
    # ``api_base`` 是用户配置的本地服务器 URL，我们刚生成的
    # 或用户通过 ``--vlm.api_base`` 传入的；bandit B310 警告
    # 是针对带有 file:/ 方案的任意用户控制 URL，这些无法到达此代码路径。
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:  # noqa: S310  # nosec B310
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def _spawn_inference_server(config: VlmConfig) -> str:
    """生成 ``transformers serve``（或 ``serve_command``），等待直到
    它接受 ``/v1/models``，并注册关闭钩子。

    在后台线程上将服务器的 stdout/stderr 实时流式传输到父终端，
    以便用户可以在模型加载进度和错误发生时看到它们。

    返回 OpenAI 客户端应使用的完整 ``api_base`` URL。
    """
    cmd = config.serve_command
    if not cmd:
        cmd = (
            f"transformers serve {shlex.quote(config.model_id)} "
            f"--port {config.serve_port} --continuous-batching"
        )
    # 将单个服务器绑定到 ``serve_port``（下面 ``api_base`` 所针对的）：
    # 替换字面的 ``{port}`` 占位符，否则追加 ``--port``。
    # 没有这个，带有 ``{port}`` 的 serve_command 会未替换地到达服务器并无法解析。
    cmd = _bind_serve_port(cmd, config.serve_port)
    api_base = f"http://localhost:{config.serve_port}/v1"
    print(f"[server] launching: {cmd}", flush=True)
    proc = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # 监视服务器输出中的 uvicorn 就绪横幅。这比轮询 /v1/models 更可靠，
    # 因为 transformers serve 会在每个模型列表请求时重新扫描其缓存，
    # 这可能超过 urllib 超时并触发无限探测循环。
    ready_event = threading.Event()
    # 参见 _spawn_parallel_inference_servers 了解为什么我们接受这些。
    ready_markers = (
        "Uvicorn running",
        "Application startup complete",
        "Starting vLLM API server",
        "Available routes are",
    )

    def _probe() -> None:
        while not ready_event.is_set() and proc.poll() is None:
            if _server_is_up(api_base):
                print("[server] ready (http probe)", flush=True)
                ready_event.set()
                return
            time.sleep(2)

    threading.Thread(target=_probe, daemon=True).start()

    def _stream_output() -> None:
        # 读取原始块而不是迭代行，以便 tqdm 进度条（使用 \r 覆盖）实时刷新。
        assert proc.stdout is not None
        buf = ""
        prefix_started = False
        while True:
            ch = proc.stdout.read(1)
            if ch == "":
                # 进程退出；刷新任何尾部
                if buf:
                    sys.stdout.write(buf)
                    sys.stdout.flush()
                return
            if not prefix_started:
                sys.stdout.write("[server] ")
                prefix_started = True
            sys.stdout.write(ch)
            sys.stdout.flush()
            buf += ch
            if ch in ("\n", "\r"):
                if any(marker in buf for marker in ready_markers):
                    ready_event.set()
                buf = ""
                prefix_started = False

    threading.Thread(target=_stream_output, daemon=True).start()

    def _shutdown() -> None:
        if proc.poll() is None:
            print(f"[server] stopping pid={proc.pid}", flush=True)
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)

    atexit.register(_shutdown)

    deadline = time.monotonic() + config.serve_ready_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"[server] inference server exited unexpectedly with rc={proc.returncode}. "
                f"See [server] log lines above for the cause."
            )
        if ready_event.wait(timeout=2):
            return api_base
    proc.terminate()
    raise RuntimeError(f"[server] did not become ready within {config.serve_ready_timeout_s}s")


def _to_openai_messages(
    messages: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """将内部消息转换为 OpenAI 聊天格式。

    返回 ``(api_messages, mm_kwargs)``。多模态处理器 kwargs
    （来自 ``video_url`` 块的 ``fps``）被提取出来，以便调用者
    可以通过 ``extra_body.mm_processor_kwargs`` 传递它们，而不是
    在内容块内部传递（transformers serve 会拒绝）。

    文件 URL 视频块被内联为 base64 数据 URL。
    """
    out_messages: list[dict[str, Any]] = []
    mm_kwargs: dict[str, Any] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out_messages.append({"role": message["role"], "content": content})
            continue
        out_blocks: list[dict[str, Any]] = []
        for block in content:
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "text":
                out_blocks.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                out_blocks.append(
                    {"type": "image_url", "image_url": {"url": _pil_to_data_url(block["image"])}}
                )
            elif block_type == "video":
                frames = block.get("video", [])
                for img in frames:
                    out_blocks.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
            elif block_type == "video_url":
                video_url = dict(block["video_url"])
                url = video_url.get("url", "")
                if url.startswith("file://"):
                    video_url["url"] = _file_to_data_url(url[len("file://") :])
                out_blocks.append({"type": "video_url", "video_url": video_url})
                fps = block.get("fps")
                if fps is not None:
                    mm_kwargs["fps"] = fps
            else:
                out_blocks.append(block)
        out_messages.append({"role": message["role"], "content": out_blocks})
    return out_messages, mm_kwargs


def _file_to_data_url(path: str) -> str:
    """读取本地视频文件并返回 base64 ``data:video/mp4`` URL。"""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def _pil_to_data_url(image: Any) -> str:
    """将 PIL.Image 编码为 base64 数据 URL。"""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
