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

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any

from lerobot.utils.constants import LANGUAGE_PERSISTENT, MESSAGES_RENDERED
from lerobot.utils.utils import unwrap_scalar

from .language import column_for_style
from .recipe import DEFAULT_BINDINGS, TrainingRecipe, render_message_turns

LanguageRow = dict[str, Any]
RenderedMessages = dict[str, list[Any]]

_RESOLVER_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$")


def active_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """返回在时刻 ``t`` 处于活动状态的指定 ``style`` 的 persistent 行。

    当某条 persistent 行自身的 ``timestamp`` 是给定
    ``style``/``role``/``tool_name``/``camera`` 选择器下最近的
    一个 ``<= t`` 的时间戳时，该行在 ``t`` 时刻“处于活动状态”。
    仅适用于 persistent 风格。
    """
    _validate_persistent_resolver("active_at", style)
    matches = [
        row
        for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
        if _timestamp(row) <= t
    ]
    if not matches:
        return None
    latest_ts = max(_timestamp(row) for row in matches)
    return _select_one(
        [row for row in matches if _timestamp(row) == latest_ts],
        style=style,
        role=role,
        tool_name=tool_name,
        camera=camera,
    )


EMITTED_AT_TOLERANCE_S = 0.1
"""``emitted_at`` 中将 persistent 行与帧时间戳匹配时使用的半窗口大小。
Persistent 时间戳来自 parquet（float32），而 ``t`` 同样是来自
parquet 的 float32，因此在理想的热路径上精确匹配就已足够——
但任何通过算术方式推导 ``t`` 的调用方（例如
``frame_idx / fps``）都会破坏位级相等。0.1 秒的容差可以覆盖
常见的算术漂移，同时在典型控制频率（30–100 Hz）下不会放入
明显相隔很远的帧。这确实意味着：同一选择器下、相隔不足 0.1 秒
发出的两条 persistent 行无法被 ``emitted_at`` 区分——这是可以
接受的，因为 persistent 标注（子任务/计划/记忆的切换）发生在
人类动作的时间尺度上，而非相机帧率的时间尺度上。"""


def emitted_at(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    style: str | None = None,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """返回恰好在时刻 ``t`` 发出的指定 ``style`` 的行。

    对于 persistent 风格，本函数匹配自身 ``timestamp`` 与 ``t``
    相差不超过 ``EMITTED_AT_TOLERANCE_S`` 的 persistent 行（为何
    使用容差而非位级相等，参见该常量的说明）。对于 event 风格，
    假定 ``events`` 列表来自帧 ``t`` 对应的数据集行
    （event 行本身不带时间戳），因此所有匹配的 event 行都被视为
    在 ``t`` 时刻发出。``camera`` 按行的 ``camera`` 字段过滤——
    当多个依赖视角的行在不同相机间共享相同的 ``(t, role)`` 时，
    需要用它来消除歧义。
    """
    if column_for_style(style) == LANGUAGE_PERSISTENT:
        matches = [
            row
            for row in _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera)
            if abs(_timestamp(row) - t) <= EMITTED_AT_TOLERANCE_S
        ]
    else:
        matches = _matching_rows(events, style=style, role=role, tool_name=tool_name, camera=camera)
    return _select_one(matches, style=style, role=role, tool_name=tool_name, camera=camera)


def nth_prev(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """返回在 ``t`` 之前 ``offset`` 步处于活动状态的 persistent 行。

    沿按时间排序的指定 ``style`` 的 persistent 行向后遍历
    （可用 ``role``/``tool_name``/``camera`` 过滤），返回相对
    ``t`` 时刻活动行之前 ``offset`` 个位置的那一行。仅适用于
    persistent 风格。
    """
    return _nth_relative("nth_prev", t, persistent, style, -offset, role, tool_name, camera)


def nth_next(
    t: float,
    *,
    persistent: Sequence[LanguageRow],
    style: str | None = None,
    offset: int = 1,
    role: str | None = None,
    tool_name: str | None = None,
    camera: str | None = None,
) -> LanguageRow | None:
    """返回在 ``t`` 之后 ``offset`` 步变为活动状态的 persistent 行。

    沿按时间排序的指定 ``style`` 的 persistent 行向前遍历
    （可用 ``role``/``tool_name``/``camera`` 过滤），返回相对
    ``t`` 时刻活动行之后 ``offset`` 个位置的那一行。仅适用于
    persistent 风格。
    """
    return _nth_relative("nth_next", t, persistent, style, offset, role, tool_name, camera)


def render_sample(
    *,
    recipe: TrainingRecipe,
    persistent: Sequence[LanguageRow] | None,
    events: Sequence[LanguageRow] | None,
    t: float,
    sample_idx: int,
    task: str | None = None,
    dataset_ctx: Any | None = None,
) -> RenderedMessages | None:
    """为一个数据集样本渲染配方（recipe）定义的消息和监督信号。

    在帧时间戳 ``t`` 处，针对 ``persistent`` 和 ``events``
    解析绑定。混合配方（blend recipe）首先路由匹配的稀疏 VQA 标注，
    然后对剩余样本使用确定性的加权选择。当所选配方没有为该样本
    提供文本或底层动作监督时，返回 ``None``。
    """
    persistent_rows = _normalize_rows(persistent or [])
    event_rows = _normalize_rows(events or [])

    # 在加权选择之前，先将稀疏的 VQA 帧路由到匹配的视角专属组件。
    # 这样可以避免丢弃带标注的帧，或选中没有标注的 VQA。
    if recipe.blend is not None:
        vqa_rendered = _render_vqa_if_present(
            recipe,
            persistent=persistent_rows,
            events=event_rows,
            t=t,
            sample_idx=sample_idx,
            task=task,
            dataset_ctx=dataset_ctx,
        )
        if vqa_rendered is not None:
            return vqa_rendered

    selected_recipe = _select_recipe(recipe, sample_idx)
    if selected_recipe is None:
        return None
    bindings = _resolve_bindings(
        selected_recipe,
        persistent=persistent_rows,
        events=event_rows,
        t=t,
        sample_idx=sample_idx,
        task=task,
        dataset_ctx=dataset_ctx,
    )
    return _render_message_recipe(selected_recipe, bindings)


def _render_vqa_if_present(
    recipe: TrainingRecipe,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
    sample_idx: int,
    task: str | None,
    dataset_ctx: Any | None,
) -> RenderedMessages | None:
    """渲染一个匹配的 VQA 组件，若没有则返回 ``None`` 以走正常选择流程。

    多个匹配的视角会按相对权重被确定性地选中。
    """
    if recipe.blend is None:
        return None
    renderable: list[tuple[float, RenderedMessages]] = []
    for component in recipe.blend.values():
        if component.route != "vqa":
            continue
        bindings = _resolve_bindings(
            component,
            persistent=persistent,
            events=events,
            t=t,
            sample_idx=sample_idx,
            task=task,
            dataset_ctx=dataset_ctx,
        )
        rendered = _render_message_recipe(component, bindings)
        if rendered is not None:
            if component.weight is None:
                raise ValueError("Routed VQA blend components must define a weight.")
            renderable.append((component.weight, rendered))

    if not renderable:
        return None
    if len(renderable) == 1:
        return renderable[0][1]

    # 在匹配的相机之间，按已校验为正的相对权重进行选择。
    total = sum(weight for weight, _ in renderable)
    digest = hashlib.blake2b(f"vqa:{sample_idx}".encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / 2**64 * total
    cumulative = 0.0
    for weight, rendered in renderable:
        cumulative += weight
        if draw < cumulative:
            return rendered
    return renderable[-1][1]


def _select_recipe(recipe: TrainingRecipe, sample_idx: int) -> TrainingRecipe | None:
    """为普通样本挑选一个确定性的、非路由型的组件。"""
    if recipe.blend is None:
        return recipe

    components = [component for component in recipe.blend.values() if component.route is None]
    total_weight = sum(component.weight or 0.0 for component in components)
    if total_weight <= 0:
        return None

    digest = hashlib.blake2b(str(sample_idx).encode(), digest_size=8).digest()
    draw = int.from_bytes(digest, "big") / 2**64 * total_weight
    cumulative = 0.0
    last_component: TrainingRecipe | None = None
    for component in components:
        last_component = component
        cumulative += component.weight or 0.0
        if draw < cumulative:
            return component
    if last_component is None:
        return None
    return last_component


def _resolve_bindings(
    recipe: TrainingRecipe,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
    sample_idx: int,
    task: str | None,
    dataset_ctx: Any | None,
) -> dict[str, LanguageRow | str | None]:
    """在时刻 ``t`` 解析 ``recipe`` 中的每个绑定（以及 ``task``）。"""
    bindings: dict[str, LanguageRow | str | None] = {
        "task": _resolve_task(task, dataset_ctx, persistent=persistent, sample_idx=sample_idx),
    }
    declared = recipe.bindings or {}
    specs = {**DEFAULT_BINDINGS, **declared}
    # 只解析配方实际使用的绑定：未被引用的默认绑定可能
    # 无法解析，例如多相机帧上无相机的 ``vqa`` 默认绑定。
    needed = recipe.referenced_binding_names() | set(declared)
    for name, spec in specs.items():
        if name not in needed:
            continue
        bindings[name] = _resolve_spec(spec, persistent=persistent, events=events, t=t)
    return bindings


def _resolve_task(
    task: str | None,
    dataset_ctx: Any | None,
    *,
    persistent: Sequence[LanguageRow] = (),
    sample_idx: int = 0,
) -> str | None:
    """返回 ``sample_idx`` 对应的任务字符串。

    解析顺序：

    1. 显式的 ``task`` 覆盖（调用方提供）优先级最高。
    2. 如果 ``persistent`` 中包含风格为 ``task_aug``（role=user）的
       行，则按 ``sample_idx`` 确定性地挑选一行，使一个 episode
       的每一帧在一个 epoch 内轮换使用所有可用的改写版本。
       这实现了 Xiao 2022 / CAST 风格的任务提示多样性，而无需
       修改 ``meta/tasks.parquet``，也不强制配方显式启用：
       ``${task}`` 在存在改写版本时会自动选用改写版本，否则
       回退到规范任务。需要字面规范任务的配方可以覆盖该绑定。
    3. 否则从 ``dataset_ctx`` 读取规范任务（其底层由
       ``meta/tasks.parquet`` 支持）。
    """
    if task is not None:
        return task

    aug_rows = [r for r in persistent if r.get("style") == "task_aug" and r.get("role") == "user"]
    if aug_rows:
        # 基于 blake2b、以 sample_idx 为键进行确定性挑选，
        # 从而保证轮换在多次运行间可复现（Python 内置的
        # ``hash`` 会随进程随机化）。
        digest = hashlib.blake2b(f"task_aug:{sample_idx}".encode(), digest_size=8).digest()
        idx = int.from_bytes(digest, "big") % len(aug_rows)
        chosen = aug_rows[idx].get("content")
        if chosen:
            return str(chosen)

    if dataset_ctx is None:
        return None
    if isinstance(dataset_ctx, dict):
        return dataset_ctx.get("task")
    return getattr(dataset_ctx, "task", None)


def _resolve_spec(
    spec: str,
    *,
    persistent: Sequence[LanguageRow],
    events: Sequence[LanguageRow],
    t: float,
) -> LanguageRow | None:
    """解析单个绑定的解析器表达式，并分派到对应的函数。"""
    match = _RESOLVER_RE.match(spec.strip())
    if match is None:
        raise ValueError(f"Invalid resolver expression: {spec!r}")
    name = match.group("name")
    kwargs = _parse_resolver_args(match.group("args"))
    kwargs.pop("t_arg", None)

    if name == "emitted_at":
        return emitted_at(t, persistent=persistent, events=events, **kwargs)
    if name == "active_at":
        return active_at(t, persistent=persistent, **kwargs)
    if name == "nth_prev":
        return nth_prev(t, persistent=persistent, **kwargs)
    if name == "nth_next":
        return nth_next(t, persistent=persistent, **kwargs)
    raise ValueError(f"Unknown language resolver: {name!r}")


def _parse_resolver_args(args: str) -> dict[str, Any]:
    """将以逗号分隔的解析器参数列表解析为 kwargs 字典。"""
    kwargs: dict[str, Any] = {}
    if not args.strip():
        return kwargs

    parts = [part.strip() for part in args.split(",") if part.strip()]
    for part in parts:
        if part == "t":
            kwargs["t_arg"] = True
            continue
        if "=" not in part:
            raise ValueError(f"Invalid resolver argument: {part!r}")
        key, value = (item.strip() for item in part.split("=", 1))
        if key == "offset":
            kwargs[key] = int(value)
        else:
            kwargs[key] = value.strip("\"'")
    return kwargs


def _render_message_recipe(
    recipe: TrainingRecipe,
    bindings: dict[str, LanguageRow | str | None],
) -> RenderedMessages | None:
    """使用 ``bindings`` 将 ``recipe.messages`` 展开为渲染后的聊天消息。"""
    if recipe.messages is None:
        raise ValueError("Cannot render a blend recipe as a message recipe.")
    rendered = render_message_turns(recipe.messages, bindings)

    # 保留同时具有文本目标或底层动作监督的样本。
    has_low_level = any(stream == "low_level" for stream in rendered["message_streams"])
    if not rendered["target_message_indices"] and not has_low_level:
        return None

    _validate_rendered(rendered)
    return rendered


def _validate_rendered(rendered: RenderedMessages) -> None:
    """对渲染输出做健全性检查，确保流与目标对齐。"""
    messages = rendered[MESSAGES_RENDERED]
    streams = rendered["message_streams"]
    target_indices = rendered["target_message_indices"]

    if len(streams) != len(messages):
        raise ValueError("message_streams must be aligned with messages.")
    # 要求存在文本或底层动作监督。
    if not target_indices and not any(s == "low_level" for s in streams):
        raise ValueError("Rendered samples must contain a target message or a low_level-stream message.")
    for idx in target_indices:
        if idx < 0 or idx >= len(messages):
            raise ValueError(f"Target message index {idx} is out of bounds.")


def _nth_relative(
    name: str,
    t: float,
    persistent: Sequence[LanguageRow],
    style: str | None,
    offset: int,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> LanguageRow | None:
    """``nth_prev`` / ``nth_next`` 的共用实现，``offset`` 带符号。"""
    _validate_persistent_resolver(name, style)
    if abs(offset) < 1:
        raise ValueError(f"{name} offset must be non-zero.")

    rows = sorted(
        _matching_rows(persistent, style=style, role=role, tool_name=tool_name, camera=camera),
        key=_row_sort_key,
    )
    if not rows:
        return None

    anchor_idx = None
    for idx, row in enumerate(rows):
        if _timestamp(row) <= t:
            anchor_idx = idx
        else:
            break

    target_idx = (offset - 1 if offset > 0 else None) if anchor_idx is None else anchor_idx + offset

    if target_idx is None or target_idx < 0 or target_idx >= len(rows):
        return None
    return rows[target_idx]


def _validate_persistent_resolver(name: str, style: str | None) -> None:
    """拒绝在 persistent 解析器中缺失 ``style`` 或使用仅 event 风格的调用。"""
    if style is None:
        raise ValueError(f"{name} requires a persistent style.")
    if column_for_style(style) != LANGUAGE_PERSISTENT:
        raise ValueError(f"{name} cannot be used with event-only style {style!r}.")


def _matching_rows(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> list[LanguageRow]:
    """按可选的 ``style``/``role``/``tool_name``/``camera`` 选择器过滤并返回 ``rows``。"""
    return [
        row
        for row in rows
        if (style is None or row.get("style") == style)
        and (role is None or row.get("role") == role)
        and (tool_name is None or _row_has_tool_name(row, tool_name))
        and (camera is None or row.get("camera") == camera)
    ]


def _select_one(
    rows: Sequence[LanguageRow],
    *,
    style: str | None,
    role: str | None,
    tool_name: str | None,
    camera: str | None,
) -> LanguageRow | None:
    """返回唯一的匹配行；若解析结果有歧义则抛出异常。

    出现多个匹配时总会抛出异常——即使调用方已经传入了
    部分选择器——因为剩余的歧义意味着数据中存在若干行在
    解析器看来完全相同，调用方需要明确指定其中某一行
    （例如为跨相机共享的 VQA 行添加 ``camera=...``）。
    """
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous resolver for style={style!r} role={role!r} "
            f"tool_name={tool_name!r} camera={camera!r}: {len(rows)} matching rows. "
            f"Add a selector that distinguishes them."
        )
    return rows[0]


def _row_sort_key(row: LanguageRow) -> tuple[float, str, str]:
    """同时适用于 persistent 行和 event 行的稳定排序键。

    Event 行没有 ``timestamp``（它隐含在帧中），因此默认取
    ``0.0``——在单个帧内，所有 event 行都位于同一个排序
    桶中，并通过 ``(style, role)`` 来打破并列。
    """
    timestamp = row.get("timestamp")
    ts = float(unwrap_scalar(timestamp)) if timestamp is not None else 0.0
    return (ts, row.get("style") or "", row.get("role") or "")


def _timestamp(row: LanguageRow) -> float:
    """将行的 ``timestamp`` 提取为 Python float（解开 numpy 标量包装）。"""
    return float(unwrap_scalar(row["timestamp"]))


def _row_has_tool_name(row: LanguageRow, tool_name: str) -> bool:
    """当该行的任一工具调用调用了 ``tool_name`` 时返回 ``True``。"""
    for tool_call in row.get("tool_calls") or []:
        if isinstance(tool_call, str):
            continue
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict) and function.get("name") == tool_name:
            return True
    return False


def _normalize_rows(rows: Sequence[Any]) -> list[LanguageRow]:
    """将 pyarrow 标量/映射转换为一个全新的普通 dict 行列表。"""
    normalized = []
    for row in rows:
        if row is None:
            continue
        if hasattr(row, "as_py"):
            row = row.as_py()
        if not isinstance(row, dict):
            raise TypeError(f"Language rows must be dictionaries, got {type(row).__name__}.")
        normalized.append(dict(row))
    return normalized
