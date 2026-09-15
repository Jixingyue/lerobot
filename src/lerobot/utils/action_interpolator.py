# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""用于更平滑机器人控制的动作插值。

通过在连续动作之间插值，提供可配置的 N 倍控制频率。
配合 RTC 和动作分块策略使用，可减少抖动。
"""

from torch import Tensor


class ActionInterpolator:
    """在连续动作之间插值，实现更平滑的控制。

    当以倍数 N 启用时，通过对上一个动作和当前动作做线性插值，
    为每个策略动作生成 N 个动作。

    multiplier=3 的示例：
        prev_action -> [1/3 插值, 2/3 插值, current_action]

    这相当于成倍提高控制频率，使运动更平滑。

    用法：
        interpolator = ActionInterpolator(multiplier=2)  # 2 倍控制频率

        # 在控制循环中：
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)

        # 录制保持基础 FPS：只有发出策略自身动作的那个
        # tick 才贡献一个数据集帧。
        if interpolator.emitted_policy_action:
            dataset.add_frame(...)
    """

    def __init__(self, multiplier: int = 1):
        """初始化插值器。

        参数：
            multiplier：控制频率倍数（1 = 不插值，2 = 2 倍，3 = 3 倍，以此类推）
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0
        self._emitted_policy_action = False

    @property
    def enabled(self) -> bool:
        """插值是否处于激活状态（multiplier > 1）。"""
        return self.multiplier > 1

    @property
    def emitted_policy_action(self) -> bool:
        """:meth:`get` 上一次返回的动作是否为策略自身的输出。

        :meth:`add` 将策略动作存放在插值缓冲区的最后，因此该值
        恰好在发出 ``buffer[-1]`` 的那个 tick 为 ``True``，在此之前的
        中间 tick 均为 ``False``。策略用它来控制数据集录制：
        这样无论 ``multiplier`` 是多少，帧都按 ``fps`` 落盘，且每一帧保存的
        都是真实策略动作及产生该动作的观测。

        请在 :meth:`get`（或 ``send_next_action``）*之后*读取它——它描述的是
        已经发出的动作，而不是下一次调用将返回的动作。
        :meth:`needs_new_action` 才是在分发*之前*该问的问题；如果改读它，
        录下的会是 ``buffer[0]``，即离上一个动作最近、离当前动作最远的中间插值动作。
        """
        return self._emitted_policy_action

    def reset(self):
        """重置插值状态（在回合之间调用）。"""
        self._prev = None
        self._buffer = []
        self._idx = 0
        self._emitted_policy_action = False

    def needs_new_action(self) -> bool:
        """检查是否需要从队列获取新动作。"""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """添加新动作并计算插值序列。

        参数：
            action：来自策略/队列的新动作张量（已在 CPU 上）。
        """
        if self.multiplier > 1 and self._prev is not None:
            self._buffer = []
            for i in range(1, self.multiplier):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                self._buffer.append(interp)
            # 端点就是策略动作本身，原样追加而不是计算
            # ``prev + 1.0 * (action - prev)``，后者可能差一个 ULP。
            # ``emitted_policy_action`` 承诺录制的帧携带策略自身的输出，
            # 因此必须保证完全一致。
            self._buffer.append(action.clone())
        else:
            # 第一步：还没有上一个动作，因此按基础 FPS 运行，不做插值。
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """获取下一个插值动作。

        返回值：
            下一个动作张量；若缓冲区已耗尽则返回 None。
        """
        if self._idx >= len(self._buffer):
            self._emitted_policy_action = False
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        self._emitted_policy_action = self._idx == len(self._buffer)
        return action
