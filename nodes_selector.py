"""懒执行模式选择器：只执行当前选中模式的分支，未选中的分支完全不运行。"""

import torch


class PanoramaModeSelectorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["A · 全景图转换", "B · 照片拼接", "C · AI生成"],
                    {
                        "default": "A · 全景图转换",
                        "tooltip": "选择使用哪条来源分支。懒执行：未选中的分支不会运行，"
                        "不浪费时间和显存",
                    },
                ),
            },
            "optional": {
                "input_a": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "模式A输入：格式转换后的全景图",
                    },
                ),
                "input_b": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "模式B输入：照片拼接出的全景图",
                    },
                ),
                "input_c": (
                    "IMAGE",
                    {
                        "lazy": True,
                        "tooltip": "模式C输入：AI 生成出的全景图",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "pick"
    CATEGORY = "360panoramastudio"
    DESCRIPTION = (
        "360 全景工作室·总开关：A=全景图转换 / B=照片拼接 / C=AI生成。"
        "只有被选中的分支才会执行（懒执行）。"
    )

    def check_lazy_status(self, mode, input_a=None, input_b=None, input_c=None):
        key = mode.strip().split(" ")[0].upper()
        needed = []
        if key == "A" and input_a is None:
            needed.append("input_a")
        if key == "B" and input_b is None:
            needed.append("input_b")
        if key == "C" and input_c is None:
            needed.append("input_c")
        return needed

    def pick(self, mode, input_a=None, input_b=None, input_c=None):
        key = mode.strip().split(" ")[0].upper()
        chosen = {"A": input_a, "B": input_b, "C": input_c}.get(key)
        if chosen is None:
            raise ValueError(
                f"模式「{mode}」的输入为空：请把对应分支的输出接到 input_{key.lower()}")
        return (chosen,)
