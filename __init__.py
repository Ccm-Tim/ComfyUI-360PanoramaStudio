"""ComfyUI-360PanoramaStudio

360 全景工作室：把「多来源全景素材」统一成标准 2:1 等距柱状全景，
再接 360 旋转视频等下游节点。

- 模式A 全景格式转换：小行星 / 双鱼眼 / 单鱼眼 / 手机全景长条 / 宽幅照片 / 立方体贴图 → equirect
- 模式B 照片拼全景：OpenCV / Hugin 双引擎，拍摄方式开关（单排环绕/多排网格/散拍/自动）+ 室内外
- 模式C 全景AI生成：Qwen-Image 2.1 参考生图（引擎可插拔，预留 H3 槽位）
- 懒执行模式选择器：多分支只执行选中的那一条
"""

import torch  # noqa: F401  提前加载，保证 custom_nodes 环境一致

from . import studio_core  # noqa: F401
from .nodes_aigen import PanoramaAIGeneratorNode
from .nodes_convert import PanoramaFormatConvertNode
from .nodes_selector import PanoramaModeSelectorNode
from .nodes_stitch import PhotoPanoramaStitchNode

NODE_CLASS_MAPPINGS = {
    "PanoramaFormatConvertNode": PanoramaFormatConvertNode,
    "PhotoPanoramaStitchNode": PhotoPanoramaStitchNode,
    "PanoramaAIGeneratorNode": PanoramaAIGeneratorNode,
    "PanoramaModeSelectorNode": PanoramaModeSelectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PanoramaFormatConvertNode": "全景格式转换 (任意格式→2:1)",
    "PhotoPanoramaStitchNode": "照片拼全景 (多图→全景图)",
    "PanoramaAIGeneratorNode": "全景AI生成 (照片参考→全景图)",
    "PanoramaModeSelectorNode": "全景模式选择器 (懒执行开关)",
}

__version__ = "0.2.0"
