"""模式A：全景图格式归一化节点（任意主流格式 → 等距柱状 2:1）。"""

import torch

try:
    from . import studio_core as sc
except ImportError:  # 独立脚本直接导入时
    import studio_core as sc


class PanoramaFormatConvertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "输入图片（batch 中的每一张会分别转换）"},
                ),
                "input_format": (
                    [
                        "equirect",
                        "tiny_planet",
                        "dual_fisheye_lr",
                        "dual_fisheye_tb",
                        "single_fisheye",
                        "cylindrical_strip",
                        "wide_photo",
                        "cubemap_3x2",
                    ],
                    {
                        "default": "equirect",
                        "tooltip": "输入格式：equirect=标准2:1等距柱状(直通校验)；"
                        "tiny_planet=小行星；dual_fisheye_lr/tb=双鱼眼原片(左右/上下两个圆)；"
                        "single_fisheye=单鱼眼(圆形/全画幅)；"
                        "cylindrical_strip=手机全景长条(柱面投影)；"
                        "wide_photo=普通宽幅照片/部分全景(直线投影)；"
                        "cubemap_3x2=立方体贴图三行两列排布",
                    },
                ),
                "fisheye_type": (
                    ["equidistant", "equisolid"],
                    {
                        "default": "equidistant",
                        "tooltip": "鱼眼投影类型（仅鱼眼类格式使用）：equidistant=等距(多数360相机)；"
                        "equisolid=等立体角。转换后圆形边缘明显变形就换另一个试试",
                    },
                ),
                "fisheye_fov_deg": (
                    "FLOAT",
                    {
                        "default": 210.0, "min": 100.0, "max": 359.9, "step": 0.5,
                        "tooltip": "鱼眼镜头视场角（仅鱼眼类格式使用）。360相机原片常见 190~235；"
                        "接缝处重影/黑边就微调此值",
                    },
                ),
                "radius_frac": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.5, "max": 1.2, "step": 0.01,
                        "tooltip": "鱼眼圆半径比例，1.0=圆正好贴住图片边缘（仅鱼眼类格式使用）",
                    },
                ),
                "spacing_frac": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.8, "max": 1.2, "step": 0.01,
                        "tooltip": "双鱼眼两个圆心间距比例，1.0=标准左右各占一半/上下各占一半（仅双鱼眼）",
                    },
                ),
                "strip_hfov_deg": (
                    "FLOAT",
                    {
                        "default": 360.0, "min": 10.0, "max": 360.0, "step": 1.0,
                        "tooltip": "输入的水平视场角（仅 cylindrical_strip / wide_photo 使用）。"
                        "手机全景横条一般 360；普通宽照片按实际视角填（如 90）",
                    },
                ),
                "tp_projection": (
                    ["stereographic", "equidistant"],
                    {
                        "default": "stereographic",
                        "tooltip": "小行星投影类型（仅 tiny_planet 使用）：stereographic=相机App/DJI经典；"
                        "equidistant=Photoshop极坐标风格",
                    },
                ),
                "tp_edge_angle_deg": (
                    "FLOAT",
                    {
                        "default": 106.0, "min": 60.0, "max": 179.5, "step": 0.5,
                        "tooltip": "小行星图外边缘相对画面中心的角度（仅 tiny_planet 使用）。"
                        "转换后地平线弯曲或不在竖直正中就每次±5微调",
                    },
                ),
                "tp_radius_frac": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.4, "max": 1.2, "step": 0.01,
                        "tooltip": "小行星圆半径比例（仅 tiny_planet 使用），1.0=行星圆碰到图片左右边缘",
                    },
                ),
                "cubemap_layout": (
                    ["LFR_BUD", "FRB_LUD"],
                    {
                        "default": "LFR_BUD",
                        "tooltip": "3x2 立方体面顺序（仅 cubemap_3x2 使用）：LFR_BUD=上行[左 前 右]下行[后 上 下]"
                        "(Facebook 3x2 常用)；FRB_LUD=上行[前 右 后]下行[左 上 下](py360convert顺序)。"
                        "拼错会得到错乱画面，换另一个即可",
                    },
                ),
                "mirror": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "转换结果左右镜像（招牌文字反写）时打开（鱼眼/小行星类格式）",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 2048, "min": 512, "max": 8192, "step": 64,
                        "tooltip": "输出等距柱状全景宽度（高自动=宽/2）",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "convert"
    CATEGORY = "360panoramastudio"
    DESCRIPTION = (
        "模式A·全景格式归一化：把小行星/双鱼眼/单鱼眼/手机全景长条/宽幅照片/立方体贴图"
        "统一转成标准 2:1 等距柱状全景，可直接接 360 旋转视频等下游节点。"
    )

    def convert(
        self, image, input_format,
        fisheye_type="equidistant", fisheye_fov_deg=210.0,
        radius_frac=1.0, spacing_frac=1.0, strip_hfov_deg=360.0,
        tp_projection="stereographic", tp_edge_angle_deg=106.0, tp_radius_frac=1.0,
        cubemap_layout="LFR_BUD", mirror=False, output_width=2048,
    ):
        out_frames = []
        for i in range(image.shape[0]):
            img = image[i:i + 1]
            if input_format == "equirect":
                b, h, w, _ = img.shape
                if abs(w / h - 2.0) <= 0.1:
                    out = sc.resize_width(img, int(output_width))
                else:  # 宽高比差太多，可能不是标准全景，原样透传
                    out = img
            elif input_format == "tiny_planet":
                out = sc.tinyplanet_to_equirect(
                    img, projection=tp_projection, edge_angle_deg=tp_edge_angle_deg,
                    planet_radius_frac=tp_radius_frac, mirror=mirror,
                    output_width=int(output_width),
                )
            elif input_format == "dual_fisheye_lr":
                out = sc.dual_fisheye_to_equirect(
                    img, layout="lr", proj=fisheye_type, fov_deg=fisheye_fov_deg,
                    radius_frac=radius_frac, spacing_frac=spacing_frac,
                    mirror=mirror, output_width=int(output_width),
                )
            elif input_format == "dual_fisheye_tb":
                out = sc.dual_fisheye_to_equirect(
                    img, layout="tb", proj=fisheye_type, fov_deg=fisheye_fov_deg,
                    radius_frac=radius_frac, spacing_frac=spacing_frac,
                    mirror=mirror, output_width=int(output_width),
                )
            elif input_format == "single_fisheye":
                out = sc.single_fisheye_to_equirect(
                    img, proj=fisheye_type, fov_deg=fisheye_fov_deg,
                    radius_frac=radius_frac, mirror=mirror,
                    output_width=int(output_width),
                )
            elif input_format == "cylindrical_strip":
                out = sc.cylindrical_strip_to_equirect(
                    img, strip_hfov_deg=strip_hfov_deg, output_width=int(output_width),
                )
            elif input_format == "wide_photo":
                out = sc.wide_photo_to_equirect(
                    img, hfov_deg=strip_hfov_deg, output_width=int(output_width),
                )
            elif input_format == "cubemap_3x2":
                out = sc.cubemap32_to_equirect(
                    img, layout=cubemap_layout, output_width=int(output_width),
                )
            else:
                raise ValueError(f"未知格式: {input_format}")
            out_frames.append(out)
        return (torch.cat(out_frames, dim=0),)
