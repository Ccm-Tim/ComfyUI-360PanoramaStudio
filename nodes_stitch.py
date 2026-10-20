"""模式B：多张同场景照片 → 全景图（OpenCV / Hugin 双引擎，含拍摄方式开关）。"""

import tempfile

import numpy as np
import torch

import cv2

try:
    from . import studio_core as sc
    from . import hugin_engine
except ImportError:  # 独立脚本直接导入时
    import studio_core as sc
    import hugin_engine

_MODE_MAP = {
    "panorama": cv2.Stitcher_PANORAMA,
    "single_row": cv2.Stitcher_PANORAMA,
    "multi_row": cv2.Stitcher_PANORAMA,
    "free_scans": cv2.Stitcher_SCANS,
}
_STATUS_MSG = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "需要更多照片或相邻照片重叠不足（建议相邻两张重叠 30%~50%）",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "特征匹配失败：照片间重叠太少或场景纹理太弱（白墙/天空）",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相机参数估计失败：照片可能不是原地旋转拍摄的（平移视差过大）",
}


def _to_bgr_list(images: torch.Tensor, max_images: int):
    frames = []
    n = images.shape[0]
    if max_images and max_images > 0:
        n = min(n, max_images)
    for i in range(n):
        t = images[i].clamp(0.0, 1.0).cpu().numpy()
        frames.append((t * 255.0).round().astype(np.uint8)[..., ::-1])  # RGB->BGR
    return frames


def _balance_exposure(frames):
    """室内亮度均衡：把每张图的亮度中位数增益到第一张的水平（全局增益）。
    上下/前后拍摄时光差大（窗光、灯光），先拉齐能明显减少接缝。"""
    smalls = [
        cv2.resize(f, (160, max(1, round(160 * f.shape[0] / f.shape[1]))))
        for f in frames
    ]
    medians = [float(np.median(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)))
               for s in smalls]
    ref = medians[0]
    if ref < 1.0:
        return frames
    out = []
    for f, m in zip(frames, medians):
        if m < 1.0:
            out.append(f)
            continue
        gain = float(np.clip(ref / m, 0.5, 2.0))
        out.append(np.clip(f.astype(np.float32) * gain, 0, 255).astype(np.uint8))
    return out


def _stitch_once(frames, mode, confidence):
    stitcher = cv2.Stitcher_create(mode)
    stitcher.setPanoConfidenceThresh(float(confidence))
    return stitcher.stitch(frames)


class PhotoPanoramaStitchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "同一场景的多张照片（batch）。可用 VHS Load Images (Path) 从目录加载"},
                ),
                "shooting_mode": (
                    ["auto", "single_row", "multi_row", "free_scans"],
                    {
                        "default": "auto",
                        "tooltip": "拍摄方式：auto=自动(先单排环绕策略,失败换散拍策略)；"
                        "single_row=原地转一圈的单排照片(最快最稳)；"
                        "multi_row=上下也拍了多排(网格)；"
                        "free_scans=随手拍的散片(容错高,仅适合平面场景)",
                    },
                ),
                "scene": (
                    ["outdoor", "indoor"],
                    {
                        "default": "outdoor",
                        "tooltip": "场景类型：indoor=室内(拼接前先做亮度均衡，应对窗光/灯光明暗差)；"
                        "outdoor=室外(不做预处理)",
                    },
                ),
                "confidence": (
                    "FLOAT",
                    {
                        "default": 1.0, "min": 0.3, "max": 2.0, "step": 0.05,
                        "tooltip": "拼接匹配置信度阈值，一般保持 1.0。若出现拼错/重影可提高到 1.2~1.5 拒绝坏匹配；"
                        "调低反而容易引入错误配对导致失败",
                    },
                ),
                "max_images": (
                    "INT",
                    {
                        "default": 0, "min": 0, "max": 99, "step": 1,
                        "tooltip": "最多使用前几张照片，0=全部使用。照片很多时可限制以提速",
                    },
                ),
                "pad_to_equirect": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "开启=上下补天补地到标准 2:1（边缘延展填充），可直接接旋转视频节点",
                    },
                ),
                "output_width": (
                    "INT",
                    {
                        "default": 4096, "min": 1024, "max": 8192, "step": 64,
                        "tooltip": "输出全景图宽度",
                    },
                ),
                "engine": (
                    ["auto", "opencv", "hugin"],
                    {
                        "default": "auto",
                        "tooltip": "拼接引擎：auto=有Hugin用Hugin，没有用OpenCV；"
                        "opencv=快(几秒)，输出上下由补天补地填充；"
                        "hugin=慢(约1分钟)但全局优化+回环闭合，输出真 2:1 无缝全景"
                        "(需安装 Hugin：winget install Hugin.Hugin)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("panorama", "info")
    FUNCTION = "stitch"
    CATEGORY = "360panoramastudio"
    DESCRIPTION = (
        "模式B·照片拼全景：把同一场景的多张重叠照片拼成一张全景图。"
        "双引擎：OpenCV（快）与 Hugin（真 360 无缝回环，需安装 Hugin）。"
        "按拍摄方式选择策略（单排环绕/多排网格/自由散拍/自动），室内外可选。"
        "全景质量受拍摄影响：请原地转动拍摄、相邻照片重叠约三分之一。"
    )

    def stitch(self, images, shooting_mode, scene, confidence, max_images,
               pad_to_equirect, output_width, engine="auto"):
        frames = _to_bgr_list(images, max_images)
        if len(frames) == 0:
            raise ValueError("输入为空：没有收到任何照片")

        hugin_bin = hugin_engine.find_hugin_bin()
        use_hugin = (engine == "hugin") or (engine == "auto" and hugin_bin)
        if engine == "hugin" and not hugin_bin:
            raise RuntimeError(
                "engine=hugin 但未找到 Hugin。请先安装："
                "winget install Hugin.Hugin（免费开源），或设置 HUGIN_PATH。")

        engine_used = "hugin" if use_hugin else "opencv"
        if len(frames) == 1:
            pano = frames[0]
            info = "只有 1 张照片，跳过拼接直接输出。"
        elif use_hugin:
            balanced = _balance_exposure(frames) if scene == "indoor" else frames
            with tempfile.TemporaryDirectory(prefix="hugin_stitch_") as td:
                pano = hugin_engine.stitch_hugin(
                    balanced, td, equirect_width=int(output_width),
                    multirow=(shooting_mode == "multi_row"))
            info = f"拼接成功(Hugin引擎)：输入 {len(frames)} 张照片，全局优化+回环闭合，输出 {pano.shape[1]}x{pano.shape[0]} 真2:1 equirect"
            pano_t = torch.from_numpy(pano[..., ::-1].astype(np.float32) / 255.0
                                      ).unsqueeze(0)
            if int(pano_t.shape[2]) != int(output_width):
                pano_t = sc.resize_width(pano_t, int(output_width))
            return (pano_t, info)
        else:
            indoor = (scene == "indoor")
            # 环绕类照片太多时（>8）cv2 匹配容易被远距误匹配搞垮，自动均匀抽样到 8 张
            if shooting_mode in ("auto", "single_row", "multi_row") and len(frames) > 8:
                idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
                idx = sorted(set(idx.tolist()))
                frames = [frames[i] for i in idx]
            tried = []
            if shooting_mode == "free_scans":
                attempts = [("free_scans", confidence)]
            else:
                attempts = [("panorama", confidence)]
                if abs(float(confidence) - 1.0) > 1e-6:
                    attempts.append(("panorama", 1.0))  # 用户调低了阈值反而易失败，回退默认重试
            pano = None
            stitch_frames = _balance_exposure(frames) if indoor else frames
            for mode_key, conf in attempts:
                status, result = _stitch_once(
                    stitch_frames, _MODE_MAP[mode_key], conf)
                label = ("自由散拍" if mode_key == "free_scans" else "环形拼接")
                tried.append(label)
                if status == cv2.Stitcher_OK:
                    pano = result
                    break
                reason = _STATUS_MSG.get(status, f"未知错误码 {status}")
                tried[-1] = f"{label}(置信度{conf})失败：{reason}"
            if pano is None:
                raise RuntimeError(
                    "拼接失败：" + "；".join(tried)
                    + "。建议：①补拍使相邻照片重叠≥1/3；②原地旋转拍摄减少视差；"
                    "③照片特别多时先用 max_images 限制数量；"
                    "④白墙等弱纹理场景尽量把有物体的画面拍进重叠区")

        pano_t = torch.from_numpy(pano[..., ::-1].astype(np.float32) / 255.0
                                  ).unsqueeze(0)  # BGR->RGB, [1,H,W,C]
        pano_t = sc.resize_width(pano_t, int(output_width))
        if pad_to_equirect:
            pano_t = sc.pad_to_equirect(pano_t)
        info = (
            f"拼接成功({engine_used}引擎)：输入 {len(frames)} 张照片，"
            f"策略 {'→'.join(tried)}，输出 {pano_t.shape[2]}x{pano_t.shape[1]}"
            + ("（已补天补地到2:1）" if pad_to_equirect else "")
        )
        return (pano_t, info)
