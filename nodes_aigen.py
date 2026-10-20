"""模式C：全景 AI 生成器（模块化引擎接口，v2 实现 Qwen-Image 2.1，预留 H3 槽位）。

单一接口：参考照片 + 可选补充描述 + 分辨率 → 2:1 等距柱状全景图。
内部固定全景提示词模板，用户无需写提示词。
"""

import math
import os

import node_helpers
import folder_paths
import comfy.sample
import comfy.sd
import comfy.utils
import torch

try:
    from . import studio_core as sc
except ImportError:  # 独立脚本直接导入时
    import studio_core as sc

# --------------------------------------------------------------- 固定模板

_POSITIVE_TEMPLATE = (
    "Photorealistic 360-degree equirectangular panorama in 2:1 aspect ratio of "
    "the exact same scene as the reference photos: identical room layout, "
    "objects, materials, colors and lighting. Full 360-degree horizontal view, "
    "the left edge continues seamlessly into the right edge. Horizon at the "
    "exact vertical center, ceiling at the top, floor at the bottom. Straight "
    "lines remain straight, no fisheye distortion, no frame borders, no text."
    "{extra}"
)
_NEGATIVE = (
    "text, watermark, logo, signature, blurry, low quality, distorted geometry, "
    "warped straight lines, duplicated furniture, inconsistent lighting, "
    "visible seams, frame border, cropped subject"
)

DIFFUSION_CHOICES = [
    "qwen_image_2.1_int8_convrot (推荐·省显存)",
    "qwen_image_2.1_bf16 (最高质量·显存要求高)",
]
DIFFUSION_FILES = {
    DIFFUSION_CHOICES[0]: "qwen_image_2.1_int8_convrot.safetensors",
    DIFFUSION_CHOICES[1]: "qwen_image_2.1_bf16.safetensors",
}
ENCODER_CHOICES = [
    "qwen3vl_8b_int8_convrot (推荐·省显存)",
    "qwen3vl_8b_bf16 (工作流同款)",
    "qwen3vl_4b_fp8_scaled (最低显存)",
]
ENCODER_FILES = {
    ENCODER_CHOICES[0]: "qwen3vl_8b_int8_convrot.safetensors",
    ENCODER_CHOICES[1]: "qwen3vl_8b_bf16.safetensors",
    ENCODER_CHOICES[2]: "qwen3vl_4b_fp8_scaled.safetensors",
}
RESOLUTIONS = [
    "512 x 256 · 快速测试",
    "1024 x 512 · 标准",
    "2048 x 1024 · 推荐",
    "4096 x 2048 · 生成2048x1024后2x放大",
]
RESOLUTION_MAP = {
    RESOLUTIONS[0]: (512, 256),
    RESOLUTIONS[1]: (1024, 512),
    RESOLUTIONS[2]: (2048, 1024),
    RESOLUTIONS[3]: (2048, 1024),  # 生成尺寸，最后放大
}

_MODEL_CACHE = {}


def _full_path(folder, name):
    p = folder_paths.get_full_path(folder, name)
    if not p:
        for alt in ("text_encoders", "clip", "diffusion_models", "vae"):
            p = folder_paths.get_full_path(alt, name)
            if p:
                break
    if not p:
        raise FileNotFoundError(f"找不到模型文件 {name}（models 目录里没有）")
    return p


def _load_models(diffusion_name, encoder_name):
    key = (diffusion_name, encoder_name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    model = comfy.sd.load_diffusion_model(
        _full_path("diffusion_models", diffusion_name))
    if model is None:
        raise RuntimeError(f"加载扩散模型失败: {diffusion_name}")
    clip = comfy.sd.load_clip(
        ckpt_paths=[_full_path("text_encoders", encoder_name)],
        embedding_directory=folder_paths.get_folder_paths("text_encoders")[0],
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE)
    if clip is None:
        raise RuntimeError(f"加载文本编码器失败: {encoder_name}")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(
        _full_path("vae", "qwen_image_2.1_vae_bf16.safetensors")))
    _MODEL_CACHE.clear()  # 16GB 显存机器只保留一套，避免叠加
    _MODEL_CACHE[key] = (model, clip, vae)
    return _MODEL_CACHE[key]


def _encode_refs(clip, vae, images, positive_text, negative_text,
                 ref_res=1024):
    """参考图双路编码（VL 视觉 token + VAE reference_latents），
    与 comfy_extras/nodes_qwen.py TextEncodeQwenImage21 同一逻辑。"""
    images_vl, ref_latents = [], []
    image_prompt = ""
    for i, image in enumerate(images):
        samples = image.movedim(-1, 1)
        total = 384 * 384
        scale_by = (total / (samples.shape[2] * samples.shape[3])) ** 0.5
        s_vl = comfy.utils.common_upscale(
            samples, round(samples.shape[3] * scale_by),
            round(samples.shape[2] * scale_by), "lanczos", "disabled")
        images_vl.append(s_vl.movedim(1, -1))
        if vae is not None:
            ratio = samples.shape[3] / samples.shape[2]
            wv = round(math.sqrt(ref_res * ref_res * ratio) / 32) * 32
            hv = round(math.sqrt(ref_res * ref_res / ratio) / 32) * 32
            s_vae = comfy.utils.common_upscale(
                samples, max(32, wv), max(32, hv), "lanczos", "disabled")
            ref_latents.append(vae.encode(s_vae.movedim(1, -1)[:, :, :, :3]))
        image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

    def encode(text):
        tokens = clip.tokenize(image_prompt + text, images=images_vl,
                               keep_vision=len(ref_latents) == 0,
                               prevent_empty_text=True)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_latents:
            cond = node_helpers.conditioning_set_values(
                cond, {"reference_latents": ref_latents}, append=True)
        return cond

    return encode(positive_text), encode(negative_text), ref_latents


class PanoramaAIGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "同一场景的参考照片（batch，可用 LoadImage 或 "
                     "VHS Load Images (Path) 提供），自动取前若干张做参考"},
                ),
                "engine": (
                    ["Qwen-Image 2.1", "MiniMax-H3（三期接入）"],
                    {
                        "default": "Qwen-Image 2.1",
                        "tooltip": "生成引擎。模块化接口：以后新增引擎只需扩展此下拉，"
                        "节点接口和下游工作流不变",
                    },
                ),
                "diffusion_model": (
                    DIFFUSION_CHOICES,
                    {"default": DIFFUSION_CHOICES[0],
                     "tooltip": "Qwen-Image 2.1 扩散模型权重（本地已下载）"},
                ),
                "text_encoder": (
                    ENCODER_CHOICES,
                    {"default": ENCODER_CHOICES[0],
                     "tooltip": "Qwen3-VL 文本编码器（同时负责读懂参考图和提示词，"
                     "管线必需件）。int8 版省显存，效果等同"},
                ),
                "resolution": (
                    RESOLUTIONS,
                    {
                        "default": RESOLUTIONS[2],
                        "tooltip": "输出全景分辨率。4096x2048 档为生成 2048x1024 后 "
                        "2x 放大（建议再接放大模型精修）",
                    },
                ),
                "extra_prompt": (
                    "STRING",
                    {
                        "multiline": True, "default": "",
                        "tooltip": "可选补充描述（时间/天气/风格等，如 “night scene, "
                        "warm lights”）。留空则纯还原参考场景",
                    },
                ),
                "steps": (
                    "INT",
                    {"default": 25, "min": 4, "max": 50, "step": 1,
                     "tooltip": "采样步数（工作流同款默认 25，低显存可降）"},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                     "control_after_generate": True,
                     "tooltip": "随机种子"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "generate"
    CATEGORY = "360panoramastudio"
    DESCRIPTION = (
        "模式C·全景AI生成：上传同一场景的照片，自动生成 2:1 等距柱状全景图"
        "（提示词模板已内置，可加补充描述）。引擎可插拔，二期为 Qwen-Image 2.1。"
        "注意：AI 生成不保证严格投影几何，左右接缝可能有瑕疵，适合创意场景。"
    )

    def generate(self, image, engine, diffusion_model, text_encoder,
                 resolution, extra_prompt, steps, seed):
        if engine.startswith("MiniMax"):
            raise NotImplementedError(
                "MiniMax-H3 引擎属三期规划，当前版本尚未接入。"
                "请选择 Qwen-Image 2.1 引擎。")
        out_w, out_h = RESOLUTION_MAP[resolution]
        n_refs = min(int(image.shape[0]), 8)
        refs = [image[i:i + 1, :, :, :3] for i in range(n_refs)]

        model, clip, vae = _load_models(
            DIFFUSION_FILES[diffusion_model], ENCODER_FILES[text_encoder])

        pos_text = _POSITIVE_TEMPLATE.format(
            extra=f" {extra_prompt.strip()}." if extra_prompt.strip() else "")
        positive, negative, ref_latents = _encode_refs(
            clip, vae, refs, pos_text, _NEGATIVE)

        channels = ref_latents[0].shape[1] if ref_latents else 16
        latent_image = torch.zeros([1, channels, out_h // 8, out_w // 8])
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        noise = comfy.sample.prepare_noise(latent_image, seed)

        out_latent = comfy.sample.sample(
            model, noise, steps, 1.0, "euler", "simple",
            positive, negative, latent_image, denoise=1.0, seed=seed)
        out = vae.decode(out_latent)
        if len(out.shape) == 5:
            out = out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])
        out = out.clamp(0.0, 1.0)
        if resolution.startswith("4096"):
            out = comfy.utils.common_upscale(
                out.movedim(-1, 1), out_w, out_h, "lanczos", "disabled"
            ).movedim(1, -1).clamp(0.0, 1.0)
        return (out,)
