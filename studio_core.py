"""Core reprojection math for ComfyUI-360PanoramaStudio.

所有转换都是「目标图每个像素 → 源图坐标」的反向映射 + grid_sample，
与 ComfyUI-TinyPlanet2Equirect 保持同一套管线风格（bilinear + border 填充，
越界区域自动用最外圈颜色延展，形成"补天补地"效果）。

世界坐标约定：x=右, y=上(天), z=前。等距柱状(ERP)输出：
    lon ∈ [-π, π) 从左到右，lat ∈ [π/2, -π/2) 从上到下。
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- utilities

def lonlat_grids(out_w: int, out_h: int, device):
    lon = (
        torch.arange(out_w, device=device, dtype=torch.float32) + 0.5
    ) / out_w * 2 * math.pi - math.pi
    lat = math.pi / 2 - (
        torch.arange(out_h, device=device, dtype=torch.float32) + 0.5
    ) / out_h * math.pi
    return lon.unsqueeze(0).expand(out_h, out_w), lat.unsqueeze(1).expand(out_h, out_w)


def _sample(src: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """src [B,H,W,C], px/py [out_h,out_w] (pixel coords, may be far out of range)."""
    B, H, W, _ = src.shape
    gx = ((px + 0.5) / W * 2 - 1).unsqueeze(-1)
    gy = ((py + 0.5) / H * 2 - 1).unsqueeze(-1)
    grid = torch.cat((gx, gy), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    out = F.grid_sample(
        src.permute(0, 3, 1, 2).float(), grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )
    return out.permute(0, 2, 3, 1).clamp(0.0, 1.0)


def _dir_from_lonlat(lon: torch.Tensor, lat: torch.Tensor):
    cos_lat = lat.cos()
    return torch.stack(
        (cos_lat * lon.sin(), lat.sin(), cos_lat * lon.cos()), dim=-1
    )  # [out_h, out_w, 3]


# ---------------------------------------------------------------- fisheye

def _fisheye_px(
    lon: torch.Tensor, lat: torch.Tensor,
    axis, e1, e2,                       # lens basis (unit world vectors)
    proj: str, half_fov_rad: float,
    radius: float, cx: float, cy: float, mirror: bool,
):
    """Return pixel coords (px, py) [out_h,out_w] for equirect pixels on this lens.
    Far-out-of-range values mean 'this lens doesn't cover that direction'."""
    d = _dir_from_lonlat(lon, lat)                       # [..., 3]
    a = torch.tensor(axis, dtype=torch.float32)
    v1 = torch.tensor(e1, dtype=torch.float32)
    v2 = torch.tensor(e2, dtype=torch.float32)
    cos_c = (d * a).sum(-1).clamp(-1.0, 1.0)
    c = torch.acos(cos_c)                                # 角距
    # in-plane components
    p1 = (d * v1).sum(-1)
    p2 = (d * v2).sum(-1)
    sin_c = torch.sqrt((p1 * p1 + p2 * p2)).clamp_min(1e-6)

    if proj == "equisolid":
        r = radius * 2 * torch.sin(c / 2) / (2 * math.sin(half_fov_rad / 2))
    else:  # equidistant（多数360相机鱼眼）
        r = radius * c / half_fov_rad

    sx = -1.0 if mirror else 1.0
    px = cx + sx * r * p1 / sin_c
    py = cy - r * p2 / sin_c

    invalid = (c > half_fov_rad) | (r > radius * 1.6)
    px = torch.where(invalid, torch.full_like(px, 1e6), px)
    py = torch.where(invalid, torch.full_like(py, 1e6), py)
    c = torch.where(invalid, torch.full_like(c, 99.0), c)
    return px, py, c


def dual_fisheye_to_equirect(
    img: torch.Tensor, layout: str = "lr", proj: str = "equidistant",
    fov_deg: float = 210.0, radius_frac: float = 1.0, spacing_frac: float = 1.0,
    mirror: bool = False, output_width: int = 2048,
) -> torch.Tensor:
    B, H, W, _ = img.shape
    out_w, out_h = int(output_width), max(1, int(output_width) // 2)
    lon, lat = lonlat_grids(out_w, out_h, img.device)
    half = math.radians(min(max(fov_deg, 100.0), 359.9)) / 2

    if layout == "lr":
        gap = 0.25 * W * spacing_frac
        lenses = [
            dict(axis=(0, 0, 1), e1=(1, 0, 0), e2=(0, 1, 0),
                 cx=W / 2 - gap, cy=H / 2, radius=min(H / 2, gap) * radius_frac),
            dict(axis=(0, 0, -1), e1=(-1, 0, 0), e2=(0, 1, 0),
                 cx=W / 2 + gap, cy=H / 2, radius=min(H / 2, gap) * radius_frac),
        ]
    else:  # tb 上下
        gap = 0.25 * H * spacing_frac
        lenses = [
            dict(axis=(0, 1, 0), e1=(1, 0, 0), e2=(0, 0, 1),
                 cx=W / 2, cy=H / 2 - gap, radius=min(W / 2, gap) * radius_frac),
            dict(axis=(0, -1, 0), e1=(1, 0, 0), e2=(0, 0, -1),
                 cx=W / 2, cy=H / 2 + gap, radius=min(W / 2, gap) * radius_frac),
        ]

    p1x, p1y, c1 = _fisheye_px(lon, lat, proj=proj, half_fov_rad=half,
                               mirror=mirror, **lenses[0])
    p2x, p2y, c2 = _fisheye_px(lon, lat, proj=proj, half_fov_rad=half,
                               mirror=mirror, **lenses[1])
    use1 = c1 <= c2  # 取角距更小的镜头（无效侧 c=99 自然被淘汰），接缝落在重叠区中央
    px = torch.where(use1, p1x, p2x)
    py = torch.where(use1, p1y, p2y)
    return _sample(img, px, py)


def single_fisheye_to_equirect(
    img: torch.Tensor, proj: str = "equidistant", fov_deg: float = 210.0,
    radius_frac: float = 1.0, mirror: bool = False, output_width: int = 2048,
) -> torch.Tensor:
    B, H, W, _ = img.shape
    out_w, out_h = int(output_width), max(1, int(output_width) // 2)
    lon, lat = lonlat_grids(out_w, out_h, img.device)
    half = math.radians(min(max(fov_deg, 100.0), 359.9)) / 2
    px, py, _c = _fisheye_px(
        lon, lat, axis=(0, 0, 1), e1=(1, 0, 0), e2=(0, 1, 0),
        proj=proj, half_fov_rad=half, mirror=mirror,
        radius=min(W, H) / 2 * radius_frac, cx=W / 2, cy=H / 2,
    )
    return _sample(img, px, py)


# ---------------------------------------------------------------- cylinder / wide / cubemap

def cylindrical_strip_to_equirect(
    img: torch.Tensor, strip_hfov_deg: float = 360.0, output_width: int = 2048,
) -> torch.Tensor:
    """手机全景长条：横向角度线性，纵向 tan(lat) 线性。"""
    B, H, W, _ = img.shape
    out_w, out_h = int(output_width), max(1, int(output_width) // 2)
    lon, lat = lonlat_grids(out_w, out_h, img.device)
    hfov = math.radians(min(max(strip_hfov_deg, 10.0), 360.0))
    f = W / hfov  # 像素焦距
    px = (lon / hfov + 0.5) * W
    py = H / 2 - f * torch.tan(lat.clamp(-1.55, 1.55))
    return _sample(img, px, py)


def wide_photo_to_equirect(
    img: torch.Tensor, hfov_deg: float = 90.0, output_width: int = 2048,
) -> torch.Tensor:
    """普通宽幅照片/部分全景（rectilinear 直线投影）。"""
    B, H, W, _ = img.shape
    out_w, out_h = int(output_width), max(1, int(output_width) // 2)
    lon, lat = lonlat_grids(out_w, out_h, img.device)
    hfov = math.radians(min(max(hfov_deg, 10.0), 179.0))
    f = (W / 2) / math.tan(hfov / 2)
    px = W / 2 + f * torch.tan(lon.clamp(-(hfov / 2 - 1e-3), hfov / 2 - 1e-3))
    py = H / 2 - f * torch.tan(lat.clamp(-1.55, 1.55))
    return _sample(img, px, py)


# facelet 布局约定：3x2 网格 (row-major) 中六个面的位置
_CUBEMAP_LAYOUTS = {
    "LFR_BUD": {  # Facebook 3x2：左 前 右 / 后 上 下
        "L": (0, 0), "F": (0, 1), "R": (0, 2),
        "B": (1, 0), "U": (1, 1), "D": (1, 2),
    },
    "FRB_LUD": {  # py360convert 顺序 F,R,B,L,U,D
        "F": (0, 0), "R": (0, 1), "B": (0, 2),
        "L": (1, 0), "U": (1, 1), "D": (1, 2),
    },
}


def cubemap32_to_equirect(img: torch.Tensor, layout: str = "LFR_BUD",
                          output_width: int = 2048) -> torch.Tensor:
    B, H, W, _ = img.shape
    out_w, out_h = int(output_width), max(1, int(output_width) // 2)
    fh, fw = H // 2, W // 3
    lon, lat = lonlat_grids(out_w, out_h, img.device)
    d = _dir_from_lonlat(lon, lat)                       # [...,3] x右 y上 z前
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    ax, ay, az = dx.abs(), dy.abs(), dz.abs()

    pos = _CUBEMAP_LAYOUTS[layout]
    # 每个面: (face_name, 遮罩, fu, fv)   fu,fv∈[0,1) 面内坐标
    defs = [
        ("F", (az >= ax) & (az >= ay) & (dz > 0), (dx / az + 1) / 2, (1 - dy / az) / 2),
        ("B", (az >= ax) & (az >= ay) & (dz <= 0), (1 - dx / az) / 2, (1 - dy / az) / 2),
        ("R", (ax > az) & (ax >= ay) & (dx > 0), (1 - dz / ax) / 2, (1 - dy / ax) / 2),
        ("L", (ax > az) & (ax >= ay) & (dx <= 0), (1 + dz / ax) / 2, (1 - dy / ax) / 2),
        ("U", (ay > ax) & (ay > az), (dx / ay + 1) / 2, (dz / ay + 1) / 2),
        ("D", (ay > ax) & (ay > az) & (dy <= 0), (dx / ay + 1) / 2, (1 - dz / ay) / 2),
    ]
    px = torch.full_like(lon, 1e6)
    py = torch.full_like(lon, 1e6)
    for name, mask, fu, fv in defs:
        if not mask.any():
            continue
        row, col = pos[name]
        px = torch.where(mask, (col + fu.clamp(0, 1)) * fw, px)
        py = torch.where(mask, (row + fv.clamp(0, 1)) * fh, py)
    return _sample(img, px, py)


# ---------------------------------------------------------------- tiny planet（自包含副本，与 TinyPlanet2Equirect 同源）

def tinyplanet_to_equirect(
    img: torch.Tensor, projection: str = "stereographic",
    edge_angle_deg: float = 106.0, planet_radius_frac: float = 1.0,
    mirror: bool = False, output_width: int = 2048,
) -> torch.Tensor:
    B, H, W, C = img.shape
    out_w = int(output_width)
    out_h = max(1, out_w // 2)
    cx, cy = W / 2.0, H / 2.0
    r_pix = float(planet_radius_frac) * min(H, W) / 2.0
    edge_rad = math.radians(min(float(edge_angle_deg), 179.5))

    device = img.device
    lon, lat = lonlat_grids(out_w, out_h, device)
    c = torch.acos((-lat.sin()).clamp(-1.0, 1.0))
    if projection == "stereographic":
        rho = torch.tan(c / 2) / math.tan(edge_rad / 2)
    else:
        rho = c / edge_rad
    sx = -1.0 if mirror else 1.0
    u = cx + sx * r_pix * rho * lon.cos()
    v = cy - r_pix * rho * lon.sin()
    return _sample(img, u, v)


# ---------------------------------------------------------------- helpers for stitched panos

def resize_width(img: torch.Tensor, width: int) -> torch.Tensor:
    """[B,H,W,C] 等比缩放到指定宽度。"""
    b, h, w, c = img.shape
    if w == width:
        return img
    nh = max(1, round(h * width / w))
    return F.interpolate(
        img.permute(0, 3, 1, 2).float(), size=(nh, width),
        mode="bilinear", align_corners=False,
    ).permute(0, 2, 3, 1).clamp(0.0, 1.0)


def pad_to_equirect(img: torch.Tensor) -> torch.Tensor:
    """把拼接出的宽幅全景补天补地到 2:1（上下边缘行延展填充）。"""
    b, h, w, c = img.shape
    target_h = w // 2
    if h == target_h:
        return img
    if h > target_h:  # 过高：居中裁剪
        top = (h - target_h) // 2
        return img[:, top:top + target_h]
    pad = target_h - h
    top_n = pad // 2
    bot_n = pad - top_n
    top = img[:, :1].expand(-1, top_n, -1, -1).contiguous()
    bot = img[:, -1:].expand(-1, bot_n, -1, -1).contiguous()
    return torch.cat((top, img, bot), dim=1)
