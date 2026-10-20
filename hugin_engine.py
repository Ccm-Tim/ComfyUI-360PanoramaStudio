"""Hugin 引擎：调用 Hugin 命令行实现真 360° 无缝回环的全景拼接。

管线：pto_gen（建工程）→ cpfind（控制点，多排可选）→ autooptimiser（全局优化，
含回环闭合与水平矫正）→ 重写 p 行为 equirect f4 → nona（重投影渲染）→
enblend（多图融合）。输出即标准 2:1 等距柱状全景。
"""

import glob
import os
import re
import shutil
import subprocess

import cv2
import numpy as np


def find_hugin_bin():
    candidates = [os.environ.get("HUGIN_PATH")]
    for base in (r"C:\Program Files\Hugin\bin",
                 r"C:\Program Files (x86)\Hugin\bin",
                 r"C:\Hugin\bin"):
        candidates.append(base)
    candidates += glob.glob(r"C:\Program Files\Hugin*\bin")
    which = shutil.which("hugin_executor") or shutil.which("nona")
    if which:
        candidates.append(os.path.dirname(which))
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "nona.exe")):
            return c
    return None


def _run(exe, args, workdir, timeout):
    r = subprocess.run([exe] + args, cwd=workdir, capture_output=True,
                       text=True, timeout=timeout, errors="replace")
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "")[-700:]
        raise RuntimeError(
            f"Hugin 命令失败：{os.path.basename(exe)} {' '.join(args[:3])} …\n{tail}")


def _set_equirect(pto_path, width, height):
    """把工程输出投影改写为 equirectangular(f4) 并设定画布尺寸。"""
    with open(pto_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    def fix_line(m):
        toks = m.group(0).split()
        out = []
        for t in toks:
            if re.fullmatch(r"f\d+", t):
                t = "f4"
            elif re.fullmatch(r"w\d+", t):
                t = f"w{width}"
            elif re.fullmatch(r"h\d+", t):
                t = f"h{height}"
            out.append(t)
        return " ".join(out)

    for i, line in enumerate(lines):
        if line.startswith("p "):
            lines[i] = fix_line(re.match(r"^p .*$", line)) + "\n"
            break
    else:
        raise RuntimeError("pto 文件缺少 p 行（工程异常）")
    with open(pto_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def stitch_hugin(frames_bgr, workdir, equirect_width=4096, multirow=False,
                 timeout=900):
    """frames_bgr: list[np.uint8 BGR]。返回 equirect 2:1 的 np BGR 全景图。"""
    bin_dir = find_hugin_bin()
    if not bin_dir:
        raise RuntimeError(
            "未找到 Hugin。请安装：winget install Hugin.Hugin，"
            "或把 HUGIN_PATH 环境变量指向 Hugin 的 bin 目录后重试。")
    exe = lambda name: os.path.join(bin_dir, name)

    files = []
    for i, frame in enumerate(frames_bgr):
        p = os.path.join(workdir, f"img_{i:02d}.jpg")
        cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(os.path.basename(p))

    os.makedirs(workdir, exist_ok=True)
    _run(exe("pto_gen"), ["-o", "p.pto"] + files, workdir, 180)
    cpfind_args = [exe("cpfind")]
    if multirow:
        cpfind_args.append("--multirow")
    cpfind_args += ["-o", "p.pto", "p.pto"]
    _run(cpfind_args, workdir, timeout)

    # 全局优化：-a 自动对中 -m 计算最佳画幅 -l 水平矫正 -S 摆正
    opt = subprocess.run(
        [exe("autooptimiser"), "-a", "-m", "-l", "-S", "-o", "pa.pto", "p.pto"],
        cwd=workdir, capture_output=True, text=True, timeout=timeout,
        errors="replace")
    if opt.returncode != 0:
        # 参数不兼容时退回基础旗标
        _run(exe("autooptimiser"), ["-a", "-l", "-o", "pa.pto", "p.pto"],
             workdir, timeout)

    _set_equirect(os.path.join(workdir, "pa.pto"), equirect_width,
                  equirect_width // 2)
    _run(exe("nona"), ["-o", "pano", "pa.pto"], workdir, timeout)
    tifs = sorted(glob.glob(os.path.join(workdir, "pano*.tif")))
    if not tifs:
        raise RuntimeError("nona 未输出全景图（工程可能没有足够控制点）")
    final = os.path.join(workdir, "pano_final.tif")
    _run(exe("enblend"), ["-o", "pano_final.tif"] +
         [os.path.basename(t) for t in tifs], workdir, timeout)
    pano = cv2.imread(final, cv2.IMREAD_COLOR)
    if pano is None:
        raise RuntimeError("无法读取 enblend 输出")
    return pano
