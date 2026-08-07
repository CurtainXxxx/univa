"""
评测指标：CLIP Score / DINO Score / MLLM Judge
===============================================
UniVA-Bench 评测用的三个"尺子"：

1. CLIP Score   — 生成视频 vs 文本 prompt 的语义对齐度
2. DINO Score   — 生成视频 vs 源视频/参考图的内容保留度
3. MLLM Judge   — 视觉大模型对生成质量的主观打分

全部 CPU 可跑（本地 Windows，无 GPU）。
CLIP/DINO 用 transformers 加载模型，MLLM 复用 DashScope qwen-vl-max。
"""

import os
import cv2
import numpy as np
from functools import lru_cache

import torch
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor


# =============================================================================
# 视频抽帧工具
# =============================================================================
def sample_frames(video_path: str, n: int = 16) -> list:
    """从视频均匀抽 n 帧，返回 RGB numpy 数组列表。

    用 opencv 读取，均匀取 n 帧代表整个视频。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"视频无帧: {video_path}")

    # 均匀取 n 个帧索引
    n = min(n, total)
    indices = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"抽帧失败: {video_path}")
    return frames


def sample_frame(video_path: str) -> np.ndarray:
    """取视频第一帧，作为图片输入（用于 DINO 对比参考图）."""
    frames = sample_frames(video_path, n=1)
    return frames[0]


# =============================================================================
# ① CLIP Score — 语义对齐
# =============================================================================
@lru_cache(maxsize=1)
def _load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return model, processor


def clip_text_video_sim(prompt: str, video_path: str, n: int = 16) -> float:
    """CLIP Score: 文本 prompt ↔ 生成视频的语义相似度（0~1）.

    lt2v / it2v / lve 用：衡量"说啥做啥"。
    """
    model, processor = _load_clip()
    frames = sample_frames(video_path, n)

    # 每个 frame 配同一个 prompt，一次性编码
    inputs = processor(
        text=[prompt] * len(frames),
        images=frames,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    sims = torch.nn.functional.cosine_similarity(
        outputs.text_embeds, outputs.image_embeds, dim=-1
    )
    return sims.mean().item()


def clip_image_video_sim(image_path: str, video_path: str, n: int = 16) -> float:
    """CLIP Score: 参考图 ↔ 生成视频前几帧的相似度.

    it2v 用：衡量生成的视频是否保留参考图内容。
    """
    model, processor = _load_clip()
    frames = sample_frames(video_path, min(n, 8))
    ref_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    inputs = processor(
        images=[ref_img] + frames,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = model.get_image_features(**inputs)
    embeds = torch.nn.functional.normalize(embeds, dim=-1)
    # 参考图 vs 每帧的余弦
    ref = embeds[0]
    sims = (embeds[1:] * ref).sum(dim=-1)
    return sims.mean().item()


# =============================================================================
# ② DINO Score — 内容保留
# =============================================================================
@lru_cache(maxsize=1)
def _load_dino():
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model.eval()
    return model, processor


def _dino_embed(images: list) -> np.ndarray:
    model, processor = _load_dino()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        feats = model(**inputs).last_hidden_state.mean(dim=1)  # [N, 768]
    feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.numpy()


def dino_video_sim(video_a: str, video_b: str, n: int = 16) -> float:
    """DINO Score: 两个视频逐帧内容一致性（0~1）.

    v2v / lve 用：衡量编辑/转换后是否保留原视频结构。
    """
    fa = sample_frames(video_a, n)
    fb = sample_frames(video_b, n)
    ea = _dino_embed(fa)
    eb = _dino_embed(fb)
    # 逐帧余弦相似度取平均
    sims = (ea * eb).sum(axis=-1)
    return sims.mean().item()


# =============================================================================
# ③ MLLM Judge — 主观质量
# =============================================================================
def mllm_judge(video_path: str, prompt: str, max_score: int = 5) -> float:
    """MLLM Judge: 视觉大模型对生成视频打分（1~max_score）.

    复用 DashScope qwen-vl-max（multimodal_query）。
    注意：此函数需要 DashScope key，且消耗 token。
    """
    try:
        from utils.query_llm import multimodal_query
    except ImportError:
        # 兼容不同 cwd
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "univa"))
        from utils.query_llm import multimodal_query

    question = (
        f"请为这个视频与描述『{prompt}』的匹配程度打分，"
        f"输出一个 1 到 {max_score} 的整数，只输出数字。"
    )
    r = multimodal_query(question, video_path=video_path, video_frames_to_extract=16)
    # 从回答里提取数字
    import re
    nums = re.findall(r"\d+", r or "")
    if nums:
        val = int(nums[0])
        return max(1, min(max_score, val))
    return 0.0


# =============================================================================
# 便捷汇总
# =============================================================================
def score_task(task: str, video_path: str, prompt: str = "",
               source_video: str = None, reference_image: str = None) -> dict:
    """按任务类型返回对应指标的字典.

    Returns:
        {"clip": float|None, "dino": float|None, "mllm": float|None}
    """
    result = {"clip": None, "dino": None, "mllm": None}

    if task in ("longtext2video", "image_text2video", "longvideoediting"):
        if prompt:
            result["clip"] = clip_text_video_sim(prompt, video_path)

    if task in ("video2video", "longvideoediting") and source_video:
        result["dino"] = dino_video_sim(source_video, video_path)

    if reference_image and task == "image_text2video":
        result["clip"] = clip_image_video_sim(reference_image, video_path)

    # MLLM 可选（耗时 + 耗 token），默认不自动跑
    return result


if __name__ == "__main__":
    # 快速自测：用一个生成的视频算 CLIP
    import sys, json
    if len(sys.argv) < 2:
        print("用法: python metrics.py <video.mp4> [prompt]")
        sys.exit(0)
    v = sys.argv[1]
    p = sys.argv[2] if len(sys.argv) > 2 else "a cat on a beach"
    s = clip_text_video_sim(p, v)
    print(f"CLIP Score({p[:30]}...): {s:.4f}")
