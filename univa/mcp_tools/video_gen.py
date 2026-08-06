"""
视频生成 MCP 服务器
===================
UniVA 最核心的 MCP 工具，提供 6 个视频生成/编辑工具。

作为独立子进程运行，通过 stdio 与主 Agent 进程通信（MCP 协议）。
ActAgent 调用这里的工具，这里再转发给 Wavespeed API 实际生成视频。

## MCP 工具注册模式
每个函数加 @mcp.tool() 装饰器 → FastMCP 自动暴露给 Agent 进程。
Agent 通过 LLM function call 调用，参数和返回值都是 JSON。

## 模型选择
工具根据 config.yaml 里的 model 字段选择后端 API：
- seedance  → ByteDance Seedance (文本/图片→视频)
- wan_api   → Wan2.1 (帧间插值/帧到帧)
- flux-kontext → 图像生成

## 调用链路
用户输入 → PlanAgent 拆任务 → ActAgent function call → MCP 协议 → 本文件函数
  → Wavespeed API (提交任务+轮询) → 下载视频 → 返回路径
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.video_process import merge_videos, storyboard_generate, save_last_frame_decord
from utils.query_llm import refine_gen_prompt, audio_prompt_gen
from utils.image_process import download_image
from utils.wavespeed_api import text_to_video_generate, image_to_video_generate, frame_to_frame_video, text_to_image_generate, image_to_image_generate, audio_gen, hailuo_i2v_pro

# =============================================================================
# 配置加载：YAML + .env 覆盖
# 注意：这是独立子进程，必须自己加载 .env，主进程的环境变量不会传递过来
# =============================================================================
_univa_root = Path(__file__).resolve().parents[1]
os.chdir(str(_univa_root))

# 加载 .env 中的 API Key
_env_file = _univa_root.parent / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=str(_env_file), override=False)

# 读取 YAML 基础配置
config_path = _univa_root / "config" / "mcp_tools_config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# .env 中的 WAVESPEED_API_KEY 覆盖所有 sections 的 wavespeed_api
_wavespeed_key = os.environ.get("WAVESPEED_API_KEY", "")
if _wavespeed_key:
    for _section in ["image_gen", "video_editing", "video_gen", "audio_gen"]:
        if _section in config:
            config[_section]["wavespeed_api"] = _wavespeed_key
if os.environ.get("LLM_OPENAI_API_KEY"):
    config.setdefault("llm", {})["openai_api_key"] = os.environ["LLM_OPENAI_API_KEY"]

video_gen_config = config.get('video_gen', {})
image_gen_config = config.get('image_gen', {})

logger = setup_logger(__name__, "logs/mcp_tools", "video_gen.log")
logger.info(f"Loaded video_gen_config: {video_gen_config}")

# FastMCP 实例化 — 这是 MCP 服务器的入口
mcp = FastMCP("Video_Generation_Server")


# =============================================================================
# 工具1: text2video_gen — 文本 → 视频（最常用）
# =============================================================================
@mcp.tool()
async def text2video_gen(prompt: str) -> Dict:
    """
    根据文本描述生成约 5 秒的短视频。

    使用场景：
    - "一只小猫在跳舞" → 5秒小猫跳舞视频
    - "日落海滩，海浪轻轻拍打沙滩" → 5秒海滩视频

    后端模型：ByteDance Seedance（通过 Wavespeed API 代理）

    Args:
        prompt: 视频描述文本（英文最佳）
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    model = video_gen_config.get("text_to_video")

    if model == "seedance":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = text_to_video_generate(api_key, prompt, save_path=save_path)

        return return_dict


# =============================================================================
# 工具2: storyvideo_gen — 故事视频（多镜头叙事）
# =============================================================================
@mcp.tool()
async def storyvideo_gen(prompt: str) -> ToolResponse:
    """
    进阶工具：从一段故事描述生成多镜头完整视频。

    内部流程（6步流水线）：
    1. LLM 生成故事板（角色列表 + 分镜表 + 风格）
    2. 为每个角色生成参考图（text → image）
    3. 为每个镜头的每个角色生成关键帧图
    4. 关键帧图 → 视频片段
    5. （可选）为视频片段配乐
    6. ffmpeg 拼接所有片段成一个完整视频

    这个工具是整个项目最复杂的工具，完整展示了
    "LLM 做规划 + API 做生成 + ffmpeg 做后处理" 的组合模式。

    Args:
        prompt: 故事描述（英文最佳），例如 "A young wizard discovers a hidden library..."
    Returns:
        ToolResponse(success, output_path, message)
    """
    model = video_gen_config.get("text_to_video")
    if model == "seedance":
        save_dir = f"infer/v2v/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        # ---- 步骤1：LLM 生成故事板 ----
        storyboard = await storyboard_generate(prompt)
        time = datetime.now().strftime("%m%d%H%M%S")
        with open(f"{time}_storyboard.json", "w") as f:
            json.dump(storyboard, f, indent=4)

        # ---- 步骤2：为每个角色生成参考图 ----
        characters = storyboard.get("characters")
        characters_image_path = dict()

        style = storyboard.get("style")

        for character in characters:
            char_id = character.get("id")
            char_description = character.get("description")
            # 用 LLM 优化角色描述，让图片生成更稳定
            refined_char_description = refine_gen_prompt(char_description, media_type="character")

            api_key = image_gen_config.get("wavespeed_api")
            image_url = text_to_image_generate(api_key, refined_char_description, aspect_ratio="1:1")
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{char_id}.jpg"
            characters_image_path[char_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        # ---- 步骤3-4：每个镜头生成关键帧 → 视频 ----
        shots = storyboard.get("shots")
        shots_image_path = dict()
        for shot in shots:
            shot_id = shot.get("id")
            setting_description = shot.get("setting_description")
            plot_correspondence = shot.get("plot_correspondence")
            static_shot_description = shot.get("static_shot_description")
            onstage_characters = shot.get("onstage_characters")
            onstage_characters_image_path_list = [characters_image_path.get(char_id) for char_id in onstage_characters]

            shot_perspective_design = shot.get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"

            api_key = image_gen_config.get("wavespeed_api")
            if len(onstage_characters_image_path_list) == 0:
                image_url = text_to_image_generate(api_key, keyframe_prompt, aspect_ratio="4:3")
            else:
                image_url = image_to_image_generate(api_key, keyframe_prompt, onstage_characters_image_path_list, aspect_ratio="4:3")
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{shot_id}.jpg"
            shots_image_path[shot_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        # ---- 步骤5：关键帧 → 视频片段 ----
        video_segment_list = []
        for idx, (keyframe_id, keyframe) in enumerate(shots_image_path.items()):
            api_key = video_gen_config.get("wavespeed_api")

            setting_description = shots[idx].get("setting_description")
            plot_correspondence = shots[idx].get("plot_correspondence")
            static_shot_description = shots[idx].get("static_shot_description")

            shot_perspective_design = shots[idx].get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"

            time = datetime.now().strftime("%m%d%H%M%S")
            save_path = f"{save_dir}/{time}_{keyframe_id}.mp4"
            return_dict = image_to_video_generate(api_key, keyframe_prompt, keyframe, save_path=save_path)

            if return_dict.get("success"):
                video_segment_list.append(return_dict.get("output_path"))

        # ---- 步骤6：ffmpeg 拼接所有片段 ----
        time = datetime.now().strftime("%m%d%H%M%S")
        prompt_name = prompt.replace(" ", "_")[:50]
        movie_save_path = f"{save_dir}/{time}_{prompt_name}.mp4"
        video_path = merge_videos(video_segment_list, output_file=movie_save_path)

        return ToolResponse(
            success=True,
            output_path=video_path,
            message="Video generated successfully."
        )


# =============================================================================
# 工具3: entity2video — 用户提供角色图片的故事视频
# =============================================================================
@mcp.tool()
async def entity2video(prompt: str, images: List[str]) -> ToolResponse:
    """
    与 storyvideo_gen 类似，但角色图片由用户提供而非 AI 生成。

    适用场景：用你的宠物/产品/人物的照片制作故事视频。
    这样可以保持角色外观一致性（identity preservation）。

    Args:
        prompt: 故事描述
        images: 角色图片的本地路径列表
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    model = video_gen_config.get("text_to_video")
    if model == "seedance":
        save_dir = f"infer/v2v/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)

        prompt = f"{prompt}\nsource images path: {str(images)}"
        storyboard = await storyboard_generate(prompt, gentype="entity2video")
        characters_image_path = dict()

        style = storyboard.get("style")

        # 直接用用户提供的图片作为角色参考图
        characters = storyboard.get("characters")
        for idx, character in enumerate(characters):
            char_id = f"char_{idx+1}"
            character_path = character.get("path")
            characters_image_path[char_id] = character_path

        shots = storyboard.get("shots")
        shots_image_path = dict()
        for shot in shots:
            shot_id = shot.get("id")
            setting_description = shot.get("setting_description")
            plot_correspondence = shot.get("plot_correspondence")
            static_shot_description = shot.get("static_shot_description")
            onstage_characters = shot.get("onstage_characters")
            onstage_characters_image_path_list = [characters_image_path.get(char_id) for char_id in onstage_characters]

            shot_perspective_design = shot.get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"
            api_key = image_gen_config.get("wavespeed_api")
            if len(onstage_characters_image_path_list) == 0:
                image_url = text_to_image_generate(api_key, keyframe_prompt)
            else:
                image_url = image_to_image_generate(api_key, keyframe_prompt, onstage_characters_image_path_list)
            time = datetime.now().strftime("%m%d%H%M%S")
            image_save_path = f"{save_dir}/{time}_{shot_id}.jpg"
            shots_image_path[shot_id] = image_save_path
            download_image(image_url, save_path=image_save_path)

        video_segment_list = []
        for idx, (keyframe_id, keyframe) in enumerate(shots_image_path.items()):
            api_key = video_gen_config.get("wavespeed_api")

            setting_description = shots[idx].get("setting_description")
            plot_correspondence = shots[idx].get("plot_correspondence")
            static_shot_description = shots[idx].get("static_shot_description")

            shot_perspective_design = shots[idx].get("shot_perspective_design")
            distance = shot_perspective_design.get("distance")
            angle = shot_perspective_design.get("angle")
            lens = shot_perspective_design.get("lens")

            keyframe_prompt = f"{setting_description} {plot_correspondence} {static_shot_description} {distance}, {angle}, {lens}"

            time = datetime.now().strftime("%m%d%H%M%S")
            save_path = f"{save_dir}/{time}_{keyframe_id}.mp4"
            return_dict = image_to_video_generate(api_key, keyframe_prompt, keyframe, save_path=save_path)

            if return_dict.get("success"):
                video_segment_list.append(return_dict.get("output_path"))

        time = datetime.now().strftime("%m%d%H%M%S")
        prompt_name = prompt.replace(" ", "_")[:50]
        movie_save_path = f"{save_dir}/{time}_{prompt_name}.mp4"
        video_path = merge_videos(video_segment_list, output_file=movie_save_path)

        return ToolResponse(
            success=True,
            output_path=video_path,
            message="Video generated successfully."
        )


# =============================================================================
# 工具4: image2video_gen — 图片 + 文本 → 视频（图生视频）
# =============================================================================
@mcp.tool()
async def image2video_gen(prompt: str, image_path: str) -> Dict:
    """
    以一张图片为视觉参考，生成约 5 秒视频。

    这是图生视频（image-to-video），核心用途：
    - 让静态图片"动起来"
    - 保持与参考图一致的视觉风格

    Args:
        prompt: 描述想要的动作/效果
        image_path: 参考图片路径
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    model = video_gen_config.get("image_to_video")

    if model == "seedance":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = image_to_video_generate(api_key, prompt, image_path, save_path=save_path)

        return return_dict


# =============================================================================
# 工具5: video_extension — 视频延长（取最后一帧向前生成）
# =============================================================================
@mcp.tool()
async def video_extension(prompt: str, video_path: str) -> Dict:
    """
    给已有视频续写新内容。

    原理：取视频最后一帧作为 image2video 的参考图 → 生成新片段。
    效果类似于让视频"多播几秒"。

    Args:
        prompt: 续写内容的描述
        video_path: 原视频路径
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    time_ = datetime.now().strftime("%m%d%H%M%S")
    last_frame_save_path = save_last_frame_decord(video_path, f"results/{time_}_last_frame.png")
    extend_result = await image2video_gen(prompt, last_frame_save_path)
    if extend_result.get("success"):
        output_video_path = extend_result.get("output_path")
        return ToolResponse(
            success=True,
            output_path=output_video_path,
            message="Video extended and merged successfully."
        )
    else:
        return ToolResponse(
            success=False,
            message="Video extension failed at image2video generation step.",
        )


# =============================================================================
# 工具6: frame2frame_video_gen — 首帧到末帧过渡（帧间插值）
# =============================================================================
@mcp.tool()
async def frame2frame_video_gen(prompt: str, first_frame_path: str, last_frame_path: str) -> Dict:
    """
    给定首帧和末帧，生成中间过渡视频。

    适用场景：动态动作序列、平滑转场。

    Args:
        prompt: 描述过渡动作
        first_frame_path: 起始帧路径
        last_frame_path: 结束帧路径
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    model = video_gen_config.get("frame_to_frame_video")

    if model == "wan_api":
        api_key = video_gen_config.get("wavespeed_api")
        save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}_{prompt[:30].replace(' ', '_')}"
        os.makedirs(save_dir, exist_ok=True)
        _time = datetime.now().strftime("%m%d%H%M%S")
        save_path = f"{save_dir}/{_time}.mp4"
        return_dict = hailuo_i2v_pro(api_key, prompt, first_frame_path, last_frame_path, save_path=save_path)

        return return_dict


# =============================================================================
# 工具7: merge2videos — 视频拼接
# =============================================================================
@mcp.tool()
async def merge2videos(video_paths: list[str]):
    """
    把多个视频文件拼接成一个（ffmpeg concat）。

    Args:
        video_paths: 视频文件路径列表，或包含视频的文件夹路径
    Returns:
        {'success': bool, 'output_path': str, 'message': str}
    """
    save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    _time = datetime.now().strftime("%m%d%H%M%S")
    save_path = f"{save_dir}/{_time}.mp4"
    video_path = merge_videos(video_paths, output_file=save_path)

    return ToolResponse(
        success=True,
        output_path=video_path,
        message="Videos merged successfully."
    )


if __name__ == "__main__":
    # 作为独立 MCP 服务器启动，通过 stdio 和父进程通信
    mcp.run(transport="stdio")
