"""
Wavespeed AI API 客户端
========================
UniVA 所有生成能力（图片/视频/音频）的底层 API 封装。

## Wavespeed AI 是什么
一个 AI 模型聚合平台（API 代理），统一封装了：
- ByteDance Seedance（视频生成）
- ByteDance Seedream-v4（图片编辑/生成）
- Flux Kontext（图片生成）
- MiniMax Hailuo（图生视频）
- MiniMax Speech（TTS 语音合成）
- MMAudio-v2（视频配乐）
- RunwayML Gen-4（视频风格迁移）
- Wan2.1-VACE（视频编辑/深度/姿态）

## 为什么用 Wavespeed 而不是直接调各厂商 API
- 统一鉴权：一个 API Key 访问所有模型
- 统一接口：所有模型都是 POST 提交 → poll 等待 → GET 下载
- 免部署：不用自己搭 GPU 服务器跑 Wan2.1/Qwen 等大模型

## 所有函数遵循同一种模式（异步任务模式）

    def xxx_generate(api_key, prompt, ...):
        # 1. 构造请求 payload（图片要先转 base64）
        payload = {"prompt": prompt, ...}
        response = requests.post(TASK_SUBMIT_URL, headers, json=payload)

        # 2. 拿到 task_id，开始轮询
        request_id = response.json()["data"]["id"]

        # 3. 轮询直到 completed 或 failed
        while True:
            status = requests.get(POLL_URL + request_id)
            if status == "completed":
                # 4. 下载输出文件（视频/图片/音频）
                download(output_url, save_path)
                return {'success': True, 'output_path': save_path}
            elif status == "failed":
                return {'success': False, 'error': ...}

    这种模式来自 GPU 推理的异步特性——视频生成通常需要几十秒到几分钟。

## 调试技巧
- 所有 API 调用的调试日志在 TEMP/univa_logs/logs/ 下
- 关键看 logger.info 输出的 Request ID 和状态变化
"""

import requests
import json
import time
import logging
import os
import base64
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()


# 全局 requests session：显式绕过系统/环境代理。
# 背景：本机 7897 代理对大请求体（base64 图片/视频）处理有缺陷，
# 会导致 Wavespeed 提交任务时 ProxyError 断连。wavespeed.ai 直连即可，
# 因此这里禁用 trust_env（不读系统代理）并强制直连。
_requests_session = requests.Session()
_requests_session.trust_env = False
_requests_session.proxies = {"http": None, "https": None}


def _get_with_retry(url, headers, retries=5, backoff=1.0, timeout=60):
    """轮询 GET，SSL/网络异常自动重试，指数退避。

    背景：Wavespeed 生成任务常需几十秒~几分钟，轮询期间本地网络
    SSL 偶发断连（SSL: UNEXPECTED_EOF_WHILE_READING）。无重试会导致
    整个调用崩溃、浪费已提交的任务。重试耗尽后返回空 Response，
    让调用方走 status_code != 200 的错误分支。
    """
    for attempt in range(retries):
        try:
            return _requests_session.get(url, headers=headers, timeout=timeout)
        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            logger.warning(f"GET failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt == retries - 1:
                return requests.Response()  # 空响应，status_code != 200
            time.sleep(backoff * (2 ** attempt))


# =============================================================================
# 图片生成
# =============================================================================

def text_to_image_generate(api_key: str, prompt: str, model: str = "flux-kontext-pro",
                           provider: str = "wavespeed-ai", aspect_ratio: str = "16:9",
                           guidance_scale: float = 3.5, safety_tolerance: str = "5",
                           num_images: int = 1) -> str | None:
    """
    文本 → 图片（Flux Kontext 模型）。

    异步任务模式的标准示例——下面所有函数都是这个结构。
    """
    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}/text-to-image"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    seed = int(datetime.now().timestamp())
    payload = {
        "prompt": prompt,
        "num_images": num_images,
        "aspect_ratio": aspect_ratio,
        "guidance_scale": guidance_scale,
        "safety_tolerance": safety_tolerance,
        "seed": seed
    }

    # Step 1: 提交任务
    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    # Step 2-3: 轮询直到完成或失败
    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                return url
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return None
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return None


def image_to_image_generate(api_key, prompt, images, model="flux-kontext-pro",
                            provider="wavespeed-ai", aspect_ratio="16:9",
                            guidance_scale=3.5, safety_tolerance="5"):
    """
    图片 → 图片（图生图，Flux Kontext）。

    支持单图和多图两种模式：
    - 单图：POST 到 /{provider}/{model}，image 字段是单张 base64
    - 多图：POST 到 /{provider}/{model}/multi，images 字段是 base64 数组

    多图模式用于"角色一致性"场景——提供多张角色参考图，
    让 AI 生成的图片保持角色外观一致。
    """
    if isinstance(images, list):
        # ---- 多图模式 ----
        logger.info("Hello from WaveSpeedAI!")

        url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}/multi"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # 把本地图片转成 base64 data URI（API 要求）
        b64_list = []
        for image in images:
            with open(image, "rb") as f:
                img_bytes = f.read()
            b = base64.b64encode(img_bytes).decode("utf-8")
            ext = os.path.splitext(image)[1].lower()
            mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.strip(".")
            b64_list.append(f"data:image/{mime};base64,{b}")

        seed = int(datetime.now().timestamp())
        payload = {
            "guidance_scale": guidance_scale,
            "images": b64_list,
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
            "aspect_ratio": aspect_ratio,
            "seed": seed
        }

        begin = time.time()
        response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()["data"]
            request_id = result["id"]
            logger.info(f"Task submitted successfully. Request ID: {request_id}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

        url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        headers = {"Authorization": f"Bearer {api_key}"}

        while True:
            response = _get_with_retry(url, headers=headers)
            if response.status_code == 200:
                result = response.json()["data"]
                status = result["status"]

                if status == "completed":
                    end = time.time()
                    logger.info(f"Task completed in {end - begin} seconds.")
                    url = result["outputs"][0]
                    logger.info(f"Task completed. URL: {url}")
                    return url
                elif status == "failed":
                    logger.info(f"Task failed: {result.get('error')}")
                    return None
                else:
                    logger.info(f"Task still processing. Status: {status}")
            else:
                logger.info(f"Error: {response.status_code}, {response.text}")
                return None
    else:
        # ---- 单图模式 ----
        logger.info("Hello from WaveSpeedAI!")

        url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        with open(images, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {
            "guidance_scale": 3.5,
            "image": f"data:image/jpeg;base64,{b64}",
            "prompt": prompt,
            "safety_tolerance": "5"
        }

        begin = time.time()
        response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            result = response.json()["data"]
            request_id = result["id"]
            logger.info(f"Task submitted successfully. Request ID: {request_id}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

        url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
        headers = {"Authorization": f"Bearer {api_key}"}

        while True:
            response = _get_with_retry(url, headers=headers)
            if response.status_code == 200:
                result = response.json()["data"]
                status = result["status"]

                if status == "completed":
                    end = time.time()
                    logger.info(f"Task completed in {end - begin} seconds.")
                    url = result["outputs"][0]
                    logger.info(f"Task completed. URL: {url}")
                    return url
                elif status == "failed":
                    logger.info(f"Task failed: {result.get('error')}")
                    return None
                else:
                    logger.info(f"Task still processing. Status: {status}")
            else:
                logger.info(f"Error: {response.status_code}, {response.text}")
                return None


# =============================================================================
# 视频生成（核心函数，其余视频相关函数结构相同）
# =============================================================================

def text_to_video_generate(api_key, prompt, save_path: str = None,
                           model="seedance-v1-pro-t2v-480p", provider="bytedance"):
    """
    文本 → 视频。后端：ByteDance Seedance。

    与图片生成的唯一区别：完成后不仅返回 URL，还下载视频到本地。
    因为视频文件大（~几 MB），MCP 工具需要本地路径做后续处理（拼接、配乐等）。
    """
    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "aspect_ratio": "16:9",
        "duration": 5,
        "prompt": prompt,
        "seed": -1
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                # 下载视频到本地
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': "Video generated successfully."
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


def image_to_video_generate(api_key, prompt, image, save_path: str = None,
                            model="seedance-v1-pro-i2v-480p", provider="bytedance",
                            duration=5):
    """图片 → 视频。图片先转 base64 再提交。"""
    logger.info("Hello from WaveSpeedAI!")

    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with open(image, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    seed = int(datetime.now().timestamp())
    payload = {
        "duration": duration,
        "image": f"data:image/jpeg;base64,{b64}",
        "prompt": prompt,
        "seed": seed
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': "Video generated successfully."
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


def frame_to_frame_video(api_key, prompt, images, save_path: str = None,
                         model="wan-flf2v", provider="wavespeed-ai"):
    """首帧+末帧 → 过渡视频（Wan FLF2V 模型）。images[0] 是首帧，images[-1] 是末帧。"""
    logger.info("Hello from WaveSpeedAI!")

    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    with open(images[0], "rb") as f:
        img_bytes = f.read()
    first_frame_b64 = base64.b64encode(img_bytes).decode("utf-8")
    with open(images[-1], "rb") as f:
        img_bytes = f.read()
    last_frame_b64 = base64.b64encode(img_bytes).decode("utf-8")
    seed = int(datetime.now().timestamp())
    payload = {
        "duration": 5,
        "enable_safety_checker": True,
        "first_image": f"data:image/jpeg;base64,{first_frame_b64}",
        "guidance_scale": 5,
        "last_image": f"data:image/jpeg;base64,{last_frame_b64}",
        "negative_prompt": "",
        "num_inference_steps": 30,
        "prompt": prompt,
        "seed": seed,
        "size": "832*480"
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': f"{prompt} success generate video"
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Task failed: {result.get('error')}"}


# =============================================================================
# 音频生成
# =============================================================================

def audio_gen(api_key, prompt, video_url, model="mmaudio-v2", save_path=None,
              provider="wavespeed-ai", duration=5, guidance_scale=4.5,
              mask_away_clip=False, negative_prompt="", num_inference_steps=25):
    """
    为视频生成配乐/音效（MMAudio-v2 模型）。

    用 video_url 提供视觉上下文，模型根据画面内容生成匹配的音频。

    特殊处理：如果 video_url 是本地文件，先转成 base64 data URI 再提交。
    如果已经是网络 URL 则直接使用。
    """
    logger.info("Hello from WaveSpeedAI!")

    # 本地文件 → base64 data URI
    if os.path.exists(video_url):
        with open(video_url, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        ext = os.path.splitext(video_url)[1].lower()
        mime_type = "mp4" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "mp4"
        video_data_uri = f"data:video/{mime_type};base64,{video_b64}"
    else:
        video_data_uri = video_url

    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "duration": duration,
        "guidance_scale": guidance_scale,
        "mask_away_clip": mask_away_clip,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "prompt": prompt,
        "video": video_data_uri
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': f"{prompt} success generate video"
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

        time.sleep(0.5)


# =============================================================================
# 视频编辑
# =============================================================================

def runway_video_editing(api_key, prompt, video_url, aspect_ratio="16:9",
                         save_path: str = None):
    """RunwayML Gen-4：视频风格迁移/编辑。"""
    url = "https://api.wavespeed.ai/api/v3/runwayml/gen4-aleph"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    with open(video_url, "rb") as f:
        video_bytes = f.read()
    video_b64 = base64.b64encode(video_bytes).decode("utf-8")
    ext = os.path.splitext(video_url)[1].lower()
    mime_type = "mp4" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "mp4"
    video_data_uri = f"data:video/{mime_type};base64,{video_b64}"

    payload = {
        "aspect_ratio": aspect_ratio,
        "prompt": prompt,
        "video": video_data_uri
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                os.makedirs("results", exist_ok=True)
                output_filename = save_path if save_path else f"results/{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': f"{prompt} success generate video"
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


def vace_api(api_key, prompt, image_url: str = None, video_url: str = None,
             context_scale: int = 1, flow_shift: int = 16, guidance_scale: int = 5,
             duration: int = 5, num_inference_steps: int = 40, task: str = "depth",
             size: str = "1280*720", save_path: str = None):
    """
    Wan2.1-VACE：视频编辑（深度修改、姿态参考、风格迁移、重绘）。

    与 RunwayML 的不同：VACE 是开源模型，支持更细粒度的控制参数。
    task 参数决定编辑类型：depth / pose / style / repainting。
    """
    url = "https://api.wavespeed.ai/api/v3/wavespeed-ai/wan-2.1-14b-vace"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if os.path.exists(video_url):
        with open(video_url, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")
        ext = os.path.splitext(video_url)[1].lower()
        mime_type = "mp4" if ext in [".mp4", ".mov", ".avi", ".mkv"] else "mp4"
        video_data_uri = f"data:video/{mime_type};base64,{video_b64}"
    else:
        video_data_uri = video_url

    with open(image_url, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    image_b64 = f"data:image/jpeg;base64,{b64}"

    seed = int(datetime.now().timestamp())
    payload = {
        "context_scale": context_scale,
        "duration": duration,
        "flow_shift": flow_shift,
        "guidance_scale": guidance_scale,
        "images": [image_b64],
        "negative_prompt": "",
        "num_inference_steps": num_inference_steps,
        "prompt": prompt,
        "seed": seed,
        "size": size,
        "task": task,
        "video": video_data_uri if video_url else ""
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': f"{prompt} success generate video"
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


# =============================================================================
# TTS 语音合成
# =============================================================================

def speech_gen(api_key: str, prompt: str, voice_id: str = "Wise_Woman",
               emotion: str = "surprised", english_normalization: bool = False,
               pitch: int = 0, speed: float = 1.0, volume: float = 1,
               save_path=None, provider="minimax", model="speech-2.5-turbo-preview"):
    """MiniMax Speech TTS：文本 → 语音。用于给视频加旁白。"""
    url = f"https://api.wavespeed.ai/api/v3/{provider}/{model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "emotion": emotion,
        "english_normalization": english_normalization,
        "pitch": pitch,
        "speed": speed,
        "text": prompt,
        "voice_id": voice_id,
        "volume": volume
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                time_ft = datetime.now().strftime("%m%d%H%M%S")
                url_name = url.split("/")[-1]
                output_filename = save_path if save_path else f"{time_ft}_{url_name}"
                resp = _requests_session.get(url, stream=True)
                resp.raise_for_status()
                with open(output_filename, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        f.write(chunk)
                return {
                    'success': True,
                    'output_path': output_filename,
                    'message': f"{prompt[:30]} success generate video"
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

        time.sleep(0.5)


# =============================================================================
# 图片编辑（ByteDance Seedream-v4）
# =============================================================================

def seedream_v4_sequential_edit(api_key: str, prompt: str, images: list[str],
                                max_images: int = 2, size: str = "2048*2048",
                                enable_base64_output: bool = False,
                                enable_sync_mode: bool = False):
    """Seedream-v4 连续编辑：多图 + 提示词 → 生成一系列编辑后的图片。"""
    b64_list = []
    for image in images:
        if image and os.path.exists(image):
            with open(image, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            ext = os.path.splitext(image)[1].lower()
            mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.strip(".")
            b64_list.append(f"data:image/{mime};base64,{b64}")
        elif image:
            b64_list.append(image)
        else:
            b64_list.append("")

    padded_images = b64_list[:10]  # 最多 10 张

    url = "https://api.wavespeed.ai/api/v3/bytedance/seedream-v4/edit-sequential"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "enable_base64_output": enable_base64_output,
        "enable_sync_mode": enable_sync_mode,
        "images": padded_images,
        "max_images": max_images,
        "prompt": prompt,
        "size": size
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                output_url = result["outputs"]
                logger.info(f"Task completed. URL: {output_url}")
                return {
                    'success': True,
                    'output_path': output_url,
                    'message': "Image editing completed successfully."
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


def seedream_v4_edit(api_key: str, prompt: str, images: str | list[str],
                     size: str = "1024*1024", enable_base64_output: bool = False,
                     enable_sync_mode: bool = False, save_path: str = None):
    """Seedream-v4 单次编辑：图片 + 提示词 → 编辑后的图片。"""
    url = "https://api.wavespeed.ai/api/v3/bytedance/seedream-v4/edit"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if isinstance(images, str):
        images = [images]

    b64_list = []
    for image in images:
        if image and os.path.exists(image):
            with open(image, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            ext = os.path.splitext(image)[1].lower()
            mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.strip(".")
            b64_list.append(f"data:image/{mime};base64,{b64}")
        elif image:
            b64_list.append(image)

    image_list = b64_list[:10]

    payload = {
        "enable_base64_output": enable_base64_output,
        "enable_sync_mode": enable_sync_mode,
        "images": image_list,
        "prompt": prompt,
        "size": size
    }

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                url = result["outputs"][0]
                logger.info(f"Task completed. URL: {url}")
                return {
                    'success': True,
                    'output_path': url,
                    'message': "Image editing completed successfully."
                }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}


# =============================================================================
# MiniMax Hailuo 图生视频
# =============================================================================

def hailuo_i2v_pro(api_key: str, prompt: str, image: str, end_image: str = None,
                   enable_prompt_expansion: bool = True, save_path: str = None):
    """MiniMax Hailuo-02 I2V-Pro：图片 + 可选末帧 → 视频。"""
    logger.info("Hello from WaveSpeedAI!")

    if os.path.exists(image):
        with open(image, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        ext = os.path.splitext(image)[1].lower()
        mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.strip(".")
        image_data = f"data:image/{mime};base64,{b64}"
    else:
        image_data = image

    end_image_data = None
    if end_image:
        if os.path.exists(end_image):
            with open(end_image, "rb") as f:
                img_bytes = f.read()
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            ext = os.path.splitext(end_image)[1].lower()
            mime = "jpeg" if ext in [".jpg", ".jpeg"] else ext.strip(".")
            end_image_data = f"data:image/{mime};base64,{b64}"
        else:
            end_image_data = end_image

    url = "https://api.wavespeed.ai/api/v3/minimax/hailuo-02/i2v-standard"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "image": image_data,
        "prompt": prompt,
        "enable_prompt_expansion": enable_prompt_expansion
    }

    if end_image_data:
        payload["end_image"] = end_image_data

    begin = time.time()
    response = _requests_session.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()["data"]
        request_id = result["id"]
        logger.info(f"Task submitted successfully. Request ID: {request_id}")
    else:
        logger.info(f"Error: {response.status_code}, {response.text}")
        return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

    url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = _get_with_retry(url, headers=headers)
        if response.status_code == 200:
            result = response.json()["data"]
            status = result["status"]

            if status == "completed":
                end = time.time()
                logger.info(f"Task completed in {end - begin} seconds.")
                output_url = result["outputs"][0]
                logger.info(f"Task completed. URL: {output_url}")

                if save_path:
                    resp = _requests_session.get(output_url, stream=True)
                    resp.raise_for_status()
                    with open(save_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            f.write(chunk)
                    return {
                        'success': True,
                        'output_path': save_path,
                        'message': "Video generated successfully."
                    }
                else:
                    return {
                        'success': True,
                        'output_url': output_url,
                        'message': "Video generated successfully."
                    }
            elif status == "failed":
                logger.info(f"Task failed: {result.get('error')}")
                return {'success': False, 'error': f"Task failed: {result.get('error')}"}
            else:
                logger.info(f"Task still processing. Status: {status}")
        else:
            logger.info(f"Error: {response.status_code}, {response.text}")
            return {'success': False, 'error': f"Error: {response.status_code}, {response.text}"}

        time.sleep(0.5)
