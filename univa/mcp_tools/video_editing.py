import os
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from typing import Optional, List, Dict
import json

from utils.wavespeed_api import runway_video_editing, vace_api
from utils.video_process import save_last_frame_decord

from mcp_tools.base import ToolResponse, setup_logger


# Load configuration
_univa_root = Path(__file__).resolve().parents[1]
os.chdir(str(_univa_root))

# Load .env
_env_file = _univa_root.parent / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=str(_env_file), override=False)

config_path = _univa_root / "config" / "mcp_tools_config" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Override with env vars
_wavespeed_key = os.environ.get("WAVESPEED_API_KEY", "")
if _wavespeed_key:
    for _section in ["image_gen", "video_editing", "video_gen", "audio_gen"]:
        if _section in config:
            config[_section]["wavespeed_api"] = _wavespeed_key

video_editing_config = config.get('video_editing', {})

# Configure logging
log_dir = "logs"
logger = setup_logger(__name__, "logs/mcp_tools", "video_editing.log")
logger.info(f"Loaded video_editing_config: {video_editing_config}")

# Create an MCP server
mcp = FastMCP("Video_Edit_Server")



# @mcp.tool()
def swap_object_tool(
    prompt: str,
    video: str,
    image: str,
    label: str
) -> dict:
    """
    Swaps a specified object in a target video with the corresponding object from a reference image.

    This function identifies all instances of a given object class (e.g., "person", "car") in the input video and replaces them with the object provided in the reference image, guided by a textual prompt.

    Args:
        prompt (str): A textual description guiding the swapping and generation process.
        video (str): Path to the target input video file where objects will be replaced.
        image (str): Path to the reference image file containing the object to swap in.
        label (str): The class name of the object to be swapped (e.g., "person", "face", "cat").

    Returns:
        dict: A dictionary containing the operation's success status and results.
              - 'success' (bool): True if the operation completed successfully, False otherwise.
              - 'output_path' (str, optional): Path to the generated video if successful.
              - 'error' (str, optional): An error message if the operation failed.
              
    Example Usage:
        swap_anything_tool(
            prompt="a man with a beard",
            video="assets/videos/input_person.mp4",
            image="assets/images/bearded_man_face.jpg",
            label="person"
        )
    """

    mode: str = "label,salientbboxtrack",
    base_seed: int = None,
    frame_num: int = 81,
    size: str = None,
    sample_steps: int = 50,
    maskaug_mode: str = "bbox",
    maskaug_ratio: float = 0.3,
    skip_preprocess: bool = False,

    if not all([prompt, video, image, label, mode]):
        return {
            'success': False,
            'error': "Missing required arguments. 'prompt', 'video', 'image', 'label', and 'mode' are all required."
        }

    task = "swap_anything"
    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    preprocessed_files = {}
    preprocess_cmd_str = None

    if not skip_preprocess:
        pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"

        preprocess_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python", "vace/vace_preproccess.py",
            "--task", task,
            "--video", video,
            "--image", image,
            "--label", label,
            "--mode", mode,
            "--pre_save_dir", pre_save_dir
        ]

        if maskaug_mode:
            preprocess_cmd.extend(["--maskaug_mode", maskaug_mode])
        if maskaug_ratio != 0.1:
            preprocess_cmd.extend(["--maskaug_ratio", str(maskaug_ratio)])
        
        preprocess_cmd_str = ' '.join(preprocess_cmd)
        preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
        
        try:
            with open(preprocess_log, "w") as log_file:
                log_file.write(f"Executing Preprocessing Command:\n{preprocess_cmd_str}\n\n")
                result = subprocess.run(
                    preprocess_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True
                )

            processed_dir = Path(pre_save_dir)
            if processed_dir.exists():
                video_files = list(processed_dir.glob("src_video*.mp4"))
                mask_files = list(processed_dir.glob("src_mask*.mp4"))
                ref_images = list(processed_dir.glob("src_ref_image*.png"))
                
                if video_files:
                    preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                if mask_files:
                    preprocessed_files['src_mask'] = str(max(mask_files, key=os.path.getctime))
                if ref_images:
                    preprocessed_files['src_ref_images'] = [str(f) for f in sorted(ref_images)]

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_message = f"Preprocessing failed. See log for details: {preprocess_log}. Error: {str(e)}"
            return {
                'success': False,
                'error': error_message,
                'log_files': {'preprocess': preprocess_log},
                'command': {'preprocess': preprocess_cmd_str}
            }
        except Exception as e:
            return {'success': False, 'error': f"An unexpected error occurred during preprocessing: {str(e)}"}

    save_dir = f"/home/zhengyangliang/UniVideo/results/{datetime_now}"

    inference_cmd = [
        "/home/zhengyangliang/miniconda3/envs/vace/bin/python", "vace/vace_wan_inference.py",
        "--prompt", prompt,
        "--ckpt_dir", video_editing_config.get("model_path"),
        "--save_dir", save_dir,
    ]

    if 'src_video' in preprocessed_files:
        inference_cmd.extend(["--src_video", preprocessed_files['src_video']])
    if 'src_mask' in preprocessed_files:
        inference_cmd.extend(["--src_mask", preprocessed_files['src_mask']])
    if 'src_ref_images' in preprocessed_files:
        ref_images_str = ",".join(preprocessed_files['src_ref_images'])
        inference_cmd.extend(["--src_ref_images", ref_images_str])

    if base_seed is not None:
        inference_cmd.extend(["--base_seed", str(base_seed)])
    if frame_num != 81:
        inference_cmd.extend(["--frame_num", str(frame_num)])
    if size:
        inference_cmd.extend(["--size", size])
    if sample_steps != 50:
        inference_cmd.extend(["--sample_steps", str(sample_steps)])
        
    inference_cmd_str = ' '.join(inference_cmd)
    inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
    
    try:
        with open(inference_log, "w") as log_file:
            log_file.write(f"Executing Inference Command:\n{inference_cmd_str}\n\n")
            result = subprocess.run(
                inference_cmd,
                cwd="/home/zhengyangliang/VACE",
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )

        time = datetime.now().strftime("%m%d%H%M%S")
        output_path_path = os.path.join(save_dir, f"{time}_output.mp4")

        return {
            'success': True,
            'output_path': output_path_path
        }
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_message = f"Inference failed. See log for details: {inference_log}. Error: {str(e)}"
        return {
            'success': False,
            'error': error_message
        }
    except Exception as e:
        return {'success': False, 'error': f"An unexpected error occurred during inference: {str(e)}"}

@mcp.tool()
def depth_modify(
    prompt: str,
    video: str
) -> dict:
    """
    Based on a text prompt, use depth information to edit or replace the foreground or background of a video.

    This function is specifically designed for video editing tasks that require distinguishing between foreground and background. It is suitable for intelligent video editing, such as replacing the background or changing the foreground color, while leaving the other content unchanged.

    Args:
        prompt (str): Text describing the video editing instructions. This is key to instructing the AI on how to modify the video.
        For example: "Change the background to a snowy mountain" or "Keep the characters the same, but change the background to a cyberpunk style."
        video (str): The path to the original video file to be edited. For example: 'data/input_video.mp4'.

    Returns:
        dict: A dictionary containing the result of the operation.
        - If successful, the format is: {'success': True, 'output_path': 'path/to/output.mp4'}
        - If failed, the format may be: {'success': False, 'error': 'error message', 'log_file': 'path/to/log.log'}
    """
    model = video_editing_config.get("style_transfer")
    api_key = video_editing_config.get("wavespeed_api")

    if model == "vace":

        task = "depth"
        
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        preprocessed_files = {}
        pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"
        preprocess_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_preproccess.py",
            "--task", task,
            "--video", video,
            "--pre_save_dir", pre_save_dir
        ]
            
        # Execute preprocessing
        preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
        try:
            with open(preprocess_log, "w") as log_file:
                log_file.write("Preprocessing command: " + ' '.join(preprocess_cmd) + "\n")
                result = subprocess.run(
                    preprocess_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Preprocessing failed with return code {result.returncode}",
                    'log_file': preprocess_log
                }
                    
            # Find preprocessed files in ./processed/
            # processed_dir = Path("/home/zhengyangliang/VACE/processed")
            processed_dir = Path(pre_save_dir)
            if processed_dir.exists():
                # Get latest files
                video_files = list(processed_dir.glob("src_video*.mp4"))
                
                if video_files:
                    preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                        
        except Exception as e:
            return {
                'success': False,
                'error': f"Preprocessing error: {str(e)}"
            }
        
        # Step 2: Inference
        if not prompt:
            return {
                'success': False,
                'error': "Prompt is required for inference"
            }
        
        # Build inference command
        inference_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_wan_inference.py",
            "--prompt", prompt,
            "--src_video", preprocessed_files['src_video'],
            "--ckpt_dir", video_editing_config.get("model_path"),
            "--save_dir", f"/home/zhengyangliang/UniVideo/results/{datetime_now}",
        ]
        
        # Execute inference
        inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
        try:
            with open(inference_log, "w") as log_file:
                result = subprocess.run(
                    inference_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': f"results/{datetime_now}/out_video.mp4"
                }
            else:
                return {
                    'success': False,
                    'error': f"Inference failed with return code {result.returncode}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Inference error: {str(e)}"
            }
    else:
        res = runway_video_editing(api_key, prompt, video)

        return res


# @mcp.tool()
def recolor(
    prompt: str,
    video: str
) -> dict:
    """
    Recolorize a video or modify the color of specific areas based on a text prompt, allowing for overall stylistic recoloring.

    Args:
        prompt (str): Text describing the color editing instructions. This is key to instructing the AI on how to colorize.
        For example: "Turn her skirt red" or "Turn the entire scene into a retro sepia tone."
        video (str): The path to the original video file to be edited. For example: 'data/input_video.mp4'.

    Returns:
        dict: A dictionary containing the results of the operation.
        - If successful, the format is: {'success': True, 'output_path': 'path/to/output.mp4'}
        - If failed, the format may be:
        {'success': False, 'error': 'error message', 'log_file': 'path/to/log.log'}
    """

    task = "gray"
    
    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    preprocessed_files = {}
    pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"
    preprocess_cmd = [
        "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
        "vace/vace_preproccess.py",
        "--task", task,
        "--video", video,
        "--pre_save_dir", pre_save_dir
    ]
        
    # Execute preprocessing
    preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
    try:
        with open(preprocess_log, "w") as log_file:
            log_file.write("Preprocessing command: " + ' '.join(preprocess_cmd) + "\n")
            result = subprocess.run(
                preprocess_cmd,
                cwd="/home/zhengyangliang/VACE",
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
        if result.returncode != 0:
            return {
                'success': False,
                'error': f"Preprocessing failed with return code {result.returncode}",
                'log_file': preprocess_log
            }
                
        # Find preprocessed files in ./processed/
        # processed_dir = Path("/home/zhengyangliang/VACE/processed")
        processed_dir = Path(pre_save_dir)
        if processed_dir.exists():
            # Get latest files
            video_files = list(processed_dir.glob("src_video*.mp4"))
            
            if video_files:
                preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                    
    except Exception as e:
        return {
            'success': False,
            'error': f"Preprocessing error: {str(e)}"
        }
    
    # Step 2: Inference
    if not prompt:
        return {
            'success': False,
            'error': "Prompt is required for inference"
        }
    
    # Build inference command
    inference_cmd = [
        "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
        "vace/vace_wan_inference.py",
        "--prompt", prompt,
        "--src_video", preprocessed_files['src_video'],
        "--ckpt_dir", video_editing_config.get("model_path"),
        "--save_dir", f"/home/zhengyangliang/UniVideo/results/{datetime_now}",
    ]
    
    # Execute inference
    inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
    try:
        with open(inference_log, "w") as log_file:
            result = subprocess.run(
                inference_cmd,
                cwd="/home/zhengyangliang/VACE",
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
        if result.returncode == 0:
            return {
                'success': True,
                'output_path': f"results/{datetime_now}/out_video.mp4"
            }
        else:
            return {
                'success': False,
                'error': f"Inference failed with return code {result.returncode}"
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Inference error: {str(e)}"
        }

@mcp.tool()
def pose_reference(
    prompt: str,
    image: str=None,
    video: str=None
) -> dict:
    """
    Based on a text prompt, transfer the motions of a person in a video to a new character while preserving the original motion sequence. This function implements the pose transfer functionality, resulting in a new character performing the motions from the old video.

    Args:
        prompt (str): Describes the look and style of the new character you want to generate. The AI will apply the motions from the original video to this new character.
        For example: "A dancing astronaut", "A walking stormtrooper".
        image (str): The path to the source image file providing the pose reference. The image should contain one or more people whose poses will be extracted. For example: 'data/dancing_person.mp4'.
        video (str): The path to the source video file providing the motion reference. The video should contain one or more people whose poses will be extracted. For example: 'data/dancing_person.mp4'.

    Returns:
        dict: A dictionary containing the results of the operation.
        - If successful, the format is: {'success': True, 'output_path': 'path/to/output.mp4'}
        - If failed, the format may be: {'success': False, 'error': 'error message', 'log_file': 'path/to/log.log'}
    """
    model = video_editing_config.get("pose_reference")
    api_key = video_editing_config.get("wavespeed_api")


    if model == "vace":
        task = "pose"
        
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        preprocessed_files = {}
        pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"
        preprocess_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_preproccess.py",
            "--task", task,
            "--video", video,
            "--pre_save_dir", pre_save_dir
        ]
            
        # Execute preprocessing
        preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
        try:
            with open(preprocess_log, "w") as log_file:
                log_file.write("Preprocessing command: " + ' '.join(preprocess_cmd) + "\n")
                result = subprocess.run(
                    preprocess_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Preprocessing failed with return code {result.returncode}",
                    'log_file': preprocess_log
                }
                    
            # Find preprocessed files in ./processed/
            # processed_dir = Path("/home/zhengyangliang/VACE/processed")
            processed_dir = Path(pre_save_dir)
            if processed_dir.exists():
                # Get latest files
                video_files = list(processed_dir.glob("src_video*.mp4"))
                
                if video_files:
                    preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                        
        except Exception as e:
            return {
                'success': False,
                'error': f"Preprocessing error: {str(e)}"
            }
        
        # Step 2: Inference
        if not prompt:
            return {
                'success': False,
                'error': "Prompt is required for inference"
            }
        
        # Build inference command
        inference_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_wan_inference.py",
            "--prompt", prompt,
            "--src_video", preprocessed_files['src_video'],
            "--ckpt_dir", video_editing_config.get("model_path"),
            "--save_dir", f"/home/zhengyangliang/UniVideo/results/{datetime_now}",
        ]
        
        # Execute inference
        inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
        try:
            with open(inference_log, "w") as log_file:
                result = subprocess.run(
                    inference_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': f"results/{datetime_now}/out_video.mp4"
                }
            else:
                return {
                    'success': False,
                    'error': f"Inference failed with return code {result.returncode}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Inference error: {str(e)}"
            }
    else:
        res = vace_api(api_key, prompt, image, video, task="pose")

        return res


@mcp.tool()
def style_transfer(
    prompt: str,
    video: str
) -> dict:
    """
    Based on a text prompt, converts a video into a specified artistic style, achieving style transfer.

    This function extracts edges and contours to generate a line drawing video. This process retains the core structure and dynamic information of the original video, but removes its original colors and textures, providing an ideal structural foundation for applying the new style. It then "renders" the line drawing video, generating a video with the same content and dynamics as the original video, but with a completely new visual style.

    Args:
        prompt (str): Text describing the target artistic style. This is key to guiding the AI on which style to apply.
        For example: "Van Gogh Starry Night Style", "Transformed into a Watercolor Animation", "Cyberpunk City".
        video (str): The path to the original video file to be style transferred. For example: 'data/input_video.mp4'.

    Returns:
        dict: A dictionary containing the results of the operation.
        - If successful, the format is: {'success': True, 'output_path': 'path/to/output.mp4'}
        - If failed, the format may be: 
        {'success': False, 'error': 'error message', 'log_file': 'path/to/log.log'} 
    """
    model = video_editing_config.get("style_transfer")
    api_key = video_editing_config.get("wavespeed_api")

    if model == "vace":
        task = "scribble"
        
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        preprocessed_files = {}
        pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"
        preprocess_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_preproccess.py",
            "--task", task,
            "--video", video,
            "--pre_save_dir", pre_save_dir
        ]
            
        # Execute preprocessing
        preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
        try:
            with open(preprocess_log, "w") as log_file:
                log_file.write("Preprocessing command: " + ' '.join(preprocess_cmd) + "\n")
                result = subprocess.run(
                    preprocess_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Preprocessing failed with return code {result.returncode}",
                    'log_file': preprocess_log
                }
                    
            # Find preprocessed files in ./processed/
            # processed_dir = Path("/home/zhengyangliang/VACE/processed")
            processed_dir = Path(pre_save_dir)
            if processed_dir.exists():
                # Get latest files
                video_files = list(processed_dir.glob("src_video*.mp4"))
                
                if video_files:
                    preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                        
        except Exception as e:
            return {
                'success': False,
                'error': f"Preprocessing error: {str(e)}"
            }
        
        # Step 2: Inference
        if not prompt:
            return {
                'success': False,
                'error': "Prompt is required for inference"
            }
        
        # Build inference command
        inference_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_wan_inference.py",
            "--prompt", prompt,
            "--src_video", preprocessed_files['src_video'],
            "--ckpt_dir", video_editing_config.get("model_path"),
            "--save_dir", f"/home/zhengyangliang/UniVideo/results/{datetime_now}",
        ]
        
        # Execute inference
        inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
        try:
            with open(inference_log, "w") as log_file:
                result = subprocess.run(
                    inference_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': f"results/{datetime_now}/out_video.mp4"
                }
            else:
                return {
                    'success': False,
                    'error': f"Inference failed with return code {result.returncode}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Inference error: {str(e)}"
            }
    elif model in ("vace_api", "vace-wavespeed"):
        # 云端 VACE 风格迁移（走 Wavespeed，已验证可用）
        tmp_dir = Path("eval/tmp/vace")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        first_frame = save_last_frame_decord(video, str(tmp_dir / "first_frame.png"))
        res = vace_api(
            api_key=api_key,
            prompt=prompt,
            image_url=first_frame,   # VACE 强制需要参考图
            video_url=video,
            task="inpainting",
            duration=5,
            save_path=str(tmp_dir / "style_transfer.mp4")
        )
        return res
    else:
        res = runway_video_editing(api_key, prompt, video)

        return res


@mcp.tool()
def repainting(
    prompt: str,
    video: str,
    label: str=None
) -> dict:
    """
    Partially repaint or replace a specific object in a video, changing its appearance or transforming it into something entirely new based on a text prompt.

    This function implements video inpainting or object replacement by first calling the `label` parameter to identify and locate a specific object in the video (e.g., "cat," "car"). The script generates a precise dynamic mask for the identified object while preserving the original video. This mask marks the area to be edited. Next, it fills in the content described in `prompt`, modifying or completely replacing the original object and ensuring that the new content blends seamlessly with the rest of the video.

    Args:
        prompt (str): Text describing what new content you want to generate in the specified area. This is key to guiding the AI's creation. Examples: "A cat in armor," "A futuristic flying car."
        video (str): The path to the original video file to be edited. Example: 'data/cat_video.mp4'.
        label (str): A text label identifying the target object to be replaced or modified in the video. Example: "cat," "cat", "dog".

    Returns:
    dict: A dictionary containing the results of the operation.
        - If successful, the format may be: {'success': True, 'output_path': 'path/to/output.mp4'}
        - If failed, the format may be:
        {'success': False, 'error': 'error message', 'log_file': 'path/to/log.log'}
    """
    model = video_editing_config.get("repainting")
    api_key = video_editing_config.get("wavespeed_api")

    if model == "vace":
        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Preprocessing (if needed)
        preprocessed_files = {}
        task = "inpainting"
        # Build preprocessing command
        pre_save_dir = f"/home/zhengyangliang/UniVideo/temp/{datetime_now}"
        # python vace/vace_preproccess.py --task inpainting --mode label --label cat --video assets/videos/test.mp4
        preprocess_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_preproccess.py",
            "--task", task,
            "--video", video,
            "--mode", "label",
            "--label", label,
            "--pre_save_dir", pre_save_dir
        ]
        
        # Execute preprocessing
        preprocess_log = os.path.join(log_dir, f"vace_preprocess_{task}_{datetime_now}.log")
        try:
            with open(preprocess_log, "w") as log_file:
                log_file.write("Preprocessing command: " + ' '.join(preprocess_cmd) + "\n")
                result = subprocess.run(
                    preprocess_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Preprocessing failed with return code {result.returncode}"
                }
                
            # Find preprocessed files in ./processed/
            # processed_dir = Path("/home/zhengyangliang/VACE/processed")
            processed_dir = Path(pre_save_dir)
            if processed_dir.exists():
                # Get latest files
                video_files = list(processed_dir.glob("src_video*.mp4"))
                mask_files = list(processed_dir.glob("src_mask*.mp4"))
                
                if video_files:
                    preprocessed_files['src_video'] = str(max(video_files, key=os.path.getctime))
                if mask_files:
                    preprocessed_files['src_mask'] = str(max(mask_files, key=os.path.getctime))
                    
        except Exception as e:
            return {
                'success': False,
                'error': f"Preprocessing error: {str(e)}"
            }
        
        # Step 2: Inference
        if not prompt:
            return {
                'success': False,
                'error': "Prompt is required for inference"
            }
        
        # Build inference command
        inference_cmd = [
            "/home/zhengyangliang/miniconda3/envs/vace/bin/python",
            "vace/vace_wan_inference.py",
            "--prompt", prompt,
            "--ckpt_dir", video_editing_config.get("model_path"),
            "--save_dir", f"/home/zhengyangliang/UniVideo/results/{datetime_now}",
        ]

        # Add preprocessed files
        if 'src_video' in preprocessed_files:
            inference_cmd.extend(["--src_video", preprocessed_files['src_video']])
        if 'src_mask' in preprocessed_files:
            inference_cmd.extend(["--src_mask", preprocessed_files['src_mask']])
        
        # Execute inference
        inference_log = os.path.join(log_dir, f"vace_inference_{task}_{datetime_now}.log")
        try:
            with open(inference_log, "w") as log_file:
                result = subprocess.run(
                    inference_cmd,
                    cwd="/home/zhengyangliang/VACE",
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
            if result.returncode == 0:
                return {
                    'success': True,
                    'output_path': f"results/{datetime_now}/out_video.mp4"
                }
            else:
                return {
                    'success': False,
                    'error': f"Inference failed with return code {result.returncode}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Inference error: {str(e)}"
            }
    else:
        res = runway_video_editing(api_key, prompt, video)

        return res




@mcp.tool()
def long_video_edit(
    video: str,
    edit_prompt: str,
    task: str = "style",
    label: Optional[str] = None,
    max_clips: Optional[int] = None,
    clip_granularity_s: float = 5.0,
) -> dict:
    """
    编辑任意长度的长视频（分片 → 逐片云端 VACE 编辑 → 合并）。

    云端 VACE 单次只能处理约 5 秒视频，因此把长视频切成多个 ≤5s 片段，
    每片用 vace_api 独立编辑，最后用 ffmpeg 拼回完整视频。
    失败的片段用原始片段顶替，保证输出完整时长。

    Args:
        video (str): 长视频路径。
        edit_prompt (str): 编辑指令，例如 "Turn the skeleton man into a fishman"。
        task (str, optional): 编辑类型。'style'（风格迁移/重绘）或 'pose'（姿态迁移）。默认 'style'。
        label (Optional[str], optional): 目标物体类别（当前云端 VACE 未使用，保留兼容）。
        max_clips (Optional[int], optional): 最大编辑片数。设置后从视频中均匀抽取 max_clips 个
            5 秒窗口编辑，其余部分保留原片段——适合超长视频控制成本（如 10 分钟视频只编辑 6 片）。
        clip_granularity_s (float, optional): 每片时长（秒），VACE 上限 5 秒。默认 5.0。

    Returns:
        dict: {'success': bool, 'output_path': str, 'edited_clips': int, 'failed_clips': int}
    """
    api_key = video_editing_config.get("wavespeed_api")
    _log_dir = "logs"
    os.makedirs(_log_dir, exist_ok=True)

    try:
        if not os.path.exists(video):
            return {'success': False, 'error': f"Video not found: {video}"}
        if not edit_prompt:
            return {'success': False, 'error': "Parameter 'edit_prompt' cannot be empty."}

        task = (task or "style").lower().strip()
        vace_task = "pose" if task in ("pose", "pose_reference") else "inpainting"

        datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(f"results/long_video_edit/{datetime_now}")
        save_dir.mkdir(parents=True, exist_ok=True)
        inference_log = os.path.join(_log_dir, f"vace_inference_long_{datetime_now}.log")

        # ffmpeg 路径：优先 imageio-ffmpeg 自带二进制，找不到再退回 PATH
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_bin = get_ffmpeg_exe()
        except ImportError:
            ffmpeg_bin = "ffmpeg"
        logger.info(f"Using ffmpeg: {ffmpeg_bin}")

        # ------------ 探测视频时长 ------------
        def _probe_duration(path: str) -> float:
            r = subprocess.run([ffmpeg_bin, "-i", path],
                               capture_output=True, text=True)
            import re
            m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
            if not m:
                raise ValueError(f"无法探测视频时长: {path}")
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))

        duration = _probe_duration(video)
        logger.info(f"Video duration: {duration:.1f}s")

        # ------------ 计算编辑窗口 ------------
        clip_s = min(clip_granularity_s, 5.0)
        if max_clips:
            # 抽样：把视频均分 max_clips 段，每段取 clip_s 秒窗口（对齐段中心）
            seg_len = duration / max_clips
            windows = []
            for i in range(max_clips):
                center = (i + 0.5) * seg_len
                start = max(0.0, center - clip_s / 2)
                end = min(duration, center + clip_s / 2)
                if end - start >= 1.0:
                    windows.append((start, end))
        else:
            # 全覆盖：从头到尾切分
            windows = [(i * clip_s, min(duration, (i + 1) * clip_s))
                       for i in range(int(duration // clip_s) + 1)
                       if min(duration, (i + 1) * clip_s) - i * clip_s >= 1.0]

        # 构建时间轴计划：[(start, end, 是否编辑)]，未编辑段用原片剪切，
        # 保证输出时长 = 原视频完整时长（抽样模式不会丢中间内容）
        plan = []
        prev = 0.0
        for ws, we in windows:
            if ws > prev + 0.5:
                plan.append((prev, ws, False))
            plan.append((ws, we, True))
            prev = we
        if duration - prev > 0.5:
            plan.append((prev, duration, False))

        logger.info(f"Editing {len(windows)} clips (task={vace_task}, max_clips={max_clips})")

        def _ffmpeg_cut(start: float, end: float) -> str:
            out_path = str(save_dir / f"clip_{start:.2f}_{end:.2f}.mp4")
            cmd = [ffmpeg_bin, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                   "-i", video, "-map", "0:v:0", "-c:v", "libx264",
                   "-preset", "fast", "-crf", "18", "-an", out_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_path

        # ------------ 按时间轴逐段处理 ------------
        edited_paths: List[str] = []
        failed = 0
        for i, (start_s, end_s, do_edit) in enumerate(plan):
            clip_path = _ffmpeg_cut(start_s, end_s)
            if not do_edit:
                # 未编辑段：原片剪切，直接入拼接列表
                edited_paths.append(os.path.abspath(clip_path))
                continue

            first_frame = str(save_dir / f"frame_{i}.png")
            save_last_frame_decord(clip_path, first_frame)

            try:
                res = vace_api(api_key, edit_prompt, image_url=first_frame,
                               video_url=clip_path, task=vace_task,
                               duration=max(5, int(round(end_s - start_s))),  # VACE 要求 ≥5s
                               save_path=str(save_dir / f"edited_{i}.mp4"))
                with open(inference_log, "a") as log_file:
                    log_file.write(f"[{start_s:.1f}-{end_s:.1f}s] {json.dumps(res, ensure_ascii=False)}\n")
                if isinstance(res, dict) and res.get("success") and res.get("output_path") \
                        and os.path.exists(res["output_path"]):
                    edited_paths.append(os.path.abspath(res["output_path"]))
                else:
                    # 编辑失败：用原片顶替，保证完整性
                    logger.warning(f"Clip {i} edit failed, using original: {res}")
                    edited_paths.append(os.path.abspath(clip_path))
                    failed += 1
            except Exception as e:
                logger.error(f"Clip {i} edit error: {e}")
                edited_paths.append(os.path.abspath(clip_path))
                failed += 1

        # ------------ 合并 ------------
        concat_list = save_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in edited_paths:
                f.write(f"file '{p.replace(chr(92), chr(92) * 2)}'\n")

        time_tag = datetime.now().strftime("%m%d%H%M%S")
        output_path = str(save_dir / f"{time_tag}_output.mp4")

        try:
            subprocess.run([ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                            "-i", str(concat_list), "-c", "copy", output_path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            subprocess.run([ffmpeg_bin, "-y", "-f", "concat", "-safe", "0",
                            "-i", str(concat_list), "-c:v", "libx264",
                            "-preset", "fast", "-crf", "18", "-an", output_path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return {
            'success': True,
            'output_path': output_path,
            'edited_clips': len(windows) - failed,
            'failed_clips': failed,
        }

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {'success': False, 'error': f"Long video edit failed: {e}"}
    except Exception as e:
        return {'success': False, 'error': f"An unexpected error occurred: {str(e)}"}




if __name__ == "__main__":
    mcp.run(transport="stdio")
    