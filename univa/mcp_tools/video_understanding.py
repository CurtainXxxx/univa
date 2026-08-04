import os
import yaml
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from mcp_tools.base import ToolResponse, setup_logger
from utils.query_llm import prepare_multimodal_messages_openai_format, query_openrouter, multimodal_query


# NOTE: Heavy local-model deps (torch, decord, ultralytics/SAM, video_tracking)
# are intentionally NOT imported at module level. vision2text_gen is API-only
# (multimodal LLM call); importing torch/ultralytics here would crash the MCP
# server on machines without a local GPU/torch install. Import them lazily
# inside functions that actually need local models, if any are added later.


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
if os.environ.get("VIDEO_UNDERSTAND_MODEL_PATH"):
    config.setdefault("video_understanding", {})["model_path"] = os.environ["VIDEO_UNDERSTAND_MODEL_PATH"]
if os.environ.get("VIDEO_RETRIEVER_MODEL_PATH"):
    config.setdefault("video_understanding", {})["retriever_model_path"] = os.environ["VIDEO_RETRIEVER_MODEL_PATH"]

video_understanding_config = config.get('video_understanding', {})

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "video_understanding.log")
logger.info(f"Loaded video_understanding_config: {video_understanding_config}")

# Create an MCP server
mcp = FastMCP("Video_Understanding_Server")


@mcp.tool()
def vision2text_gen(prompt: str, multimodal_path: str, type: str) -> dict:
    """
    Analyzes and describes the content of a video or image based on a given prompt, converting visual information into text.
    This tool is useful for understanding ambiguous or complex visual inputs, providing detailed textual descriptions of the content.

    Args:
        prompt (str): User's instruction.
        multimodal_path (str): The path of the video or image.
        type (str): The type of the multimodal input, either "video" or "image".

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the vision content was understood successfully, False otherwise.
              - 'message' (str, optional): The details of the vision content if successful.
              - 'error' (str, optional): An error message if the operation failed.
    """
    try:
        if type == "video":
            content = multimodal_query(prompt, video_path=multimodal_path)
        elif type == "image":
            content = multimodal_query(prompt, image_path=multimodal_path)
        else:
            return ToolResponse(
                success=False,
                message="The type of the multimodal input should be either 'video' or 'image'."
            )

        return ToolResponse(
            success=True,
            message="Vision content understood successfully.",
            content=content
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            message=f"An error occurred: {str(e)}"
        )



if __name__ == "__main__":
    mcp.run(transport="stdio")

