"""
User Memory：用户偏好记忆（论文三层记忆之一）
============================================
存储用户级偏好（如"喜欢电影感、暖色调"），在规划时注入提示词，
让 Agent 生成的计划更贴合用户习惯。

存储：SQLite 表 user_preferences(user_id, key, value)
注入：to_prompt() 把该用户所有偏好拼成一段文本，由 generate_plan 拼接进上下文。

消融用法：
  memory_cfg 含 "user" → 注入用户偏好；不含则不注入。
"""
import sqlite3
from typing import Dict, List


class UserMemory:
    def __init__(self, db_path: str = "memory_user.db"):
        """db_path: SQLite 文件路径（独立于 Task 记忆库，便于消融隔离）。"""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_preferences ("
            "user_id TEXT, key TEXT, value TEXT, "
            "PRIMARY KEY (user_id, key))"
        )
        conn.commit()
        conn.close()

    def set(self, user_id: str, key: str, value: str):
        """写入/覆盖一条用户偏好。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO user_preferences (user_id, key, value) VALUES (?,?,?)",
            (user_id, key, value),
        )
        conn.commit()
        conn.close()

    def get_all(self, user_id: str) -> List[Dict[str, str]]:
        """取回该用户全部偏好。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT key, value FROM user_preferences WHERE user_id=?",
            (user_id,),
        ).fetchall()
        conn.close()
        return [{"key": k, "value": v} for k, v in rows]

    def to_prompt(self, user_id: str) -> str:
        """把用户偏好拼成注入文本；无偏好返回空串。"""
        prefs = self.get_all(user_id)
        if not prefs:
            return ""
        lines = [f"- {p['key']}: {p['value']}" for p in prefs]
        return "请遵循以下用户偏好：\n" + "\n".join(lines)

    def seed_default(self, user_id: str):
        """写入一组固定用户偏好（消融实验用固定种子，保证可复现）。"""
        defaults = {
            "visual_style": "偏好电影感光影与暖色调",
            "prompt_language": "中文请求优先，生成 prompt 用英文更稳定",
            "video_length": "短视频优先，长内容先规划分镜",
        }
        for k, v in defaults.items():
            self.set(user_id, k, v)
