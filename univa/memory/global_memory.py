"""
Global Memory：领域知识记忆（论文三层记忆之一）
==============================================
从知识库检索与当前请求最相关的领域经验，注入规划上下文，
让 Agent 借助历史/专家知识做更合理的计划。

实现：TF-IDF 向量化 + 余弦相似度检索（sklearn，本地零下载）。
知识库：prompts/knowledge/*.md，每个 `## 标题` 块作为一条知识。

消融用法：
  memory_cfg 含 "global" → 检索注入；不含则不注入。
"""
import os
import re
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


def _load_knowledge_blocks(knowledge_dir: str) -> List[dict]:
    """扫描 knowledge/*.md，把每个 `## 标题` 段拆成一条 {title, text}。"""
    blocks = []
    if not os.path.isdir(knowledge_dir):
        return blocks
    for md in sorted(Path(knowledge_dir).glob("*.md")):
        text = md.read_text(encoding="utf-8")
        # 按 "## " 二级标题切块
        parts = re.split(r"(?m)^##\s+", text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            lines = part.splitlines()
            title = lines[0].strip()
            if title.startswith("# "):
                continue  # 一级标题（文件前言）不是知识条目，跳过
            body = "\n".join(lines[1:]).strip()
            if body:
                blocks.append({"title": title, "text": body})
    return blocks


class GlobalMemory:
    def __init__(self, knowledge_dir: str = None, top_k: int = 3):
        self.top_k = top_k
        if knowledge_dir is None:
            knowledge_dir = str(
                Path(__file__).resolve().parents[1] / "prompts" / "knowledge"
            )
        self.blocks = _load_knowledge_blocks(knowledge_dir)
        self._vectorizer = None
        self._tfidf = None
        if self.blocks:
            self._build_index()

    def _build_index(self):
        """对整个知识库做 TF-IDF 向量化，建检索索引。

        用字符级 2-4 gram（analyzer='char_wb'）：中文知识库无需分词，
        且能抓住"长视频""分片"这类连续子串的关键信息。
        """
        texts = [f"{b['title']} {b['text']}" for b in self.blocks]
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=True
        )
        self._tfidf = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = None) -> str:
        """检索与 query 最相关的 top_k 条知识，拼成注入文本。

        知识库为空或向量化失败时返回空串（不影响主流程）。
        """
        if not self.blocks or self._vectorizer is None:
            return ""
        k = top_k or self.top_k
        try:
            q_vec = self._vectorizer.transform([query])
            scores = (self._tfidf @ q_vec.T).toarray().ravel()
            top_idx = scores.argsort()[::-1][:k]
        except Exception:
            return ""

        parts = []
        for i in top_idx:
            b = self.blocks[i]
            parts.append(f"【{b['title']}】{b['text']}")
        return "以下是相关的领域经验，规划时请参考：\n" + "\n\n".join(parts)

    def count(self) -> int:
        return len(self.blocks)
