"""
文本分块器
将长文档分割成适合处理的小块
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    """文本块数据类"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: int
    token_count: Optional[int] = None


class TextSplitter:
    """文本分块器类"""

    # 默认分隔符
    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        ". ",
        "! ",
        "? ",
        "; ",
        ", ",
        " ",
        ""
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,
        keep_separator: bool = False,
    ):
        """
        初始化文本分块器

        Args:
            chunk_size: 最大块大小（字符数或 token 数）
            chunk_overlap: 块之间的重叠大小
            separators: 分隔符列表
            length_function: 计算长度的函数
            keep_separator: 是否保留分隔符
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.length_function = length_function or len
        self.keep_separator = keep_separator

    def split_text(self, text: str, metadata: Optional[Dict] = None) -> List[TextChunk]:
        """
        分割文本

        Args:
            text: 待分割的文本
            metadata: 元数据（会附加到每个块）

        Returns:
            TextChunk 列表
        """
        if metadata is None:
            metadata = {}

        chunks = []
        remaining_text = text
        chunk_id = 0

        while remaining_text:
            # 找到合适的分割点
            if self.length_function(remaining_text) <= self.chunk_size:
                chunk_text = remaining_text
                remaining_text = ""
            else:
                # 尝试按分隔符分割
                chunk_text = self._find_split_point(remaining_text)
                remaining_text = remaining_text[self.chunk_size - self.chunk_overlap:]

            # 创建块
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_size': self.length_function(chunk_text),
            })

            chunks.append(TextChunk(
                content=chunk_text.strip(),
                metadata=chunk_metadata,
                chunk_id=chunk_id,
            ))

            chunk_id += 1

        return chunks

    def split_documents(
        self,
        documents: List[Any],
        source_field: str = "page_content"
    ) -> List[TextChunk]:
        """
        批量分割文档

        Args:
            documents: 文档列表（支持 LangChain Document 或自定义格式）
            source_field: 内容字段名

        Returns:
            TextChunk 列表
        """
        all_chunks = []
        global_chunk_id = 0

        for doc in documents:
            # 提取文本和元数据
            if hasattr(doc, 'page_content'):
                text = doc.page_content
                metadata = doc.metadata
            elif isinstance(doc, dict):
                text = doc.get(source_field, "")
                metadata = {k: v for k, v in doc.items() if k != source_field}
                # 如果 metadata 中有一个 'metadata' 键，则展开它
                if 'metadata' in metadata and isinstance(metadata['metadata'], dict):
                    inner_metadata = metadata.pop('metadata')
                    metadata.update(inner_metadata)
            else:
                raise ValueError(f"不支持的文档类型: {type(doc)}")

            # 分割
            chunks = self.split_text(text, metadata)

            # 更新 chunk_id
            for chunk in chunks:
                chunk.chunk_id = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(chunks)

        return all_chunks

    def _find_split_point(self, text: str) -> str:
        """找到合适的分割点"""
        # 计算目标位置
        target_size = self.chunk_size

        # 在目标位置附近查找分隔符
        for separator in self.separators:
            # 从目标位置向前搜索
            split_pos = text.rfind(separator, 0, target_size + len(separator))
            if split_pos != -1:
                if self.keep_separator:
                    return text[:split_pos + len(separator)]
                return text[:split_pos]

        # 没找到分隔符，强制分割
        return text[:target_size]

    @staticmethod
    def merge_small_chunks(
        chunks: List[TextChunk],
        min_size: int = 100
    ) -> List[TextChunk]:
        """
        合并过小的块

        Args:
            chunks: TextChunk 列表
            min_size: 最小块大小

        Returns:
            合并后的 TextChunk 列表
        """
        if not chunks:
            return []

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            current_size = len(current_chunk.content)
            next_size = len(next_chunk.content)

            # 如果当前块太小，合并
            if current_size < min_size:
                current_chunk.content += "\n\n" + next_chunk.content
                current_chunk.metadata['merged'] = True
                current_chunk.metadata['chunk_size'] = len(current_chunk.content)
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)

        return merged

    @staticmethod
    def create_strategies():
        """创建预定义的分块策略"""
        return {
            'default': TextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            ),
            'large': TextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
            ),
            'small': TextSplitter(
                chunk_size=200,
                chunk_overlap=20,
            ),
            'code': TextSplitter(
                chunk_size=300,
                chunk_overlap=0,
                separators=["\nclass ", "\ndef ", "\n    ", "\n"],
            ),
            'markdown': TextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n"],
            ),
        }


# 兼容 LangChain 的包装器
try:
    # 新版本 LangChain (0.1.0+)
    from langchain_text_splitters import RecursiveCharacterTextSplitter as LangchainSplitter

    class LangchainTextSplitterWrapper:
        """LangChain 文本分块器包装器"""

        def __init__(self, **kwargs):
            self.splitter = LangchainSplitter(**kwargs)

        def split_documents(self, documents):
            """分割文档"""
            return self.splitter.split_documents(documents)

        def split_text(self, text):
            """分割文本"""
            return self.splitter.split_text(text)

except ImportError:
    LangchainTextSplitterWrapper = None
