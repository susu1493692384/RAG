"""
文档加载器
支持多种文档格式的加载
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
        PyMuPDFLoader,
        Docx2txtLoader,
        UnstructuredExcelLoader,
        UnstructuredPowerPointLoader,
    )
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Document = None


@dataclass
class LoadedDocument:
    """加载的文档数据类"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_type: str

    def to_langchain_document(self):
        """转换为 LangChain Document"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain 未安装")
        return Document(page_content=self.content, metadata=self.metadata)


class DocumentLoader:
    """文档加载器类"""

    # 支持的文件类型
    SUPPORTED_TYPES = {
        '.txt': 'text',
        '.md': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.doc': 'docx',
        '.xlsx': 'excel',
        '.xls': 'excel',
        '.pptx': 'pptx',
        '.ppt': 'pptx',
    }

    def __init__(self, base_path: str = None):
        """
        初始化文档加载器

        Args:
            base_path: 文档基础路径，默认为项目数据目录
        """
        if base_path is None:
            # 默认使用 enterprise_rag 目录下的 data/documents
            base_path = str(Path(__file__).parent.parent / "data" / "documents")

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def load_file(self, file_path: str) -> LoadedDocument:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            LoadedDocument 对象
        """
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()

        if file_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"不支持的文件类型: {file_type}")

        doc_type = self.SUPPORTED_TYPES[file_type]

        # 根据文件类型选择加载方式
        if doc_type == 'text':
            content, metadata = self._load_text_file(file_path)
        elif doc_type == 'pdf':
            content, metadata = self._load_pdf_file(file_path)
        elif doc_type == 'docx':
            content, metadata = self._load_docx_file(file_path)
        elif doc_type == 'excel':
            content, metadata = self._load_excel_file(file_path)
        elif doc_type == 'pptx':
            content, metadata = self._load_pptx_file(file_path)
        else:
            raise ValueError(f"暂不支持该文件类型: {file_type}")

        # 添加基础元数据
        metadata.update({
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_type,
            'doc_type': doc_type,
            'file_size': file_path.stat().st_size,
        })

        return LoadedDocument(
            content=content,
            metadata=metadata,
            source=str(file_path),
            doc_type=doc_type,
        )

    def load_directory(
        self,
        directory: Optional[str] = None,
        recursive: bool = True,
        file_pattern: str = "*.*"  # 简化：匹配所有文件（Windows 兼容）
    ) -> List[LoadedDocument]:
        """
        加载目录中的所有文档

        Args:
            directory: 目录路径，默认为 base_path
            recursive: 是否递归加载子目录
            file_pattern: 文件匹配模式

        Returns:
            LoadedDocument 对象列表
        """
        target_dir = Path(directory) if directory is not None else Path(self.base_path)

        if not target_dir.exists():
            raise FileNotFoundError(f"目录不存在: {target_dir}")

        documents = []
        files = target_dir.glob(file_pattern) if recursive else target_dir.glob("*.*")

        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_TYPES:
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")

        return documents

    def _load_text_file(self, file_path: Path) -> tuple:
        """加载文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, {}

    def _load_pdf_file(self, file_path: Path) -> tuple:
        """加载 PDF 文件"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 langchain-community 和 pymupdf")

        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()

        # 合并所有页面
        content = "\n\n".join([doc.page_content for doc in docs])
        metadata = {'total_pages': len(docs)}

        return content, metadata

    def _load_docx_file(self, file_path: Path) -> tuple:
        """加载 Word 文档"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 langchain-community 和 docx2txt")

        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])
        return content, {}

    def _load_excel_file(self, file_path: Path) -> tuple:
        """加载 Excel 文件"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 langchain-community 和 unstructured")

        loader = UnstructuredExcelLoader(str(file_path))
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])
        return content, {}

    def _load_pptx_file(self, file_path: Path) -> tuple:
        """加载 PowerPoint 文件"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 langchain-community 和 unstructured")

        loader = UnstructuredPowerPointLoader(str(file_path))
        docs = loader.load()
        content = "\n\n".join([doc.page_content for doc in docs])
        return content, {}

    def load_with_langchain(
        self,
        directory: Optional[str] = None,
        glob_pattern: str = "**/*.*"
    ) -> List:
        """
        使用 LangChain 加载文档

        Args:
            directory: 目录路径
            glob_pattern: 文件匹配模式

        Returns:
            LangChain Document 列表
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("需要安装 langchain-community")

        target_dir = directory or str(self.base_path)

        # 根据文件类型选择加载器
        loader_map = {
            '.txt': (TextLoader, {'encoding': 'utf-8'}),
            '.md': (TextLoader, {'encoding': 'utf-8'}),
        }

        # 尝试加载
        try:
            loader = DirectoryLoader(
                target_dir,
                glob=glob_pattern,
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                recursive=True,
                show_progress=True,
            )
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"加载失败: {e}")
            return []
