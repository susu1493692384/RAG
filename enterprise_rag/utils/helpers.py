"""
辅助函数
"""
import os
from typing import Dict, Any, List
from pathlib import Path


def calculate_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    估算文本的 token 数量

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        token 数量
    """
    # 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars

    estimated_tokens = chinese_chars / 1.5 + other_chars / 4

    return int(estimated_tokens)


def truncate_text(
    text: str,
    max_tokens: int = 2000,
    model: str = "gpt-3.5-turbo",
) -> str:
    """
    截断文本以适应 token 限制

    Args:
        text: 输入文本
        max_tokens: 最大 token 数量
        model: 模型名称

    Returns:
        截断后的文本
    """
    current_tokens = calculate_token_count(text, model)

    if current_tokens <= max_tokens:
        return text

    # 按比例截断
    ratio = max_tokens / current_tokens
    max_length = int(len(text) * ratio * 0.9)  # 留 10% 余量

    return text[:max_length] + "..."


def format_source_info(sources: List[Dict[str, Any]]) -> str:
    """
    格式化来源信息

    Args:
        sources: 来源列表

    Returns:
        格式化后的字符串
    """
    if not sources:
        return "无来源信息"

    formatted = []
    for i, source in enumerate(sources, 1):
        content = source.get('content', '')[:200]
        score = source.get('score', 0)
        src = source.get('source', '未知')

        formatted.append(
            f"[来源 {i}] {src}\n"
            f"相似度: {score:.2f}\n"
            f"内容: {content}...\n"
        )

    return "\n".join(formatted)


def load_env_config(
    env_file: str = ".env",
    required_keys: List[str] = None,
) -> Dict[str, str]:
    """
    加载环境变量配置

    Args:
        env_file: 环境文件路径
        required_keys: 必需的键列表

    Returns:
        环境变量字典
    """
    env_vars = {}

    # 尝试加载 .env 文件
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
                    os.environ[key.strip()] = value.strip()

    # 检查必需的键
    if required_keys:
        missing = [k for k in required_keys if k not in os.environ]
        if missing:
            raise ValueError(f"缺少必需的环境变量: {', '.join(missing)}")

    return env_vars


def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除非法字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    # Windows 非法字符
    illegal_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    sanitized = filename
    for char in illegal_chars:
        sanitized = sanitized.replace(char, '_')

    return sanitized


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    将列表分块

    Args:
        lst: 输入列表
        chunk_size: 块大小

    Returns:
        分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典

    Args:
        *dicts: 字典参数

    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误

    Args:
        a: 被除数
        b: 除数
        default: 默认值

    Returns:
        除法结果或默认值
    """
    if b == 0:
        return default
    return a / b
