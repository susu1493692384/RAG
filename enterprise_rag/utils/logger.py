"""
日志工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "./logs",
) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件名（可选）
        log_dir: 日志目录

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 禁止日志传播到根 logger（避免重复输出）
    logger.propagate = False

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_path = log_path / log_file
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_logging(
    level: int = logging.INFO,
    log_dir: str = "./logs",
):
    """
    设置全局日志配置

    Args:
        level: 日志级别
        log_dir: 日志目录
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 日志文件名（按日期）
    log_file = f"rag_{datetime.now().strftime('%Y%m%d')}.log"

    # 配置根日志记录器
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_path / log_file,
                encoding='utf-8'
            ),
        ],
    )

class LoggerMixin:
    """日志混入类"""

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
