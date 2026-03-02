"""
LLM 模型配置
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMConfig:
    """LLM 模型配置类"""

    # LLM 提供商
    provider: str = "zhipu"  # deepseek, qwen, zhipu, openai

    # API 基础 URL
    base_url: Optional[str] = None

    # API 密钥
    api_key: Optional[str] = None

    # 模型名称
    model_name: str = "glm-4"

    # 温度（0-2，越低越确定）
    temperature: float = 0.1

    # 最大 tokens
    max_tokens: int = 2000

    # Top P 采样
    top_p: float = 0.9

    # 超时时间（秒）
    timeout: int = 60

    # 重试次数
    max_retries: int = 3

    # 流式输出
    stream: bool = False

    # 额外参数
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        """初始化后设置默认值"""
        if self.extra_params is None:
            self.extra_params = {}

        # 根据提供商设置默认 API
        if self.base_url is None:
            provider_urls = {
                "deepseek": "https://api.deepseek.com/v1",
                "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "zhipu": "https://open.bigmodel.cn/api/paas/v4/",
                "openai": "https://api.openai.com/v1",
            }
            self.base_url = provider_urls.get(
                self.provider,
                "https://api.deepseek.com/v1"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'LLMConfig':
        """从字典创建配置"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "stream": self.stream,
            "extra_params": self.extra_params,
        }

    def get_llm_kwargs(self) -> dict:
        """获取 LLM 初始化参数"""
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            **self.extra_params
        }


# 预定义配置
LLM_CONFIGS = {
    "deepseek": LLMConfig(
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
        temperature=0.1,
        max_tokens=2000,
    ),
    "qwen": LLMConfig(
        provider="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-plus",
        temperature=0.1,
        max_tokens=2000,
    ),
    "zhipu": LLMConfig(
        provider="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model_name="glm-4",
        temperature=0.1,
        max_tokens=2000,
    ),
}
