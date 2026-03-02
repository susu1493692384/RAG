"""
API 数据模型定义
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """来源信息"""
    content: str = Field(..., description="文档内容")
    source: str = Field(..., description="来源文件")
    score: float = Field(..., description="相似度分数")


class QueryRequest(BaseModel):
    """查询请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    top_k: Optional[int] = Field(5, description="返回结果数量", ge=1, le=20)
    use_rerank: Optional[bool] = Field(True, description="是否使用重排序")
    conversational: Optional[bool] = Field(False, description="是否为对话模式")
    filter_expression: Optional[str] = Field(None, description="过滤表达式")


class QueryResponse(BaseModel):
    """查询响应"""
    answer: str = Field(..., description="回答内容")
    sources: List[SourceInfo] = Field(..., description="来源列表")
    query: str = Field(..., description="原始问题")


class DocumentChunk(BaseModel):
    """文档块"""
    content: str = Field(..., description="文本内容")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    documents: List[DocumentChunk] = Field(..., description="文档列表")
    batch_size: Optional[int] = Field(100, description="批处理大小")


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    total: int = Field(..., description="插入总数")
    details: Dict[str, Any] = Field(..., description="详细信息")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="版本号")
    components: Dict[str, str] = Field(..., description="组件状态")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(None, description="详细描述")
