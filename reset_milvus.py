"""
重置 Milvus Collection（删除后重建）
"""
from pymilvus import connections, utility

# 连接到 Milvus
connections.connect("default", host="localhost", port="19530")

# 删除 collection
try:
    utility.drop_collection("knowledge_base")
    print("✓ 已删除 collection: knowledge_base")
except Exception as e:
    print(f"× 删除失败: {e}")

# 断开连接
connections.disconnect("default")

print("\n现在可以重新运行: python -m enterprise_rag.main --mode build")
