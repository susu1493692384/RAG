"""
验证知识库是否构建成功
"""
from pymilvus import connections
from enterprise_rag.config import MILVUS_CONFIGS

def verify_knowledge_base():
    """验证知识库"""
    print("=" * 60)
    print("知识库验证工具")
    print("=" * 60)

    # 连接到 Milvus
    config = MILVUS_CONFIGS['standalone']

    try:
        connections.connect(
            alias="default",
            host=config.uri.replace("http://", "").split(":")[0],
            port="19530"
        )
        print(f"✓ 已连接到 Milvus: {config.uri}")
    except Exception as e:
        print(f"× 连接失败: {e}")
        return

    # 获取 collection 信息
    from pymilvus import Collection

    try:
        collection = Collection("knowledge_base")
        collection.load()
        print(f"✓ Collection 'knowledge_base' 已加载")

        # 获取统计信息
        num_entities = collection.num_entities
        print(f"\n知识库统计:")
        print(f"  - 文档块总数: {num_entities}")

        if num_entities > 0:
            print(f"\n✓ 知识库构建成功！包含 {num_entities} 个文档块")

            # 显示前几条记录
            print(f"\n示例记录（前 3 条）:")
            results = collection.query(
                expr="",
                output_fields=["text", "source"],
                limit=3
            )

            for i, result in enumerate(results, 1):
                text = result.get('text', '')[:100] + "..."
                source = result.get('source', 'unknown')
                print(f"  {i}. 来源: {source}")
                print(f"     内容: {text}\n")
        else:
            print("\n× 知识库为空，请先运行构建命令:")
            print("   python -m enterprise_rag.main --mode build")

    except Exception as e:
        print(f"× Collection 不存在或加载失败: {e}")
        print("\n请先运行构建命令:")
        print("   python -m enterprise_rag.main --mode build")

    finally:
        # 断开连接
        connections.disconnect("default")
        print("\n" + "=" * 60)

if __name__ == "__main__":
    verify_knowledge_base()
