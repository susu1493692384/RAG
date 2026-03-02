"""
OCR 处理器
用于处理图片和扫描版 PDF
"""
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import hashlib


@dataclass
class OCRResult:
    """OCR 识别结果"""
    text: str
    confidence: float
    bbox: Optional[List[Tuple[int, int, int, int]]] = None
    metadata: Optional[dict] = None


class OCRProcessor:
    """OCR 处理器类"""

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = 'ch',  # ch: 中文, en: 英文
        use_gpu: bool = False,
        show_log: bool = False,
    ):
        """
        初始化 OCR 处理器

        Args:
            use_angle_cls: 是否使用方向分类器
            lang: 语言类型
            use_gpu: 是否使用 GPU
            show_log: 是否显示日志
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        self.show_log = show_log

        self._ocr_engine = None

    @property
    def ocr_engine(self):
        """延迟加载 OCR 引擎"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=self.use_angle_cls,
                    lang=self.lang,
                    use_gpu=self.use_gpu,
                    show_log=self.show_log,
                )
            except ImportError:
                raise ImportError(
                    "需要安装 PaddleOCR: pip install paddleocr paddlepaddle"
                )
        return self._ocr_engine

    def process_image(self, image_path: str) -> OCRResult:
        """
        处理图片文件

        Args:
            image_path: 图片路径

        Returns:
            OCRResult 对象
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        # 执行 OCR
        result = self.ocr_engine.ocr(str(image_path), cls=True)

        # 提取文本和置信度
        texts = []
        confidences = []
        bboxes = []

        if result and result[0]:
            for line in result[0]:
                if line:
                    box = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    texts.append(text)
                    confidences.append(confidence)
                    bboxes.append(box)

        # 合并所有文本
        full_text = "\n".join(texts)

        # 计算平均置信度
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return OCRResult(
            text=full_text,
            confidence=avg_confidence,
            bbox=bboxes,
            metadata={
                'source': str(image_path),
                'filename': image_path.name,
                'total_lines': len(texts),
            }
        )

    def process_pdf(self, pdf_path: str) -> List[OCRResult]:
        """
        处理 PDF 文件（逐页）

        Args:
            pdf_path: PDF 文件路径

        Returns:
            OCRResult 列表（每页一个结果）
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("需要安装 pdf2image: pip install pdf2image")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        # 将 PDF 转换为图片
        images = convert_from_path(str(pdf_path), dpi=200)

        results = []
        for page_num, image in enumerate(images):
            # 保存临时图片
            temp_path = pdf_path.parent / f"_temp_page_{page_num}.jpg"
            image.save(temp_path, 'JPEG')

            # OCR 处理
            result = self.process_image(str(temp_path))
            result.metadata['page_number'] = page_num + 1
            result.metadata['total_pages'] = len(images)
            results.append(result)

            # 删除临时文件
            temp_path.unlink()

        return results

    def process_batch(self, image_paths: List[str]) -> List[OCRResult]:
        """
        批量处理图片

        Args:
            image_paths: 图片路径列表

        Returns:
            OCRResult 列表
        """
        results = []
        for path in image_paths:
            try:
                result = self.process_image(path)
                results.append(result)
            except Exception as e:
                print(f"处理失败 {path}: {e}")
                continue

        return results

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """计算文件哈希值，用于缓存"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def process_with_cache(
        self,
        image_path: str,
        cache_dir: str = None
    ) -> OCRResult:
        """
        带缓存的处理

        Args:
            image_path: 图片路径
            cache_dir: 缓存目录，默认为项目数据目录

        Returns:
            OCRResult 对象
        """
        # 设置默认缓存目录为绝对路径
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / "data" / "cache" / "ocr")

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 计算文件哈希
        file_hash = self.calculate_file_hash(image_path)
        cache_file = cache_dir / f"{file_hash}.txt"

        # 尝试从缓存读取
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_text = f.read()
            return OCRResult(
                text=cached_text,
                confidence=1.0,  # 缓存的结果假设是可靠的
                metadata={'cached': True, 'source': image_path}
            )

        # 执行 OCR
        result = self.process_image(image_path)

        # 保存到缓存
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(result.text)

        return result
