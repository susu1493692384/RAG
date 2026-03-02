"""
网页爬取器
用于获取网页内容
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class WebPage:
    """网页内容数据类"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]


class WebScraper:
    """网页爬取器类"""

    def __init__(
        self,
        delay: float = 1.0,
        timeout: int = 10,
        user_agent: Optional[str] = None,
    ):
        """
        初始化网页爬取器

        Args:
            delay: 请求间隔（秒）
            timeout: 超时时间（秒）
            user_agent: 用户代理
        """
        self.delay = delay
        self.timeout = timeout
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )

    def scrape_single(self, url: str) -> WebPage:
        """
        爬取单个网页

        Args:
            url: 网页 URL

        Returns:
            WebPage 对象
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("需要安装 requests 和 beautifulsoup4")

        headers = {'User-Agent': self.user_agent}

        try:
            # 发送请求
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            # 解析 HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取标题
            title = self._extract_title(soup)

            # 提取正文
            content = self._extract_content(soup)

            # 提取元数据
            metadata = self._extract_metadata(soup, response)

            return WebPage(
                url=url,
                title=title,
                content=content,
                metadata=metadata
            )

        except Exception as e:
            print(f"爬取失败 {url}: {e}")
            raise

    def scrape_multiple(self, urls: List[str]) -> List[WebPage]:
        """
        批量爬取多个网页

        Args:
            urls: URL 列表

        Returns:
            WebPage 列表
        """
        results = []

        for i, url in enumerate(urls):
            try:
                print(f"正在爬取 ({i+1}/{len(urls)}): {url}")
                page = self.scrape_single(url)
                results.append(page)

                # 延迟，避免被封
                if i < len(urls) - 1:
                    time.sleep(self.delay)

            except Exception as e:
                print(f"跳过 {url}: {e}")
                continue

        return results

    def _extract_title(self, soup) -> str:
        """提取网页标题"""
        # 优先级: h1 > title > meta og:title
        if soup.h1:
            return soup.h1.get_text().strip()
        if soup.title:
            return soup.title.get_text().strip()
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '').strip()

        return "无标题"

    def _extract_content(self, soup) -> str:
        """提取网页正文内容"""
        # 移除不需要的标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()

        # 尝试找到主要内容区域
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_='content') or
            soup.body
        )

        if main_content:
            # 提取文本
            paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        else:
            content = soup.get_text(separator='\n', strip=True)

        return content

    def _extract_metadata(self, soup, response) -> Dict[str, Any]:
        """提取网页元数据"""
        metadata = {
            'status_code': response.status_code,
            'content_length': len(response.content),
        }

        # 提取 meta 标签
        meta_tags = ['description', 'keywords', 'author']
        for tag in meta_tags:
            meta = soup.find('meta', attrs={'name': tag})
            if meta:
                metadata[f'meta_{tag}'] = meta.get('content', '')

        # 提取 Open Graph 标签
        og_tags = soup.find_all('meta', property=lambda x: x and x.startswith('og:'))
        for og in og_tags:
            property_name = og.get('property', '')
            metadata[f'og_{property_name}'] = og.get('content', '')

        return metadata

    def scrape_sitemap(self, sitemap_url: str) -> List[WebPage]:
        """
        爬取 sitemap 中的所有 URL

        Args:
            sitemap_url: sitemap URL

        Returns:
            WebPage 列表
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("需要安装 requests 和 beautifulsoup4")

        # 获取 sitemap
        response = requests.get(sitemap_url, timeout=self.timeout)
        soup = BeautifulSoup(response.content, 'xml')

        # 提取所有 URL
        urls = [loc.text for loc in soup.find_all('loc')]

        print(f"找到 {len(urls)} 个 URL")

        return self.scrape_multiple(urls)

    def save_to_file(self, pages: List[WebPage], output_dir: str = None):
        """
        保存爬取结果到文件

        Args:
            pages: WebPage 列表
            output_dir: 输出目录，默认为项目数据目录
        """
        # 设置默认输出目录为绝对路径
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "data" / "crawled")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for page in pages:
            # 生成文件名
            from urllib.parse import urlparse
            parsed = urlparse(page.url)
            filename = parsed.path.replace('/', '_').strip('_') or 'index'
            filename = f"{filename}.txt"
            filepath = output_dir / filename

            # 保存内容
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"URL: {page.url}\n")
                f.write(f"Title: {page.title}\n")
                f.write(f"{'='*80}\n\n")
                f.write(page.content)
