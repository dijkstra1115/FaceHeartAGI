import os
from typing import Dict, Any


class RAGConfig:
    """RAG 系統配置管理"""
    
    # API 配置
    # BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    # DEFAULT_MODEL = "deepseek/deepseek-r1:free"

    BASE_URL = "http://172.24.186.27:8000/v1/chat/completions"
    DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    FACEHEART_API_PORT = 8500
    FACEHEART_API_URL = f"http://localhost:{FACEHEART_API_PORT}"
    
    # 請求配置
    DEFAULT_MAX_TOKENS = 2000
    RETRIEVAL_TEMPERATURE = 0.1
    ENHANCEMENT_TEMPERATURE = 0.3
    
    # 向量檢索配置
    VECTOR_SEARCH_TOP_K = 5
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """獲取 API 請求標頭"""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("請設定 OPENROUTER_API_KEY 環境變數")
        
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://faceheart-agi.com",
            "X-Title": "FaceHeartAGI",
        }
    