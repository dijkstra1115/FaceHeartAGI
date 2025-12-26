import os
import json
import requests
from typing import Dict, Any, Optional, List, AsyncGenerator
from dotenv import load_dotenv
import asyncio
import aiohttp

# 載入環境變數
load_dotenv()

class LLMClient:
    """DeepSeek LLM 客戶端，透過 OpenRouter API，支援異步串流模式"""
    
    def __init__(self):
        """初始化 DeepSeek 客戶端"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("請設定 OPENROUTER_API_KEY 環境變數")
        
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1/chat/completions")
        self.model = os.getenv("LLM_DEFAULT_MODEL", "deepseek-qwen7b")
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """獲取或創建共享的 ClientSession"""
        if self._session is None or self._session.closed:
            # 配置連接池參數以優化資源使用
            connector = aiohttp.TCPConnector(
                limit=100,  # 最大連接數
                limit_per_host=30,  # 每個主機的最大連接數
                ttl_dns_cache=300,  # DNS 緩存時間
                force_close=False,  # 保持連接以複用
            )
            timeout = aiohttp.ClientTimeout(total=300, connect=30, sock_read=60)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """關閉 ClientSession"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def generate_response_stream(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        生成 LLM 回應（異步串流模式）
        
        Args:
            messages: 聊天消息列表
            max_tokens: 最大輸出 token 數
            temperature: 創意度 (0-1)
            
        Yields:
            LLM 回應的文字片段
        """
        if max_tokens is None:
            max_tokens = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 1000))
            
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            session = await self.get_session()
            async with session.post(
                url=self.base_url,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API 請求失敗: {response.status} - {error_text}")
                
                try:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]  # 移除 'data: ' 前綴
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and len(chunk['choices']) > 0:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        yield delta['content']
                            except json.JSONDecodeError:
                                continue
                except asyncio.CancelledError:
                    # 客戶端斷開時，確保清理
                    raise
                            
        except asyncio.CancelledError:
            # 向上傳播取消異常
            raise
        except Exception as e:
            raise Exception(f"LLM 異步串流回應生成失敗: {str(e)}")

    async def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = None, temperature: float = 0.7) -> str:
        """
        生成 LLM 回應（異步模式）
        
        Args:
            messages: 聊天消息列表
            max_tokens: 最大輸出 token 數
            temperature: 創意度 (0-1)
            
        Returns:
            LLM 回應的完整文字
        """
        if max_tokens is None:
            max_tokens = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 1000))
            
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            session = await self.get_session()
            async with session.post(
                url=self.base_url,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API 請求失敗: {response.status} - {error_text}")
                
                response_data = await response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return response_data['choices'][0]['message']['content']
                else:
                    raise Exception("API 回應格式錯誤")
                        
        except Exception as e:
            raise Exception(f"LLM 同步回應生成失敗: {str(e)}")

