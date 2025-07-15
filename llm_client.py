import os
import json
import requests
from typing import Dict, Any, Optional, List, AsyncGenerator
from dotenv import load_dotenv
import asyncio
import aiohttp
from config import RAGConfig

# 載入環境變數
load_dotenv()

class LLMClient:
    """DeepSeek LLM 客戶端，透過 OpenRouter API，支援異步串流模式"""
    
    def __init__(self):
        """初始化 DeepSeek 客戶端"""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("請設定 OPENROUTER_API_KEY 環境變數")
        
        self.base_url = RAGConfig.BASE_URL
        self.model = RAGConfig.DEFAULT_MODEL
        self.headers = RAGConfig.get_headers()

    async def generate_response_stream(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> AsyncGenerator[str, None]:
        """
        生成 LLM 回應（異步串流模式）
        
        Args:
            messages: 聊天消息列表
            max_tokens: 最大輸出 token 數
            temperature: 創意度 (0-1)
            
        Yields:
            LLM 回應的文字片段
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=self.base_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API 請求失敗: {response.status} - {error_text}")
                    
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
                                
        except Exception as e:
            raise Exception(f"LLM 異步串流回應生成失敗: {str(e)}")

    async def generate_response_async(self, messages: List[Dict[str, str]], max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        生成 LLM 回應（異步模式）
        
        Args:
            messages: 聊天消息列表
            max_tokens: 最大輸出 token 數
            temperature: 創意度 (0-1)
            
        Returns:
            LLM 回應的完整文字
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=self.base_url,
                    headers=self.headers,
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

