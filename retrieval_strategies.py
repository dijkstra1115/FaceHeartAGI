import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import aiohttp
import os
from vector_store import MedicalVectorStore
from prompt_builder import PromptBuilder
from data_parser import extract_medical_documents
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """檢索策略抽象基類"""
    
    @abstractmethod
    async def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """執行檢索"""
        pass


class VectorRetrievalStrategy(RetrievalStrategy):
    """向量檢索策略"""
    
    def __init__(self):
        self.vector_store = MedicalVectorStore()
    
    async def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """使用向量檢索"""
        try:
            # 動態創建向量資料庫
            logger.info("開始動態創建向量資料庫...")
            self.vector_store.add_medical_documents(database_content)
            
            # 使用向量檢索
            results = self.vector_store.search_medical_context(
                user_question, 
                top_k=int(os.getenv("VECTOR_SEARCH_TOP_K", 5))
            )
            
            if not results:
                logger.info("向量檢索未找到相關內容")
                return ""
            
            # 格式化檢索結果
            context_parts = []
            for result in results:
                score = result['score']
                content = result['content']
                metadata = result['metadata']
                
                context_parts.append(content)
            
            retrieved_context = "\n".join(context_parts)
            logger.info(f"向量檢索成功，找到 {len(results)} 個相關片段")
            return retrieved_context
            
        except Exception as e:
            logger.error(f"向量檢索過程中發生錯誤: {str(e)}")
            return ""


class LLMRetrievalStrategy(RetrievalStrategy):
    """LLM 檢索策略"""
    
    async def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """使用 LLM 檢索方法（非 streaming）"""
        try:
            documents = extract_medical_documents(database_content)

            contents = []
            for doc in documents:
                content = doc.get('content', '')
                if content.strip():
                    contents.append(content)

            # 建構檢索提示詞
            retrieval_prompt = PromptBuilder.build_retrieval_prompt(user_question, contents)
            
            messages = [
                {
                    "role": "system",
                    "content": PromptBuilder.get_system_prompt("retrieval")
                },
                {
                    "role": "user",
                    "content": retrieval_prompt
                }
            ]

            payload = {
                "model": os.getenv("LLM_DEFAULT_MODEL", "deepseek-qwen7b"),
                "messages": messages,
                "max_tokens": int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)),
                "temperature": float(os.getenv("LLM_RETRIEVAL_TEMPERATURE", 0.1))
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1/chat/completions"),
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        retrieved_context = result['choices'][0]['message']['content']
                        logger.info(f"LLM 檢索成功，長度: {len(retrieved_context)} 字符")
                        return retrieved_context
                    else:
                        error_text = await response.text()
                        logger.error(f"檢索請求失敗: {response.status} - {error_text}")
                        return ""
                
        except Exception as e:
            logger.error(f"LLM 檢索過程中發生錯誤: {str(e)}")
            return "" 