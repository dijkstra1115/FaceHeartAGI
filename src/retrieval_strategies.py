import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import aiohttp
import os
from src.vector_store import MedicalVectorStore
from src.utils.prompt_builder import PromptBuilder
from src.utils.data_parser import extract_medical_documents
from dotenv import load_dotenv
from src.llm_client import LLMClient

load_dotenv()

logger = logging.getLogger(__name__)

# 全局單例向量存儲實例
_vector_store_instance: Optional[MedicalVectorStore] = None

def get_vector_store() -> MedicalVectorStore:
    """獲取或創建單例向量存儲實例"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = MedicalVectorStore()
    return _vector_store_instance


class RetrievalStrategy(ABC):
    """檢索策略抽象基類"""
    
    @abstractmethod
    def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """執行檢索"""
        pass


class VectorRetrievalStrategy(RetrievalStrategy):
    """向量檢索策略 - 使用共享的向量存儲實例"""
    
    def __init__(self):
        # 使用單例向量存儲
        self.vector_store = get_vector_store()
    
    def retrieve(self, user_question: str, database_content: Dict[str, Any] = None) -> str:
        """
        使用向量檢索
        
        Args:
            user_question: 用戶問題
            database_content: 已棄用，保留參數以保持接口兼容性
            
        Returns:
            檢索到的相關內容
        """
        try:
            # 確保知識庫已加載（只加載一次）
            logger.info("檢查醫療知識庫加載狀態...")
            self.vector_store.load_knowledge_base()
            
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
    """LLM 檢索策略 - 使用依賴注入的 LLMClient"""
    
    def __init__(self, llm_client: LLMClient):
        """
        初始化 LLM 檢索策略
        
        Args:
            llm_client: LLM 客戶端實例（通過依賴注入傳入，避免重複創建）
        """
        self.llm_client = llm_client
        self.vector_store = get_vector_store()

    async def retrieve(self, user_question: str, database_content: Dict[str, Any] = None) -> str:
        """
        使用 LLM 檢索方法（非 streaming）
        
        Args:
            user_question: 用戶問題
            database_content: 已棄用，保留參數以保持接口兼容性
            
        Returns:
            檢索到的相關內容
        """
        try:
            # 確保知識庫已加載
            self.vector_store.load_knowledge_base()
            
            # 獲取所有文檔內容用於 LLM 檢索
            contents = self.vector_store.documents
            
            if not contents:
                logger.warning("知識庫為空，無法進行 LLM 檢索")
                return ""

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

            return await self.llm_client.generate_response(
                messages,
                max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)),
                temperature=float(os.getenv("LLM_RETRIEVAL_TEMPERATURE", 0.1))
            )
                
        except Exception as e:
            logger.error(f"LLM 檢索過程中發生錯誤: {str(e)}")
            return "" 