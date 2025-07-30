import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
import aiohttp
import os
from src.vector_store import MedicalVectorStore
from src.utils.prompt_builder import PromptBuilder
from src.utils.data_parser import extract_medical_documents
from dotenv import load_dotenv
from src.llm_client import LLMClient

load_dotenv()

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """檢索策略抽象基類"""
    
    @abstractmethod
    def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """執行檢索"""
        pass


class VectorRetrievalStrategy(RetrievalStrategy):
    """向量檢索策略"""
    
    def __init__(self):
        self.vector_store = MedicalVectorStore()
    
    def retrieve(self, user_question: str, database_content: Dict[str, Any]) -> str:
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
    
    def __init__(self):
        self.llm_client = LLMClient()

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

            return await self.llm_client.generate_response(
                messages,
                max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)),
                temperature=float(os.getenv("LLM_RETRIEVAL_TEMPERATURE", 0.1))
            )
                
        except Exception as e:
            logger.error(f"LLM 檢索過程中發生錯誤: {str(e)}")
            return "" "" 