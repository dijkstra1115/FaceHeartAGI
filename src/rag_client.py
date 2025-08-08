import logging
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from dotenv import load_dotenv
import os
from src.llm_client import LLMClient
from src.utils.prompt_builder import PromptBuilder
from src.retrieval_strategies import VectorRetrievalStrategy, LLMRetrievalStrategy, RetrievalStrategy

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class RAGClient:
    """RAG (Retrieval-Augmented Generation) 客戶端，支援向量檢索和 LLM 檢索，支援 streaming 模式"""
    
    def __init__(self):
        """
        初始化 RAG 客戶端
        """
        # 初始化 LLM 客戶端
        self.llm_client = LLMClient()
    
    async def retrieve_relevant_context(self, retrieval_strategy: RetrievalStrategy, user_question: str, database_content: Dict[str, Any]) -> str:
        """
        從資料庫中檢索相關內容
        
        Args:
            retrieval_strategy: The retrieval strategy to use.
            user_question: 用戶問題
            database_content: 資料庫內容
            
        Returns:
            檢索到的相關內容
        """
        if asyncio.iscoroutinefunction(retrieval_strategy.retrieve):
            return await retrieval_strategy.retrieve(user_question, database_content)
        else:
            return retrieval_strategy.retrieve(user_question, database_content)
    
    async def enhance_response_with_rag_stream(self, user_question: str, fhir_data: str, 
                                             database_content: Dict[str, Any], retrieval_type: str, conversation_history: str = "") -> AsyncGenerator[str, None]:
        """
        使用 RAG 增強回應（streaming 模式）
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            database_content: 資料庫內容
            retrieval_type: The retrieval type to use.
            conversation_history: 對話歷史（可選）
            
        Yields:
            增強回應的文字片段
        """
        try:
            # 根據檢索類型設定 RAG 客戶端
            if retrieval_type == "llm":
                retrieval_strategy = LLMRetrievalStrategy()
            else:
                retrieval_strategy = VectorRetrievalStrategy()

            # 檢索相關內容
            retrieved_context = await self.retrieve_relevant_context(retrieval_strategy, user_question, database_content)
            
            if not retrieved_context or retrieved_context.strip() == "沒有檢索到相關內容":
                logger.info("未找到相關的資料庫內容，將直接生成回應")
                async for chunk in self._generate_base_response_stream(user_question, fhir_data, conversation_history):
                    yield chunk
                return
            
            # 生成增強回應
            async for chunk in self._generate_enhanced_response_stream(user_question, fhir_data, retrieved_context, conversation_history):
                yield chunk
                
        except Exception as e:
            logger.error(f"增強回應（streaming）過程中發生錯誤: {str(e)}")
            yield f"生成回應時發生錯誤: {str(e)}"
    
    async def _generate_base_response_stream(self, user_question: str, fhir_data: str, conversation_history: str = "") -> AsyncGenerator[str, None]:
        """生成基礎回應（streaming 模式）"""
        base_prompt = PromptBuilder.build_base_prompt(user_question, fhir_data, conversation_history)
        
        messages = [
            {
                "role": "system",
                "content": PromptBuilder.get_system_prompt("base")
            },
            {
                "role": "user",
                "content": base_prompt
            }
        ]
        
        logger.info("開始生成基礎回應（streaming）...")
        async for chunk in self.llm_client.generate_response_stream(
            messages, 
            max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)), 
            temperature=float(os.getenv("LLM_ENHANCEMENT_TEMPERATURE", 0.3))
        ):
            yield chunk
    
    async def _generate_enhanced_response_stream(self, user_question: str, fhir_data: str, 
                                               retrieved_context: str, conversation_history: str = "") -> AsyncGenerator[str, None]:
        """生成增強回應（streaming 模式）"""
        enhancement_prompt = PromptBuilder.build_enhancement_prompt(
            user_question, fhir_data, retrieved_context, conversation_history
        )
        
        messages = [
            {
                "role": "system",
                "content": PromptBuilder.get_system_prompt("enhancement")
            },
            {
                "role": "user",
                "content": enhancement_prompt
            }
        ]
        
        logger.info("開始生成增強回應（streaming）...")
        async for chunk in self.llm_client.generate_response_stream(
            messages, 
            max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)), 
            temperature=float(os.getenv("LLM_ENHANCEMENT_TEMPERATURE", 0.3))
        ):
            yield chunk 