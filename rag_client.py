import logging
from typing import Dict, Any, Optional, AsyncGenerator
from dotenv import load_dotenv
import os
from llm_client import LLMClient
from prompt_builder import PromptBuilder
from retrieval_strategies import VectorRetrievalStrategy, LLMRetrievalStrategy

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class RAGClient:
    """RAG (Retrieval-Augmented Generation) 客戶端，支援向量檢索和 LLM 檢索，支援 streaming 模式"""
    
    def __init__(self, use_vector_search: bool = True):
        """
        初始化 RAG 客戶端
        
        Args:
            use_vector_search: 是否使用向量檢索（預設為 True）
        """
        # 初始化 LLM 客戶端
        self.llm_client = LLMClient()
        
        # 初始化檢索策略（使用懶加載）
        self.use_vector_search = use_vector_search
        self.retrieval_strategy = None  # 懶加載，不立即初始化
    
    async def retrieve_relevant_context(self, user_question: str, database_content: Dict[str, Any]) -> str:
        """
        從資料庫中檢索相關內容
        
        Args:
            user_question: 用戶問題
            database_content: 資料庫內容
            
        Returns:
            檢索到的相關內容
        """
        # 確保檢索策略已初始化
        if self.retrieval_strategy is None:
            logger.info("檢索失敗，請先切換到 LLM / Vector檢索模式")
            return ""
        
        return await self.retrieval_strategy.retrieve(user_question, database_content)
    
    async def enhance_response_with_rag_stream(self, user_question: str, fhir_data: str, 
                                             database_content: Dict[str, Any], conversation_history: str = "") -> AsyncGenerator[str, None]:
        """
        使用 RAG 增強回應（streaming 模式）
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            database_content: 資料庫內容
            conversation_history: 對話歷史（可選）
            
        Yields:
            增強回應的文字片段
        """
        try:
            # 檢索相關內容
            retrieved_context = await self.retrieve_relevant_context(user_question, database_content)
            
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

        # print(enhancement_prompt)
        
        logger.info("開始生成增強回應（streaming）...")
        async for chunk in self.llm_client.generate_response_stream(
            messages, 
            max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)), 
            temperature=float(os.getenv("LLM_ENHANCEMENT_TEMPERATURE", 0.3))
        ):
            yield chunk
    
    def switch_to_llm_search(self) -> None:
        """切換到 LLM 檢索模式"""
        if not isinstance(self.retrieval_strategy, LLMRetrievalStrategy):
            self.use_vector_search = False
            self.retrieval_strategy = LLMRetrievalStrategy()
            logger.info("已切換到 LLM 檢索模式")
        else:
            logger.info("已經在 LLM 檢索模式")
    
    def switch_to_vector_search(self) -> bool:
        """
        切換到向量檢索模式
        
        Returns:
            是否成功切換
        """
        if isinstance(self.retrieval_strategy, VectorRetrievalStrategy):
            logger.info("已經在向量檢索模式")
            return True
            
        try:
            self.retrieval_strategy = VectorRetrievalStrategy()
            self.use_vector_search = True
            logger.info("已切換到向量檢索模式")
            return True
        except Exception as e:
            logger.error(f"切換到向量檢索模式失敗: {str(e)}")
            return False 