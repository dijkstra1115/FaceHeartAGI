import logging
import asyncio
import json
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
    
    async def _classify_question(self, user_question: str) -> str:
        """
        分類用戶問題類型
        
        Args:
            user_question: 用戶問題
            
        Returns:
            問題類型 ("meta_question" 或 "medical_question")
        """
        try:
            classifier_prompt = PromptBuilder.build_question_classifier_prompt(user_question)
            
            messages = [
                {
                    "role": "system",
                    "content": PromptBuilder.get_system_prompt("question_classifier")
                },
                {
                    "role": "user",
                    "content": classifier_prompt
                }
            ]
            
            logger.info("開始分類問題類型...")
            logger.info(f"🔍 [DEBUG] 分類器輸入問題: {user_question}")
            
            # 分類器使用較低的 temperature 和較少的 tokens，因為只需要返回 JSON
            raw_response = await self.llm_client.generate_response(
                messages,
                max_tokens=50,  # 分類器只需要很少的 tokens
                temperature=0.1  # 低 temperature 確保穩定的分類結果
            )
            
            logger.info(f"🔍 [DEBUG] 分類器原始響應: {repr(raw_response)}")
            
            # 解析 JSON 響應
            try:
                # 嘗試提取 JSON（可能包含 markdown 代碼塊或其他文本）
                cleaned_response = raw_response.strip()
                logger.info(f"🔍 [DEBUG] 清理後的響應: {repr(cleaned_response)}")
                
                # 移除可能的 markdown 代碼塊標記
                if cleaned_response.startswith("```"):
                    lines = cleaned_response.split("\n")
                    cleaned_response = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_response
                    logger.info(f"🔍 [DEBUG] 移除 markdown 代碼塊後: {repr(cleaned_response)}")
                elif "```json" in cleaned_response:
                    cleaned_response = cleaned_response.split("```json")[1].split("```")[0].strip()
                    logger.info(f"🔍 [DEBUG] 提取 JSON 代碼塊後: {repr(cleaned_response)}")
                
                result = json.loads(cleaned_response)
                logger.info(f"🔍 [DEBUG] 解析後的 JSON 結果: {result}")
                
                question_type = result.get("question_type", "medical_question")
                logger.info(f"✅ 問題分類結果: {question_type}")
                return question_type
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ 無法解析分類器 JSON 響應")
                logger.warning(f"🔍 [DEBUG] 原始響應: {repr(raw_response)}")
                if 'cleaned_response' in locals():
                    logger.warning(f"🔍 [DEBUG] 清理後響應: {repr(cleaned_response)}")
                logger.warning(f"🔍 [DEBUG] JSON 解析錯誤: {e}")
                # 如果無法解析，嘗試從文本中提取
                if "meta_question" in raw_response.lower():
                    logger.info("✅ 從文本中提取到 meta_question，返回 meta_question")
                    return "meta_question"
                logger.info("✅ 無法確定類型，默認返回 medical_question")
                return "medical_question"  # 默認返回醫療問題
                
        except Exception as e:
            logger.error(f"分類問題時發生錯誤: {str(e)}")
            # 發生錯誤時默認返回醫療問題，確保系統繼續運行
            return "medical_question"
    
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
                                             database_content: Dict[str, Any], retrieval_type: str, conversation_history: str = "",
                                             require_retrieval: bool = False, enable_conversation_history: bool = True) -> AsyncGenerator[str, None]:
        """
        使用 RAG 增強回應（streaming 模式）
        
        流程：
        1. 第一層：問題分類器判斷問題類型
        2. 第二層：根據分類結果選擇不同的處理流程
           - meta_question: 使用元問題專用 prompt（跳過檢索）
           - medical_question: 使用現有的 RAG 流程
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            database_content: 資料庫內容
            retrieval_type: The retrieval type to use.
            conversation_history: 對話歷史（可選）
            require_retrieval: 是否必須檢索到資料（若為 True 且未檢索到資料則回報錯誤）
            enable_conversation_history: 是否啟用歷史對話功能（若為 False 且未檢索到資料則回報錯誤）
            
        Yields:
            增強回應的文字片段
        """
        try:
            # =============================
            # 第一層：問題分類器
            # =============================
            question_type = await self._classify_question(user_question)
            logger.info(f"問題分類結果: {question_type}")
            
            # =============================
            # 第二層：根據分類結果路由
            # =============================
            if question_type == "meta_question":
                # 元問題：跳過檢索，直接使用 FHIR 數據生成回應
                logger.info("檢測到元問題，使用元問題專用處理流程")
                async for chunk in self._generate_meta_question_response_stream(fhir_data):
                    yield chunk
                return
            
            # 醫療問題：使用現有的 RAG 流程
            logger.info("檢測到醫療問題，使用標準 RAG 處理流程")
            
            # 根據檢索類型設定 RAG 客戶端（使用依賴注入避免重複創建 LLMClient）
            if retrieval_type == "llm":
                retrieval_strategy = LLMRetrievalStrategy(self.llm_client)
            else:
                retrieval_strategy = VectorRetrievalStrategy()

            # 檢索相關內容
            retrieved_context = await self.retrieve_relevant_context(retrieval_strategy, user_question, database_content)
            
            if not retrieved_context or retrieved_context.strip() == "No relevant content retrieved.":
                # 如果設定必須檢索，則回報錯誤
                if require_retrieval:
                    logger.warning("未找到相關的資料庫內容，且設定為必須檢索模式")
                    yield "錯誤：未能從知識庫中檢索到相關資料。請確認您的問題是否與知識庫內容相關，或嘗試調整問題的表述方式。"
                    return
                
                # 如果歷史對話功能已停用，則必須要有檢索資料才能回應
                if not enable_conversation_history:
                    logger.warning("未找到相關的資料庫內容，且歷史對話功能已停用，無法生成回應")
                    yield "錯誤：未能從知識庫中檢索到相關資料。由於歷史對話功能已停用，系統需要檢索資料才能回應。請確認您的問題是否與知識庫內容相關，或嘗試調整問題的表述方式。"
                    return
                
                # 歷史對話功能已啟用，可以使用歷史對話生成回應
                logger.info("未找到相關的資料庫內容，將使用歷史對話生成回應")
                async for chunk in self._generate_base_response_stream(user_question, fhir_data, conversation_history):
                    yield chunk
                return
            
            # 根據是否有歷史對話選擇生成方法
            if conversation_history and conversation_history.strip() != "None":
                # 有歷史對話，使用增強回應
                async for chunk in self._generate_enhanced_response_stream(user_question, fhir_data, retrieved_context, conversation_history):
                    yield chunk
            else:
                # 無歷史對話，僅使用檢索資料
                async for chunk in self._generate_retrieval_only_response_stream(user_question, fhir_data, retrieved_context):
                    yield chunk
        
        except asyncio.CancelledError:
            logger.info("RAG 增強回應被取消")
            raise
                
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

        # print(f"Enhancement Prompt: {enhancement_prompt}")  # Debugging line
        
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
    
    async def _generate_retrieval_only_response_stream(self, user_question: str, fhir_data: str, 
                                                      retrieved_context: str) -> AsyncGenerator[str, None]:
        """生成僅基於檢索資料的回應（無歷史對話，streaming 模式）"""
        retrieval_only_prompt = PromptBuilder.build_retrieval_only_prompt(
            user_question, fhir_data, retrieved_context
        )
        
        messages = [
            {
                "role": "system",
                "content": PromptBuilder.get_system_prompt("retrieval_only")
            },
            {
                "role": "user",
                "content": retrieval_only_prompt
            }
        ]
        
        logger.info("開始生成僅檢索資料回應（無歷史對話，streaming）...")
        async for chunk in self.llm_client.generate_response_stream(
            messages, 
            max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)), 
            temperature=float(os.getenv("LLM_ENHANCEMENT_TEMPERATURE", 0.3))
        ):
            yield chunk
    
    async def _generate_meta_question_response_stream(self, fhir_data: str) -> AsyncGenerator[str, None]:
        """
        生成元問題回應（streaming 模式）
        用於回答 "what questions can you answer?" 等問題
        
        Args:
            fhir_data: FHIR 資料
            
        Yields:
            回應的文字片段
        """
        meta_question_prompt = PromptBuilder.build_meta_question_prompt(fhir_data)
        
        messages = [
            {
                "role": "system",
                "content": PromptBuilder.get_system_prompt("meta_question")
            },
            {
                "role": "user",
                "content": meta_question_prompt
            }
        ]
        
        logger.info("開始生成元問題回應（streaming）...")
        async for chunk in self.llm_client.generate_response_stream(
            messages,
            max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 2000)),
            temperature=float(os.getenv("LLM_ENHANCEMENT_TEMPERATURE", 0.3))
        ):
            yield chunk 