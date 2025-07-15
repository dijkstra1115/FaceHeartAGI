import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from llm_client import LLMClient
from config import RAGConfig
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)

# TODO:
# 1. add_conversation_turn 的 system_response 可以只記錄 </think> 後的輸出
# 2. format_conversation_history_for_prompt 的 system_response 可以只記錄 </think> 後的輸出
# 3. 增加資料庫來記錄對話

class ConversationManager:
    """對話管理器，負責記錄對話歷史和生成摘要"""
    
    def __init__(self):
        """初始化對話管理器"""
        self.llm_client = LLMClient()
        # 使用字典來儲存不同會話的對話記錄
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        # 儲存對話摘要
        self.conversation_summaries: Dict[str, List[str]] = {}
        # 最大保存的對話輪數
        self.max_conversations = 10


    def add_conversation_turn(self, session_id: str, user_question: str, 
                            system_response: str, fhir_data: str) -> None:
        """
        添加一輪對話記錄
        
        Args:
            session_id: 會話ID
            user_question: 用戶問題
            system_response: 系統回應
            fhir_data: FHIR資料
        """
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.conversation_summaries[session_id] = []
        
        # 記錄對話輪次
        conversation_turn = {
            "turn_number": len(self.conversations[session_id]) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_intent": user_question,
            "system_response": system_response,
            "fhir_data": fhir_data
        }
        
        self.conversations[session_id].append(conversation_turn)
        
        # 限制對話記錄數量為最多10輪
        if len(self.conversations[session_id]) > self.max_conversations:
            # 移除最舊的對話記錄
            removed_conversation = self.conversations[session_id].pop(0)
            logger.info(f"會話 {session_id} 達到最大記錄數量，移除第 {removed_conversation['turn_number']} 輪對話")
            
            # 重新編號剩餘的對話記錄
            for i, conv in enumerate(self.conversations[session_id]):
                conv['turn_number'] = i + 1
        
        # 檢查是否需要生成摘要（當對話數量超過5輪時，每一輪都生成摘要）
        if len(self.conversations[session_id]) >= 5:
            logger.info(f"會話 {session_id} 達到 {len(self.conversations[session_id])} 輪對話，開始生成摘要")
            # 在背景異步執行摘要生成
            import asyncio
            try:
                # 嘗試在當前事件迴圈中執行
                loop = asyncio.get_running_loop()
                loop.create_task(self._generate_conversation_summary(session_id))
            except RuntimeError:
                # 如果沒有運行中的事件迴圈，使用線程執行
                import threading
                def run_async_summary():
                    try:
                        asyncio.run(self._generate_conversation_summary(session_id))
                    except Exception as e:
                        logger.error(f"在線程中生成摘要時發生錯誤: {str(e)}")
                
                thread = threading.Thread(target=run_async_summary)
                thread.daemon = True
                thread.start()


    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        獲取指定會話的對話歷史
        
        Args:
            session_id: 會話ID
            
        Returns:
            對話歷史列表
        """
        return self.conversations.get(session_id, [])


    def get_conversation_summaries(self, session_id: str) -> List[str]:
        """
        獲取指定會話的摘要
        
        Args:
            session_id: 會話ID
            
        Returns:
            摘要列表
        """
        return self.conversation_summaries.get(session_id, [])


    def get_recent_conversations(self, session_id: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        獲取最近的對話記錄
        
        Args:
            session_id: 會話ID
            count: 要獲取的對話輪數
            
        Returns:
            最近的對話記錄
        """
        conversations = self.conversations.get(session_id, [])
        return conversations[-count:] if conversations else []
        
    
    def format_conversation_history_for_prompt(self, session_id: str) -> str:
        """
        格式化對話歷史為提示詞文本
        根據新的邏輯：當歷史長度 > 5 時，包含最近幾輪加上摘要
        
        Args:
            session_id: 會話ID
            
        Returns:
            格式化的對話歷史文本
        """
        conversations = self.conversations.get(session_id, [])
        summaries = self.conversation_summaries.get(session_id, [])
        
        if not conversations:
            return "This is our first conversation."
        
        history_text = ""
                
        # 如果有摘要且對話輪數 > 5，先添加摘要
        if summaries and len(conversations) > 5:
            history_text += "**Conversation Summary**\n"
            for i, summary in enumerate(summaries, 1):
                history_text += f"Summary {i}:\n {summary}\n"
            history_text += "\n"
        
        # 添加最近的對話記錄
        if len(conversations) <= 5:
            recent_conversations = conversations
        else:
            # 如果對話輪數 > 5，只顯示最近幾輪
            recent_count = len(conversations) - 5
            recent_conversations = conversations[-recent_count:]
        
        for i, conv in enumerate(recent_conversations, 1):
            history_text += f"[Turn {conv['turn_number']}]\n"
            history_text += f"**User Question**\n{conv['user_intent']}\n"
            
            # 添加FHIR資料摘要
            if conv.get('fhir_data'):
                history_text += f"**FHIR Data**\n{conv['fhir_data']}\n"
            
            # 限制回應長度以避免提示詞過長
            response_preview = conv['system_response'][:200]
            if len(conv['system_response']) > 200:
                response_preview += "..."
            history_text += f"**System Response**\n{response_preview}\n"
                
        return history_text

    
    async def _generate_conversation_summary(self, session_id: str) -> None:
        """
        生成對話摘要（永遠只對前5輪進行總結）
        
        Args:
            session_id: 會話ID
        """
        try:
            conversations = self.conversations.get(session_id, [])
            if len(conversations) < 5:
                return
            
            # 永遠只對前5輪對話進行摘要
            first_five_conversations = conversations[:5]
            
            # 構建摘要提示詞
            summary_prompt = PromptBuilder.build_summary_prompt(first_five_conversations)
            
            messages = [
                {
                    "role": "system",
                    "content": PromptBuilder.get_system_prompt("summary")
                },
                {
                    "role": "user", 
                    "content": summary_prompt
                }
            ]
            
            # 使用LLM生成摘要
            summary = await self.llm_client.generate_response_async(
                messages,
                max_tokens=1000,
                temperature=0.3
            )
            
            # 儲存摘要
            self.conversation_summaries[session_id].append(summary)
            
            logger.info(f"已為會話 {session_id} 生成第 {len(self.conversation_summaries[session_id])} 個摘要（基於前5輪對話）")
            
        except Exception as e:
            logger.error(f"生成對話摘要時發生錯誤: {str(e)}")
    

    def clear_session(self, session_id: str) -> None:
        """
        清除指定會話的記錄
        
        Args:
            session_id: 會話ID
        """
        if session_id in self.conversations:
            del self.conversations[session_id]
        if session_id in self.conversation_summaries:
            del self.conversation_summaries[session_id]
        logger.info(f"已清除會話 {session_id} 的所有記錄") 