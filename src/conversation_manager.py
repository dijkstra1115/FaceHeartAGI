import logging
import os
from typing import Dict
import asyncio
from sqlalchemy.exc import IntegrityError
from dotenv import load_dotenv
from src.utils.db import SessionLocal, ConversationTurn, ConversationSummary
from src.llm_client import LLMClient
from src.utils.prompt_builder import PromptBuilder

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class ConversationManager:
    """對話管理器，使用 SQLite 進行持久化，並以 transaction + 唯一約束 + 重試機制避免 race condition"""

    max_conversations = 10

    def __init__(self):
        self.llm_client = LLMClient()

    async def add_conversation_turn(
        self, device_id: str, user_question: str,
        system_response: str, fhir_data: str
    ) -> None:
        db = SessionLocal()
        try:
            # transaction: insert, delete oldest if needed
            with db.begin():
                # 取得該 device_id 的最大 turn_number
                max_turn = db.query(ConversationTurn.turn_number)\
                             .filter_by(device_id=device_id)\
                             .order_by(ConversationTurn.turn_number.desc())\
                             .first()
                next_turn = (max_turn[0] + 1) if max_turn else 1

                turn = ConversationTurn(
                    device_id=device_id,
                    turn_number=next_turn,
                    user_intent=user_question,
                    system_response=system_response.split("</think>")[-1].strip(),
                    fhir_data=fhir_data
                )
                db.add(turn)

                # 檢查是否超過最大對話數，如果超過則刪除最舊的記錄
                total_turns = db.query(ConversationTurn).filter_by(device_id=device_id).count()
                if total_turns > self.max_conversations:
                    # 刪除最舊的記錄（turn_number 最小的）
                    oldest = (
                        db.query(ConversationTurn)
                          .filter_by(device_id=device_id)
                          .order_by(ConversationTurn.turn_number.asc())
                          .first()
                    )
                    if oldest:
                        db.delete(oldest)

            # trigger summary when reach threshold (在事务外查询)
            total = db.query(ConversationTurn).filter_by(device_id=device_id).count()
            if total >= 5:
                logger.info(f"License {device_id} reached {total} turns, generating summary")
                # 使用追踪的任务创建方式
                from main import create_tracked_task
                create_tracked_task(self._generate_conversation_summary(device_id))
        finally:
            db.close()

    def format_conversation_history_for_prompt(self, device_id: str) -> str:
        """
        Format recent conversation history into structured XML format for prompt context.
        Compatible with the new XML-based enhancement prompt style.

        Args:
            device_id: The user's device identifier.

        Returns:
            A formatted string block representing the conversation history.
        """
        db = SessionLocal()
        try:
            turns = (
                db.query(ConversationTurn)
                .filter_by(device_id=device_id)
                .order_by(ConversationTurn.turn_number.asc())
                .all()
            )
            latest_summary = (
                db.query(ConversationSummary)
                .filter_by(device_id=device_id)
                .order_by(ConversationSummary.summary_index.desc())
                .first()
            )
        finally:
            db.close()

        if not turns:
            return "<note>This is the first conversation.</note>"

        xml_history = ""

        # If more than 5 turns and a summary exists → include summary + remaining turns
        if latest_summary and len(turns) > 5:
            xml_history += f"  <conversation_summary>{latest_summary.content.strip()}</conversation_summary>\n"
            # Skip the first 5 turns since they are already summarized
            recent_turns = turns[5:]
        else:
            # Otherwise, include all turns (no summary yet)
            recent_turns = turns

        for t in recent_turns:
            xml_history += "  <conversation_turn>\n"
            xml_history += f"    <turn_number>{t.turn_number}</turn_number>\n"
            xml_history += f"    <user_intent>{(t.user_intent or '').strip()}</user_intent>\n"
            if t.fhir_data:
                xml_history += f"    <fhir_data>{t.fhir_data.strip()}</fhir_data>\n"
            xml_history += f"    <system_response>{(t.system_response or '').strip()}</system_response>\n"
            xml_history += "  </conversation_turn>\n"

        return xml_history

    async def _generate_conversation_summary(self, device_id: str) -> None:
        # 保持原有摘要生成邏輯
        db = SessionLocal()
        try:
            turns = db.query(ConversationTurn)\
                      .filter_by(device_id=device_id)\
                      .order_by(ConversationTurn.turn_number.asc())\
                      .limit(5)\
                      .all()
        finally:
            db.close()

        prompt = PromptBuilder.build_summary_prompt(turns)
        messages = [
            {"role": "system", "content": PromptBuilder.get_system_prompt("summary")},
            {"role": "user", "content": prompt}
        ]
        try:
            summary = await self.llm_client.generate_response(
                messages, max_tokens=int(os.getenv("LLM_DEFAULT_MAX_TOKENS", 1000)), temperature=0.3
            )
            db = SessionLocal()
            try:
                with db.begin():
                    idx = db.query(ConversationSummary)\
                            .filter_by(device_id=device_id)\
                            .count() + 1
                    db.add(ConversationSummary(
                        device_id=device_id,
                        summary_index=idx,
                        content=summary.split("</think>")[-1].strip()
                    ))
                logger.info(f"新增第 {idx} 則摘要 for session {device_id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"生成摘要失敗: {e}")

    def clear_session(self, device_id: str) -> None:
        """刪除該 session 所有對話與摘要"""
        db = SessionLocal()
        try:
            with db.begin():
                db.query(ConversationTurn).filter_by(device_id=device_id).delete()
                db.query(ConversationSummary).filter_by(device_id=device_id).delete()
            logger.info(f"已清除 session {device_id} 的所有資料")
        finally:
            db.close()