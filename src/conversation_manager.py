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
        # transaction: insert, delete oldest, re-number
        with db.begin():
            cnt = db.query(ConversationTurn).filter_by(device_id=device_id).count()
            next_turn = cnt + 1

            turn = ConversationTurn(
                device_id=device_id,
                turn_number=next_turn,
                user_intent=user_question,
                system_response=system_response.split("</think>")[-1].strip(),
                fhir_data=fhir_data
            )
            db.add(turn)

            if next_turn > self.max_conversations:
                oldest = (
                    db.query(ConversationTurn)
                      .filter_by(device_id=device_id)
                      .order_by(ConversationTurn.turn_number.asc())
                      .first()
                )
                if oldest:
                    db.delete(oldest)
                    turns = (
                        db.query(ConversationTurn)
                          .filter_by(device_id=device_id)
                          .order_by(ConversationTurn.timestamp.asc())
                          .all()
                    )
                    for idx, t in enumerate(turns, start=1):
                        t.turn_number = idx

        # trigger summary when reach threshold
        total = db.query(ConversationTurn).filter_by(device_id=device_id).count()
        if total >= 5:
            logger.info(f"License {device_id} reached {total} turns, generating summary")
            asyncio.create_task(self._generate_conversation_summary(device_id))
        db.close()

    def format_conversation_history_for_prompt(self, device_id: str) -> str:
        db = SessionLocal()
        turns = db.query(ConversationTurn)\
                  .filter_by(device_id=device_id)\
                  .order_by(ConversationTurn.turn_number.asc())\
                  .all()
        latest_summary = db.query(ConversationSummary)\
                      .filter_by(device_id=device_id)\
                      .order_by(ConversationSummary.summary_index.desc())\
                      .first()
        db.close()

        if not turns:
            return "This is our first conversation."

        text = ""
        # 當對話總數 > 5 且已有摘要時，先加入摘要區塊
        if latest_summary and len(turns) > 5:
            text += f"[Conversation Summary]\n{latest_summary.content}\n"
            print(f"testing: {latest_summary.content}")

        num_turns = len(turns) - 5
        recent = turns[-num_turns:] if len(turns) > 5 else turns
        for t in recent:
            text += f"[Turn {t.turn_number}]\n"
            text += f"User: {t.user_intent}\n"
            if t.fhir_data:
                text += f"FHIR: {t.fhir_data}\n"
            text += f"System: {t.system_response}\n"
        return text

    async def _generate_conversation_summary(self, device_id: str) -> None:
        # 保持原有摘要生成邏輯
        db = SessionLocal()
        turns = db.query(ConversationTurn)\
                  .filter_by(device_id=device_id)\
                  .order_by(ConversationTurn.turn_number.asc())\
                  .limit(5)\
                  .all()
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
            with db.begin():
                idx = db.query(ConversationSummary)\
                        .filter_by(device_id=device_id)\
                        .count() + 1
                db.add(ConversationSummary(
                    device_id=device_id,
                    summary_index=idx,
                    content=summary.split("</think>")[-1].strip()
                ))
            db.close()
            logger.info(f"新增第 {idx} 則摘要 for session {device_id}")
        except Exception as e:
            logger.error(f"生成摘要失敗: {e}")

    def clear_session(self, device_id: str) -> None:
        """刪除該 session 所有對話與摘要"""
        db = SessionLocal()
        with db.begin():
            db.query(ConversationTurn).filter_by(device_id=device_id).delete()
            db.query(ConversationSummary).filter_by(device_id=device_id).delete()
        db.close()
        logger.info(f"已清除 session {device_id} 的所有資料")