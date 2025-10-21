from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, AsyncGenerator
import uvicorn
import os
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import asyncio

from piper import PiperVoice, SynthesisConfig
import io
import wave
import base64
import concurrent.futures
from collections import OrderedDict

from src.utils.db import init_db
from src.rag_client import RAGClient
from src.conversation_manager import ConversationManager
from src.utils.data_parser import parser_fhir

load_dotenv()

init_db()

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 FastAPI 應用
app = FastAPI(
    title="FaceHeartAGI API",
    description="FHIR 醫療資料分析與 RAG 增強 LLM 互動 API，支援異步串流模式",
    version="3.0.0"
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化組件
rag_client = RAGClient()
conversation_manager = ConversationManager()

# 初始化 Piper 語音模型（只載入一次）
PIPER_VOICE_PATH = "./en_US-lessac-medium.onnx"
voice = PiperVoice.load(PIPER_VOICE_PATH, use_cuda=False)  # 改 True 若要用 GPU

# 可調參數（選用）
piper_config = SynthesisConfig(
    length_scale=1.0,  # 語速（>1 慢）
    noise_scale=0.667,  # 音質變化
    noise_w_scale=0.8,  # 語調變化
    volume=1.0,         # 音量
)

tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

def synthesize_audio_bytes(text: str) -> bytes:
    """使用 PiperVoice 將文字轉為 wav bytes"""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=piper_config)
    return buffer.getvalue()

def split_text_for_tts(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    將文字分割成適合 TTS 的片段
    
    Args:
        text: 要分割的文字
        max_chunk_size: 每個片段的最大字符數
        
    Returns:
        分割後的文字片段列表
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # 按句子分割
    sentences = text.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果當前片段加上新句子不會超過限制
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence
        else:
            # 如果當前片段不為空，先保存
            if current_chunk:
                chunks.append(current_chunk)
            # 如果單個句子就超過限制，強制分割
            if len(sentence) > max_chunk_size:
                # 按字符強制分割
                for i in range(0, len(sentence), max_chunk_size):
                    chunks.append(sentence[i:i + max_chunk_size])
                current_chunk = ""
            else:
                current_chunk = sentence
    
    # 添加最後一個片段
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def load_default_knowledge_base():
    """載入預設知識庫"""
    try:
        with open('./knowledge/default_knowledge_base.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("預設知識庫檔案不存在，使用空知識庫")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"預設知識庫檔案格式錯誤: {e}")
        return {}

# 載入預設知識庫
DEFAULT_KNOWLEDGE_BASE = load_default_knowledge_base()

# Pydantic 模型
class MedicalAnalysisRequest(BaseModel):
    """醫療分析請求模型"""
    device_id: str
    knowledge_base: Optional[Dict[str, Any]] = None  # 知識庫內容（若無提供則使用預設知識庫模板）
    user_question: str  # 用戶問題
    fhir_data: Optional[str]  # 個人 FHIR 資料
    retrieval_type: Optional[str] = "vector"  # LLM 或向量檢索 ("llm" 或 "vector")

class ConversationHistoryRequest(BaseModel):
    """對話歷史請求模型"""
    device_id: str

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    timestamp: str

# 輔助函數
async def format_streaming_response(stream_generator: AsyncGenerator[str, None], response_type: str = "response") -> AsyncGenerator[str, None]:
    """
    格式化 streaming 回應為 SSE 格式
    
    Args:
        stream_generator: 異步生成器
        response_type: 回應類型
        
    Yields:
        SSE 格式的 streaming 回應
    """
    try:
        # 發送開始標記
        yield f"event: start\ndata: {json.dumps({'type': response_type, 'message': f'{response_type} 開始'}, ensure_ascii=False)}\n\n"
        
        chunk_count = 0
        async for chunk in stream_generator:
            if chunk is not None:
                chunk_count += 1
                # 發送內容片段
                yield f"event: chunk\ndata: {json.dumps({'content': chunk, 'chunk_id': chunk_count}, ensure_ascii=False)}\n\n"
        
        # 發送結束標記
        yield f"event: end\ndata: {json.dumps({'type': response_type, 'message': f'{response_type} 完成', 'total_chunks': chunk_count}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        # 發送錯誤信息
        yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

# 健康檢查端點
@app.get("/")
async def root():
    """根端點 - 健康檢查"""
    return {
        "message": "FaceHeartAGI API 運行中",
        "version": "3.0.0",
        "features": ["FHIR 分析", "RAG 增強", "醫療問答", "異步串流模式"],
        "retrieval_types": ["vector", "llm"],
        "timestamp": datetime.now().isoformat()
    }

# 主要醫療分析串流端點
@app.post("/analyze-stream")
async def analyze_stream(request: MedicalAnalysisRequest):
    """
    醫療資料分析端點，支援異步串流模式
    
    參數:
    - device_id: 識別 ID
    - knowledge_base: 知識庫內容（可選，若無提供則使用預設知識庫模板）
    - user_question: 用戶問題
    - fhir_data: 個人 FHIR 資料
    - retrieval_type: 檢索類型 ("llm" 或 "vector")
    
    返回 Server-Sent Events (SSE) 格式的串流回應
    """
    async def generate_streaming_analysis():
        """
        修改版：先收集完整文字，再一次性轉成語音
        """
        full_response = ""

        try:
            logger.info(f"收到醫療分析請求，會話ID: {request.device_id}, 問題: {request.user_question}")
            knowledge_base = request.knowledge_base if request.knowledge_base else DEFAULT_KNOWLEDGE_BASE
            fhir = parser_fhir(json.loads(request.fhir_data)) if request.fhir_data else ""
            conversation_history = conversation_manager.format_conversation_history_for_prompt(request.device_id)

            # 第一階段：收集完整文字回應
            async for chunk in rag_client.enhance_response_with_rag_stream(
                request.user_question,
                fhir,
                knowledge_base,
                request.retrieval_type,
                conversation_history
            ):
                if not chunk:
                    continue

                full_response += chunk
                # 立即傳出文字事件
                yield f"event: text\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

            # 第二階段：文字完成後，只將 </think> 後的文字轉成語音
            if full_response.strip():
                logger.info("文字回應完成，開始轉換語音...")
                yield f"event: text_complete\ndata: {json.dumps({'message': '文字回應完成，開始轉換語音'}, ensure_ascii=False)}\n\n"
                
                try:
                    # 提取 </think> 後的文字
                    think_end_marker = "</think>"
                    if think_end_marker in full_response:
                        # 找到 </think> 後的所有文字
                        parts = full_response.split(think_end_marker, 1)
                        if len(parts) > 1:
                            tts_text = parts[1].strip()
                            logger.info(f"提取到 TTS 文字，長度: {len(tts_text)} 字符")
                        else:
                            tts_text = full_response.strip()
                            logger.info("未找到 </think> 標記，使用完整回應")
                    else:
                        tts_text = full_response.strip()
                        logger.info("未找到 </think> 標記，使用完整回應")
                    
                    if tts_text:
                        # 將文字分割成多個片段
                        text_chunks = split_text_for_tts(tts_text, max_chunk_size=100)
                        logger.info(f"文字已分割成 {len(text_chunks)} 個片段")
                        
                        # 為每個片段生成音訊
                        for i, chunk in enumerate(text_chunks):
                            try:
                                logger.info(f"正在處理第 {i+1}/{len(text_chunks)} 個音訊片段...")
                                
                                # 使用執行器進行 TTS 轉換
                                audio_bytes = await asyncio.get_event_loop().run_in_executor(
                                    tts_executor, 
                                    synthesize_audio_bytes, 
                                    chunk
                                )
                                
                                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                                
                                # 發送音訊片段
                                is_last = (i == len(text_chunks) - 1)
                                yield f"event: audio\ndata: {json.dumps({'audio': audio_b64, 'chunk_index': i, 'total_chunks': len(text_chunks), 'is_last': is_last, 'complete': is_last}, ensure_ascii=False)}\n\n"
                                
                                logger.info(f"第 {i+1} 個音訊片段完成，大小: {len(audio_bytes)} bytes")
                                
                            except Exception as e:
                                logger.error(f"第 {i+1} 個音訊片段轉換失敗: {e}")
                                yield f"event: audio_error\ndata: {json.dumps({'error': f'第 {i+1} 個音訊片段轉換失敗: {str(e)}'}, ensure_ascii=False)}\n\n"
                        
                        logger.info("所有音訊片段轉換完成")
                    else:
                        logger.warning("沒有文字需要轉換為語音")
                        yield f"event: audio_error\ndata: {json.dumps({'error': '沒有文字需要轉換為語音'}, ensure_ascii=False)}\n\n"
                    
                except Exception as e:
                    logger.error(f"語音轉換失敗: {e}")
                    yield f"event: audio_error\ndata: {json.dumps({'error': f'語音轉換失敗: {str(e)}'}, ensure_ascii=False)}\n\n"

                # ✅ 記錄完整對話
                await conversation_manager.add_conversation_turn(
                    request.device_id,
                    request.user_question,
                    full_response,
                    fhir
                )
                logger.info(f"已記錄會話 {request.device_id} 的對話輪次")

            yield "event: end\ndata: {\"status\": \"complete\"}\n\n"

        except Exception as e:
            logger.error(f"醫療分析過程中發生錯誤: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_streaming_analysis(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

# 清除會話端點
@app.delete("/clear-session")
async def clear_session(request: ConversationHistoryRequest):
    """
    清除指定會話的所有記錄
    
    參數:
    - device_id: 識別 ID
    
    返回:
    - 清除確認
    """
    try:
        conversation_manager.clear_session(request.device_id)
        
        return APIResponse(
            success=True,
            data={"device_id": request.device_id},
            message="會話記錄清除成功",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"清除會話記錄時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清除會話記錄失敗: {str(e)}")

# API 協助端點
@app.get("/help")
async def help_api():
    """API 協助"""
    return {
        "title": "FaceHeartAGI API v3.0",
        "description": "FHIR 醫療資料分析與 RAG 增強 LLM 互動 API",
        "endpoints": {
            "/": "健康檢查",
            "/analyze-stream": "醫療分析串流端點",
            "/clear-session": "清除會話記錄",
            "/docs": "Swagger UI 文檔（FastAPI 自動生成）",
            "/redoc": "ReDoc 文檔（FastAPI 自動生成）"
        },
        "features": [
            "異步串流回應（醫療分析）",
            "支援傳統和向量檢索",
            "預設知識庫模板",
            "FHIR 資料解析"
        ],
        "retrieval_types": {
            "vector": "向量檢索（預設）",
            "llm": "LLM 檢索"
        }
    }

# 啟動服務器
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("FACEHEART_API_PORT", 8500)),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info")
    ) 