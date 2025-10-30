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
import tempfile
import uuid
from pathlib import Path

from src.utils.db import init_db
from src.rag_client import RAGClient
from src.conversation_manager import ConversationManager
from src.utils.data_parser import parser_fhir

load_dotenv()

# TTS 相關導入
try:
    import piper
    import wave
    import io
    TTS_AVAILABLE = True
    config_path = Path("./voices/en_US-lessac-medium.onnx.json")
    with open(config_path, "r", encoding="utf-8") as f:
        ENGLISH_CONFIG = json.load(f)
except ImportError:
    TTS_AVAILABLE = False

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

# TTS 服務類
class TTSService:
    """語音合成服務"""
    
    def __init__(self):
        self.audio_cache = {}  # 音頻文件緩存
        self.audio_dir = Path("./audio_cache")
        self.audio_dir.mkdir(exist_ok=True)

        self.voices_dir = Path("./voices")
        self.voices_dir.mkdir(exist_ok=True)
        
        if TTS_AVAILABLE:
            try:
                # 使用項目內的語音模型
                model_path = os.path.join(self.voices_dir, "en_US-lessac-medium.onnx")
                if os.path.exists(model_path):
                    self.tts = piper.PiperVoice.load(model_path, config_path=None, use_cuda=False)
                else:
                    raise FileNotFoundError(f"語音模型文件不存在: {model_path}")
                print("Piper TTS 初始化成功")
            except Exception as e:
                print(f"Piper TTS 初始化失敗: {e}")
                self.tts = None
        else:
            self.tts = None
    
    def generate_audio(self, text: str, device_id: str) -> Optional[str]:
        """生成語音文件並返回文件路徑"""
        if not self.tts:
            print("TTS 服務不可用")
            return None
        
        try:
            # 生成唯一文件名
            audio_id = str(uuid.uuid4())
            audio_path = self.audio_dir / f"{device_id}_{audio_id}.wav"

            pure_text = text.split("</think>")[-1].strip()
            
            # 生成語音
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                self.tts.synthesize_wav(pure_text, wav_file, syn_config=ENGLISH_CONFIG)

            buffer.seek(0)
            with open(audio_path, "wb") as f:
                f.write(buffer.read())
            
            # 緩存文件路徑
            self.audio_cache[audio_id] = str(audio_path)
            
            print(f"語音文件生成成功: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            print(f"語音生成失敗: {e}")
            return None
    
    def get_audio_path(self, audio_id: str) -> Optional[str]:
        """根據 audio_id 獲取音頻文件路徑"""
        return self.audio_cache.get(audio_id)

# 初始化 TTS 服務
tts_service = TTSService()

# Pydantic 模型
class MedicalAnalysisRequest(BaseModel):
    """醫療分析請求模型"""
    device_id: str
    knowledge_base: Optional[Dict[str, Any]] = None  # 知識庫內容（若無提供則使用預設知識庫模板）
    user_question: str  # 用戶問題
    fhir_data: Optional[str]  # 個人 FHIR 資料
    retrieval_type: Optional[str] = "vector"  # LLM 或向量檢索 ("llm" 或 "vector")
    generate_audio: Optional[bool] = False  # 是否生成語音

class ConversationHistoryRequest(BaseModel):
    """對話歷史請求模型"""
    device_id: str

class AudioRequest(BaseModel):
    """語音播放請求模型"""
    device_id: str
    audio_id: str

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
        full_response = ""  # 用於收集完整回應
        audio_id = None  # 語音文件ID
        try:
            logger.info(f"收到醫療分析請求，會話ID: {request.device_id}, 問題: {request.user_question}")
            logger.info(f"檢索類型: {request.retrieval_type}")
            logger.info(f"生成語音: {request.generate_audio}")
            
            # 使用預設知識庫模板（如果沒有提供）
            knowledge_base = request.knowledge_base if request.knowledge_base else DEFAULT_KNOWLEDGE_BASE

            fhir = parser_fhir(json.loads(request.fhir_data)) if request.fhir_data else ""

            conversation_history = conversation_manager.format_conversation_history_for_prompt(request.device_id)
                        
            # 直接使用 RAG 串流增強生成回應
            async for chunk in format_streaming_response(
                rag_client.enhance_response_with_rag_stream(
                    request.user_question,
                    fhir,
                    knowledge_base,
                    request.retrieval_type,
                    conversation_history
                ),
                "medical_analysis"
            ):
                # 收集回應內容用於記錄對話
                if '"content"' in chunk:
                    try:
                        # 解析 SSE 數據
                        lines = chunk.strip().split('\n')
                        for line in lines:
                            if line.startswith('data: '):
                                data_str = line[6:]
                                data = json.loads(data_str)
                                if 'content' in data:
                                    full_response += data['content']
                    except:
                        pass
                
                yield chunk
            
            # 生成語音（如果需要）
            if request.generate_audio and full_response.strip():
                try:
                    audio_path = tts_service.generate_audio(full_response, request.device_id)
                    if audio_path:
                        # 從路徑中提取 audio_id
                        audio_id = Path(audio_path).stem.split('_')[-1]
                        logger.info(f"語音生成成功，audio_id: {audio_id}")
                        
                        # 發送語音生成完成事件
                        yield f"event: audio_ready\ndata: {json.dumps({'audio_id': audio_id, 'message': '語音生成完成'}, ensure_ascii=False)}\n\n"
                    else:
                        logger.warning("語音生成失敗")
                        yield f"event: audio_error\ndata: {json.dumps({'error': '語音生成失敗'}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.error(f"語音生成過程中發生錯誤: {e}")
                    yield f"event: audio_error\ndata: {json.dumps({'error': f'語音生成錯誤: {str(e)}'}, ensure_ascii=False)}\n\n"
            
            # 記錄完整的對話輪次
            if full_response.strip():
                
                await conversation_manager.add_conversation_turn(
                    request.device_id,
                    request.user_question,
                    full_response,
                    fhir
                )
                
                logger.info(f"已記錄會話 {request.device_id} 的對話輪次")
                    
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

# 語音播放端點
@app.get("/audio/{device_id}/{audio_id}")
async def get_audio(device_id: str, audio_id: str):
    """
    獲取語音文件
    
    參數:
    - device_id: 識別 ID
    - audio_id: 語音文件 ID
    
    返回:
    - 音頻文件流
    """
    try:
        # 從 TTS 服務獲取音頻文件路徑
        audio_path = tts_service.get_audio_path(audio_id)
        
        if not audio_path or not Path(audio_path).exists():
            raise HTTPException(status_code=404, detail="語音文件不存在")
        
        # 檢查文件是否屬於該設備
        if not audio_path.endswith(f"{device_id}_{audio_id}.wav"):
            raise HTTPException(status_code=403, detail="無權限訪問此語音文件")
        
        # 讀取音頻文件
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"inline; filename={device_id}_{audio_id}.wav",
                "Cache-Control": "public, max-age=3600"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"獲取語音文件時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取語音文件失敗: {str(e)}")

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
            "/audio/{device_id}/{audio_id}": "語音文件播放端點",
            "/docs": "Swagger UI 文檔（FastAPI 自動生成）",
            "/redoc": "ReDoc 文檔（FastAPI 自動生成）"
        },
        "features": [
            "異步串流回應（醫療分析）",
            "支援傳統和向量檢索",
            "預設知識庫模板",
            "FHIR 資料解析",
            "語音合成（Piper TTS）",
            "語音文件播放"
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