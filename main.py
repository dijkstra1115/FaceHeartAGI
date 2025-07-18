from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, AsyncGenerator
import uvicorn
import json
import logging
from datetime import datetime
import asyncio

from rag_client import RAGClient
from conversation_manager import ConversationManager
from data_parser import observation_parser

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

# Pydantic 模型
class MedicalAnalysisRequest(BaseModel):
    """醫療分析請求模型"""
    session_id: str  # 會話ID，用於記錄對話歷史
    knowledge_base: Optional[Dict[str, Any]] = None  # 知識庫內容（若無提供則使用預設知識庫模板）
    user_question: str  # 用戶問題
    fhir_data: Dict[str, Any]  # 個人 FHIR 資料
    retrieval_type: str = "vector"  # LLM 或向量檢索 ("llm" 或 "vector")
    additional_context: Optional[Dict[str, Any]] = None

class RAGRetrieveRequest(BaseModel):
    """RAG 檢索請求模型"""
    session_id: str  # 會話ID，用於記錄對話歷史
    knowledge_base: Optional[Dict[str, Any]] = None  # 知識庫內容（若無提供則使用預設知識庫模板）
    user_question: str  # 用戶問題
    retrieval_type: str = "vector"  # LLM 或向量檢索 ("llm" 或 "vector")

class ConversationHistoryRequest(BaseModel):
    """對話歷史請求模型"""
    session_id: str  # 會話ID

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

# Favicon 端點
@app.get("/favicon.ico")
async def favicon():
    """處理 favicon 請求"""
    return Response(status_code=204)

# 主要醫療分析串流端點
@app.post("/analyze-stream")
async def analyze_stream(request: MedicalAnalysisRequest):
    """
    醫療資料分析端點，支援異步串流模式
    
    參數:
    - session_id: 會話ID
    - knowledge_base: 知識庫內容（可選，若無提供則使用預設知識庫模板）
    - user_question: 用戶問題
    - fhir_data: 個人 FHIR 資料
    - retrieval_type: 檢索類型 ("llm" 或 "vector")
    
    返回 Server-Sent Events (SSE) 格式的串流回應
    """
    async def generate_streaming_analysis():
        full_response = ""  # 用於收集完整回應
        try:
            logger.info(f"收到醫療分析請求，會話ID: {request.session_id}, 問題: {request.user_question}")
            logger.info(f"檢索類型: {request.retrieval_type}")
            
            # 使用預設知識庫模板（如果沒有提供）
            knowledge_base = request.knowledge_base if request.knowledge_base else DEFAULT_KNOWLEDGE_BASE
                        
            # 根據檢索類型設定 RAG 客戶端
            if request.retrieval_type == "llm":
                rag_client.switch_to_llm_search()
            else:
                rag_client.switch_to_vector_search()
            
            # 獲取對話歷史
            conversation_history = conversation_manager.format_conversation_history_for_prompt(request.session_id)
            
            # 直接使用 RAG 串流增強生成回應
            async for chunk in format_streaming_response(
                rag_client.enhance_response_with_rag_stream(
                    request.user_question,
                    observation_parser(request.fhir_data),
                    knowledge_base,
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
            
            # 記錄完整的對話輪次
            if full_response.strip():
                
                conversation_manager.add_conversation_turn(
                    request.session_id,
                    request.user_question,
                    full_response,
                    observation_parser(request.fhir_data)
                )
                
                logger.info(f"已記錄會話 {request.session_id} 的對話輪次")
                    
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

# RAG 檢索端點（非串流）
@app.post("/rag-retrieve")
async def rag_retrieve(request: RAGRetrieveRequest):
    """
    RAG 檢索端點（非串流模式）
    
    參數:
    - knowledge_base: 知識庫內容（可選，若無提供則使用預設知識庫模板）
    - user_question: 用戶問題
    - retrieval_type: 檢索類型 ("llm" 或 "vector")
    
    返回 JSON 格式的回應
    """
    try:
        logger.info(f"收到 RAG 檢索請求，會話ID: {request.session_id}, 問題: {request.user_question}")
        logger.info(f"檢索類型: {request.retrieval_type}")
        
        # 使用預設知識庫模板（如果沒有提供）
        knowledge_base = request.knowledge_base if request.knowledge_base else DEFAULT_KNOWLEDGE_BASE
        
        # 根據檢索類型設定 RAG 客戶端
        if request.retrieval_type == "llm":
            rag_client.switch_to_llm_search()
        else:
            rag_client.switch_to_vector_search()
        
        # 進行 RAG 檢索
        retrieved_context = await rag_client.retrieve_relevant_context(
            request.user_question, 
            knowledge_base
        )
        
        # 記錄檢索結果作為對話輪次
        # 注意：RAG檢索端點沒有FHIR資料，使用空的FHIR資料
        empty_fhir_data = {}
        
        conversation_manager.add_conversation_turn(
            request.session_id,
            request.user_question,
            retrieved_context or "沒有檢索到相關內容",
            empty_fhir_data
        )
        
        return APIResponse(
            success=True,
            data={
                "user_question": request.user_question,
                "retrieval_type": request.retrieval_type,
                "retrieved_context": retrieved_context,
                "context_length": len(retrieved_context) if retrieved_context else 0
            },
            message="RAG 檢索完成",
            timestamp=datetime.now().isoformat()
        )
            
    except Exception as e:
        logger.error(f"RAG 檢索過程中發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RAG 檢索失敗: {str(e)}")


# 清除會話端點
@app.delete("/clear-session")
async def clear_session(request: ConversationHistoryRequest):
    """
    清除指定會話的所有記錄
    
    參數:
    - session_id: 會話ID
    
    返回:
    - 清除確認
    """
    try:
        conversation_manager.clear_session(request.session_id)
        
        return APIResponse(
            success=True,
            data={"session_id": request.session_id},
            message="會話記錄清除成功",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"清除會話記錄時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清除會話記錄失敗: {str(e)}")

# API 文檔端點
@app.get("/api-docs")
async def get_api_docs():
    """獲取 API 文檔"""
    return {
        "title": "FaceHeartAGI API v3.0",
        "description": "FHIR 醫療資料分析與 RAG 增強 LLM 互動 API",
        "endpoints": {
            "/": "健康檢查",
            "/analyze-stream": "醫療分析串流端點",
            "/rag-retrieve": "RAG 檢索端點（非串流）",
            "/clear-session": "清除會話記錄",
            "/docs": "Swagger UI 文檔（FastAPI 自動生成）",
            "/redoc": "ReDoc 文檔（FastAPI 自動生成）"
        },
        "features": [
            "異步串流回應（醫療分析）",
            "非串流檢索（RAG 檢索）",
            "支援傳統和向量檢索",
            "預設知識庫模板",
            "FHIR 資料解析"
        ],
        "retrieval_types": {
            "vector": "向量檢索（預設）",
            "llm": "LLM 檢索"
        },
        "swagger_ui": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc"
    }

# 啟動服務器
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8500,
        reload=True,
        log_level="info"
    ) 