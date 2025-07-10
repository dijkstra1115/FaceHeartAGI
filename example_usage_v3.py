#!/usr/bin/env python3
"""
FaceHeartAGI API v3.0 使用範例

這個腳本展示了如何使用新的 v3.0 API 功能：
1. 醫療分析串流（支援向量和 LLM 檢索）
2. RAG 檢索（非串流模式）
3. 預設知識庫模板使用
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 基礎 URL
BASE_URL = "http://localhost:8000"

def load_sample_fhir_data():
    """載入示例 FHIR 資料"""
    try:
        with open('sample_fhir_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("示例 FHIR 資料檔案不存在")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"示例 FHIR 資料檔案格式錯誤: {e}")
        return {}

def load_custom_knowledge_base():
    """載入自定義知識庫"""
    try:
        with open('custom_knowledge_base.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("自定義知識庫檔案不存在")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"自定義知識庫檔案格式錯誤: {e}")
        return {}

# 載入示例資料
SAMPLE_FHIR_DATA = load_sample_fhir_data()
CUSTOM_KNOWLEDGE_BASE = load_custom_knowledge_base()

async def stream_response(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], title: str):
    """處理串流回應"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"會話ID: {payload.get('session_id', 'N/A')}")
    print(f"問題: {payload.get('user_question', 'N/A')}")
    print(f"檢索類型: {payload.get('retrieval_type', 'vector')}")
    print("-" * 60)
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ 請求失敗: {response.status} - {error_text}")
                return
            
            print("✅ 開始接收串流回應...")
            print()
            
            chunk_count = 0
            full_content = ""
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data_str = line[6:]  # 移除 'data: ' 前綴
                    try:
                        data = json.loads(data_str)
                        
                        if 'content' in data:
                            chunk_count += 1
                            content = data['content']
                            print(content, end='', flush=True)
                            full_content += content
                        elif 'message' in data:
                            print(f"\n📝 {data['message']}")
                        elif 'error' in data:
                            print(f"\n❌ 錯誤: {data['error']}")
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"\n\n📊 統計信息:")
            print(f"   總片段數: {chunk_count}")
            print(f"   完整回應長度: {len(full_content)} 字符")
            
    except Exception as e:
        print(f"❌ 連接錯誤: {str(e)}")

async def non_stream_response(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], title: str):
    """處理非串流回應"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"會話ID: {payload.get('session_id', 'N/A')}")
    print(f"問題: {payload.get('user_question', 'N/A')}")
    print(f"檢索類型: {payload.get('retrieval_type', 'vector')}")
    print("-" * 60)
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"❌ 請求失敗: {response.status} - {error_text}")
                return
            
            result = await response.json()
            
            if result.get('success'):
                print("✅ 請求成功")
                data = result.get('data', {})
                
                # 顯示檢索內容
                retrieved_context = data.get('retrieved_context', '')
                if retrieved_context:
                    print(f"\n📄 檢索內容:")
                    print(f"{retrieved_context}")
                else:
                    print(f"\n📄 檢索內容: 無相關內容")
            else:
                print(f"❌ 請求失敗: {result.get('message')}")
            
    except Exception as e:
        print(f"❌ 連接錯誤: {str(e)}")







async def example_medical_analysis():
    """範例：醫療分析串流"""
    print("\n🎯 醫療分析串流範例")
    
    async with aiohttp.ClientSession() as session:
        # 1. 使用自定義知識庫 + 向量檢索
        payload1 = {
            "session_id": "example_session_1",
            "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
            "fhir_data": SAMPLE_FHIR_DATA,
            "user_question": "痛風的症狀?",
            "retrieval_type": "vector"
        }
        
        await stream_response(
            session,
            f"{BASE_URL}/analyze-stream",
            payload1,
            "醫療分析串流（自定義知識庫 + 向量檢索）"
        )
        
        # # 2. 使用自定義知識庫 + LLM 檢索
        # payload2 = {
        #     "session_id": "example_session_2",
        #     "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        #     "fhir_data": SAMPLE_FHIR_DATA,
        #     "user_question": "高血壓的人要吃甚麼?",
        #     "retrieval_type": "llm"
        # }
        
        # await stream_response(
        #     session,
        #     f"{BASE_URL}/analyze-stream",
        #     payload2,
        #     "醫療分析串流（自定義知識庫 + LLM 檢索）"
        # )

async def example_rag_retrieval():
    """範例：RAG 檢索（非串流）"""
    print("\n🔍 RAG 檢索範例（非串流）")
    
    async with aiohttp.ClientSession() as session:
        # 1. 向量檢索
        payload1 = {
            "session_id": "rag_session_1",
            "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
            "user_question": "飲食建議",
            "retrieval_type": "vector"
        }
        
        await non_stream_response(
            session,
            f"{BASE_URL}/rag-retrieve",
            payload1,
            "RAG 檢索（向量檢索）"
        )
        
        # 2. LLM 檢索
        payload2 = {
            "session_id": "rag_session_2",
            "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
            "user_question": "高血壓的人要吃甚麼?",
            "retrieval_type": "llm"
        }
        
        await non_stream_response(
            session,
            f"{BASE_URL}/rag-retrieve",
            payload2,
            "RAG 檢索（LLM 檢索）"
        )

async def check_api_health():
    """檢查 API 健康狀態"""
    print("🔍 檢查 API 健康狀態...")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    print("✅ API 運行正常")
                    print(f"   版本: {data.get('version')}")
                    print(f"   功能: {', '.join(data.get('features', []))}")
                    print(f"   檢索類型: {', '.join(data.get('retrieval_types', []))}")
                    return True
                else:
                    print(f"❌ API 健康檢查失敗: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 無法連接到 API: {str(e)}")
        return False

async def main():
    """主函數"""
    print("🚀 FaceHeartAGI API v3.0 使用範例")
    print("=" * 60)
    
    # 檢查資料檔案
    if not SAMPLE_FHIR_DATA:
        print("❌ 無法載入示例 FHIR 資料，請確保 sample_fhir_data.json 檔案存在")
        return
    
    if not CUSTOM_KNOWLEDGE_BASE:
        print("❌ 無法載入自定義知識庫，請確保 custom_knowledge_base.json 檔案存在")
        return
    
    # 檢查 API 健康狀態
    if not await check_api_health():
        print("\n❌ API 未運行，請先啟動 API 服務器:")
        print("   python main.py")
        return
    
    # 執行範例
    try:
        # await example_rag_retrieval()
        await example_medical_analysis()
        
        print("\n" + "=" * 60)
        print("✅ 所有範例執行完成！")
        print("\n💡 提示:")
        print("   • 醫療分析使用串流模式，適合長回應")
        print("   • RAG 檢索使用非串流模式，快速獲取檢索結果")
        print("   • 向量檢索適合語義相似性查詢")
        print("   • LLM 檢索適合精確關鍵字匹配")
        print("   • 預設知識庫包含基本醫療指南")
        print("   • 自定義知識庫可根據需求調整")
        print("   • 資料檔案: sample_fhir_data.json, custom_knowledge_base.json")
        
    except KeyboardInterrupt:
        print("\n⚠️ 範例執行被用戶中斷")
    except Exception as e:
        print(f"\n❌ 範例執行過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    print("請確保 API 服務器正在運行（python main.py）")
    print("如果需要，請先設定 OPENROUTER_API_KEY 環境變數")
    print("請確保以下 JSON 檔案存在:")
    print("   • sample_fhir_data.json")
    print("   • custom_knowledge_base.json")
    print("   • default_knowledge_base.json")
    print()
    
    asyncio.run(main()) 