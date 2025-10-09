#!/usr/bin/env python3
"""
測試多輪對話的記憶效果

這個腳本展示LLM如何利用歷史對話提供更連貫的回應：
1. 進行多輪相關對話，每次使用不同的FHIR健康資料
2. 展示LLM如何參考之前的對話內容和健康狀況變化
3. 測試LLM對健康趨勢的記憶和分析能力
4. 分析回應的連貫性和對歷史資料的引用
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 基礎 URL
API_PORT = os.getenv("FACEHEART_API_PORT", 8500)
API_URL = os.getenv("FACEHEART_API_URL")
BASE_URL = f"{API_URL}:{API_PORT}"

def load_fhir_data_files():
    """載入所有 FHIR 資料檔案"""
    fhir_data_list = []
    
    for i in range(1, 7):  # 載入 sample_fhir_data_1.json 到 sample_fhir_data_6.json
        filename = f'../fhir/fhir_{i}.json'
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                json_str = f.read()
                fhir_data_list.append(json_str)
                logger.info(f"成功載入 {filename}")
        except FileNotFoundError:
            logger.warning(f"{filename} 檔案不存在")
            # 如果檔案不存在，使用空的FHIR資料
            fhir_data_list.append({})
        except json.JSONDecodeError as e:
            logger.error(f"{filename} 檔案格式錯誤: {e}")
            fhir_data_list.append({})
    
    return fhir_data_list

def load_custom_knowledge_base():
    """載入自定義知識庫"""
    try:
        with open('../knowledge/hypertension_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("自定義知識庫檔案不存在")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"自定義知識庫檔案格式錯誤: {e}")
        return {}

# 載入示例資料
FHIR_DATA_LIST = load_fhir_data_files()
CUSTOM_KNOWLEDGE_BASE = load_custom_knowledge_base()

async def send_question(session: aiohttp.ClientSession, device_id: str, question: str, turn_number: int, fhir_data: Dict[str, Any]):
    """發送問題並獲取回應"""
    print(f"\n--- Turn {turn_number} ---")
    print(f"❓ User Question: {question}")
    
    payload = {
        "device_id": device_id,
        "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        "fhir_data": fhir_data,
        "user_question": question,
        "retrieval_type": "vector"
    }
    
    try:
        async with session.post(f"{BASE_URL}/analyze-stream", json=payload) as response:
            if response.status == 200:
                print("🤖 System Response:")
                
                # 收集完整回應
                full_response = ""
                chunk_count = 0
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        try:
                            data = json.loads(data_str)
                            if 'content' in data:
                                chunk_count += 1
                                content = data['content']
                                print(content, end='', flush=True)
                                full_response += content
                        except json.JSONDecodeError:
                            continue
                
                print(f"\n\n📊 第 {turn_number} 輪統計:")
                print(f"   回應片段數: {chunk_count}")
                print(f"   回應長度: {len(full_response)} 字符")
                
                return full_response
            else:
                error_text = await response.text()
                print(f"❌ 請求失敗: {response.status} - {error_text}")
                return None
    except Exception as e:
        print(f"❌ 連接錯誤: {str(e)}")
        return None

async def test_sequential_conversation():
    """測試連續對話的記憶效果"""
    print("🧠 測試連續對話的記憶效果")
    print("=" * 60)
    
    device_id = "yuting0815"
    
    # 設計一系列相關的問題，測試LLM的記憶能力
    questions = [
        "What are the symptoms of diabites?",
        "What are the symptoms of hypertension?",
        "What are the potential risks based on my FHIR data?",
        "What are the changes in my FHIR history?",
        "What are the recommendations for my health?",
        "What kind of food should I eat?",
        "What kind of ingredients should I avoid?",
        "What are the changes in my FHIR history?",
        "What did I just ask you?",
        "Could you suggest me how to excercise to avoid hypertension?",
        "What's my BP changes within a month?",
        "Do I have any underlying diseases?"
    ]
    
    async with aiohttp.ClientSession() as session:
        print(f"開始進行 {len(questions)} 輪連續對話")
        print(f"識別 ID: {device_id}")
        print("=" * 60)
        
        responses = []
        
        for i, question in enumerate(questions, 1):
            # 使用對應的FHIR資料（循環使用，如果問題數量超過FHIR資料數量）
            fhir_data_index = (i - 1) % len(FHIR_DATA_LIST)
            fhir_data = FHIR_DATA_LIST[fhir_data_index]
            
            response = await send_question(session, device_id, question, i, fhir_data)
            if response:
                responses.append(response)
            
            # 在對話之間稍作停頓
            if i < len(questions):
                print("\n" + "-" * 40)
                await asyncio.sleep(1)
        
        print("\n" + "=" * 60)
        print("✅ 連續對話測試完成")
        print(f"成功完成 {len(responses)} 輪對話")

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
                    return True
                else:
                    print(f"❌ API 健康檢查失敗: {response.status}")
                    return False
    except Exception as e:
        print(f"❌ 無法連接到 API: {str(e)}")
        return False

async def main():
    """主函數"""
    print("🧠 FaceHeartAGI 多輪對話記憶效果測試")
    print("=" * 60)
    print(BASE_URL)
    
    # 檢查資料檔案
    if not FHIR_DATA_LIST or all(not data for data in FHIR_DATA_LIST):
        print("❌ 無法載入 FHIR 資料，請確保 sample_fhir_data_1.json 到 sample_fhir_data_6.json 檔案存在")
        return
    
    if not CUSTOM_KNOWLEDGE_BASE:
        print("❌ 無法載入自定義知識庫，請確保 custom_knowledge_base.json 檔案存在")
        return
    
    # 檢查 API 健康狀態
    if not await check_api_health():
        print("\n❌ API 未運行，請先啟動 API 服務器:")
        print("   python main.py")
        return
    
    # 執行測試
    try:
        await test_sequential_conversation()
        
        print("\n" + "=" * 60)
        print("✅ 所有記憶效果測試完成！")
        print("\n💡 測試說明:")
        print("   • 連續對話測試：展示LLM如何利用歷史對話提供連貫回應")
        print("   • 獨立對話測試：對比無歷史記憶時的回應差異")
        print("   • 上下文感知測試：驗證LLM對具體信息的記憶能力")
        print("   • 通過比較可以觀察到對話記憶對回應質量的影響")
        
    except KeyboardInterrupt:
        print("\n⚠️ 測試執行被用戶中斷")
    except Exception as e:
        print(f"\n❌ 測試執行過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    print("請確保 API 服務器正在運行（python main.py）")
    print("如果需要，請先設定 OPENROUTER_API_KEY 環境變數")
    print("請確保以下 JSON 檔案存在:")
    print("   • sample_fhir_data_1.json 到 sample_fhir_data_6.json")
    print("   • custom_knowledge_base.json")
    print()
    
    asyncio.run(main()) 
