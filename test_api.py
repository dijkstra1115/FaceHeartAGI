#!/usr/bin/env python3
"""
測試 FaceHeartAGI API 的腳本
包含串流和非串流端點的測試
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

# 示例 FHIR 資料
SAMPLE_FHIR_DATA = {
    "Patient": {
        "id": "patient-001",
        "name": [{"given": ["張"], "family": "三"}],
        "gender": "male",
        "birthDate": "1980-01-01"
    },
    "Observation": [
        {
            "code": {"coding": [{"display": "Blood Pressure"}]},
            "valueQuantity": {"value": 145, "unit": "mmHg"},
            "effectiveDateTime": "2024-01-15",
            "status": "final"
        },
        {
            "code": {"coding": [{"display": "Blood Pressure Diastolic"}]},
            "valueQuantity": {"value": 95, "unit": "mmHg"},
            "effectiveDateTime": "2024-01-15",
            "status": "final"
        }
    ],
    "Condition": [
        {
            "code": {"coding": [{"display": "Hypertension"}]},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "severity": {"coding": [{"display": "Moderate"}]},
            "onsetDateTime": "2022-06-01"
        }
    ],
    "MedicationRequest": [
        {
            "medicationCodeableConcept": {"coding": [{"display": "Amlodipine"}]},
            "status": "active",
            "intent": "order",
            "dosage": [{"text": "5mg daily"}]
        }
    ]
}

# 自定義知識庫內容
CUSTOM_KNOWLEDGE_BASE = {
    "medical_guidelines": [
        {
            "condition": "高血壓",
            "description": "血壓持續高於正常值（收縮壓≥140mmHg或舒張壓≥90mmHg）",
            "recommendations": {
                "general": [
                    "每日監測血壓，目標 < 140/90 mmHg",
                    "按時服藥，不可自行停藥",
                    "定期回診追蹤"
                ],
                "diet": [
                    "限制鈉攝入量，每日 < 2.3g",
                    "增加蔬菜水果攝入，特別是富含鉀的食物",
                    "減少飽和脂肪和反式脂肪",
                    "控制總熱量攝入，維持健康體重"
                ],
                "exercise": [
                    "規律有氧運動，每週 150 分鐘中等強度運動",
                    "每週 2-3 次肌力訓練",
                    "運動前後監測血壓",
                    "避免劇烈運動，循序漸進增加強度"
                ],
                "monitoring": [
                    "每日監測血壓，記錄變化趨勢",
                    "定期檢查腎功能和心臟功能",
                    "每 3-6 個月檢查血脂",
                    "定期眼科檢查"
                ],
                "lifestyle": [
                    "戒菸，避免二手菸",
                    "控制體重，BMI < 25",
                    "減少壓力，學習放鬆技巧",
                    "保持充足睡眠，每晚 7-8 小時"
                ]
            },
            "risk_factors": ["年齡 > 65 歲", "肥胖", "家族史", "吸菸", "缺乏運動"],
            "complications": ["心臟病", "中風", "腎臟病", "視網膜病變"]
        }
    ],
    "medication_info": {
        "Amlodipine": {
            "type": "鈣離子通道阻斷劑",
            "mechanism": "阻斷鈣離子進入血管平滑肌細胞，使血管擴張",
            "indications": ["高血壓", "心絞痛"],
            "side_effects": ["頭暈", "水腫", "疲勞", "潮紅"],
            "precautions": ["肝功能異常者慎用", "懷孕期間避免使用"],
            "dosage": "通常 5-10mg 每日一次",
            "lifestyle_considerations": [
                "服藥期間避免突然起身，防止頭暈",
                "注意足部水腫，可抬高腿部緩解",
                "避免與葡萄柚汁同時服用",
                "定期監測血壓和肝功能"
            ]
        }
    },
    "general_lifestyle_guidelines": {
        "healthy_diet": [
            "均衡飲食，多樣化食物選擇",
            "增加蔬菜水果攝入，每日 5 份",
            "選擇全穀類食物",
            "適量攝入優質蛋白質",
            "限制加工食品和含糖飲料"
        ],
        "physical_activity": [
            "每週至少 150 分鐘中等強度有氧運動",
            "每週 2-3 次肌力訓練",
            "增加日常活動量，如步行、爬樓梯",
            "運動前後充分熱身和冷卻",
            "根據身體狀況調整運動強度"
        ],
        "stress_management": [
            "學習放鬆技巧，如深呼吸、冥想",
            "保持規律作息",
            "培養興趣愛好",
            "與家人朋友保持良好關係",
            "必要時尋求專業心理諮詢"
        ],
        "preventive_care": [
            "定期健康檢查",
            "接種必要疫苗",
            "避免吸菸和過量飲酒",
            "保持良好個人衛生",
            "定期牙科檢查"
        ]
    }
}

async def test_streaming_endpoint(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], endpoint_name: str):
    """
    測試串流端點
    
    Args:
        session: aiohttp 會話
        url: 端點 URL
        payload: 請求負載
        endpoint_name: 端點名稱
    """
    print(f"\n=== 測試 {endpoint_name} ===")
    print(f"URL: {url}")
    print(f"問題: {payload.get('user_question', 'N/A')}")
    print(f"檢索類型: {payload.get('retrieval_type', 'vector')}")
    print("-" * 50)
    
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

async def test_non_streaming_endpoint(session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], endpoint_name: str):
    """
    測試非串流端點
    
    Args:
        session: aiohttp 會話
        url: 端點 URL
        payload: 請求負載
        endpoint_name: 端點名稱
    """
    print(f"\n=== 測試 {endpoint_name} ===")
    print(f"URL: {url}")
    print(f"問題: {payload.get('user_question', 'N/A')}")
    print(f"檢索類型: {payload.get('retrieval_type', 'vector')}")
    print("-" * 50)
    
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
                print(f"   檢索類型: {data.get('retrieval_type')}")
                print(f"   檢索內容長度: {data.get('context_length', 0)} 字符")
                print(f"   檢索內容: \n{data.get('retrieved_context', '')[:200]}...")
                print(f"   訊息: {result.get('message')}")
                print(f"   時間戳: {result.get('timestamp')}")
            else:
                print(f"❌ 請求失敗: {result.get('message')}")
            
    except Exception as e:
        print(f"❌ 連接錯誤: {str(e)}")

async def test_medical_analysis_stream():
    """測試醫療分析串流端點"""
    # 測試向量檢索
    payload_vector = {
        "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        "fhir_data": SAMPLE_FHIR_DATA,
        "user_question": "根據我的高血壓病史，請提供詳細的治療建議和生活方式調整建議。",
        "retrieval_type": "vector"
    }
    
    # 測試 LLM 檢索
    payload_traditional = {
        "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        "fhir_data": SAMPLE_FHIR_DATA,
        "user_question": "根據我的高血壓病史，請提供詳細的治療建議和生活方式調整建議。",
        "retrieval_type": "llm"
    }
    
    # 測試使用預設知識庫
    payload_default = {
        "fhir_data": SAMPLE_FHIR_DATA,
        "user_question": "根據我的高血壓病史，請提供詳細的治療建議和生活方式調整建議。",
        "retrieval_type": "vector"
    }
    
    async with aiohttp.ClientSession() as session:
        await test_streaming_endpoint(
            session,
            f"{BASE_URL}/analyze-stream",
            payload_vector,
            "醫療分析串流（向量檢索）"
        )
        
        await test_streaming_endpoint(
            session,
            f"{BASE_URL}/analyze-stream",
            payload_traditional,
            "醫療分析串流（LLM 檢索）"
        )
        
        await test_streaming_endpoint(
            session,
            f"{BASE_URL}/analyze-stream",
            payload_default,
            "醫療分析串流（預設知識庫）"
        )

async def test_rag_retrieve():
    """測試 RAG 檢索端點（非串流）"""
    # 測試向量檢索
    payload_vector = {
        "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        "user_question": "高血壓的症狀和治療方法有哪些？",
        "retrieval_type": "vector"
    }
    
    # 測試 LLM 檢索
    payload_traditional = {
        "knowledge_base": CUSTOM_KNOWLEDGE_BASE,
        "user_question": "高血壓的症狀和治療方法有哪些？",
        "retrieval_type": "llm"
    }
    
    # 測試使用預設知識庫
    payload_default = {
        "user_question": "高血壓的症狀和治療方法有哪些？",
        "retrieval_type": "vector"
    }
    
    async with aiohttp.ClientSession() as session:
        await test_non_streaming_endpoint(
            session,
            f"{BASE_URL}/rag-retrieve",
            payload_vector,
            "RAG 檢索（向量檢索）"
        )
        
        await test_non_streaming_endpoint(
            session,
            f"{BASE_URL}/rag-retrieve",
            payload_traditional,
            "RAG 檢索（LLM 檢索）"
        )
        
        await test_non_streaming_endpoint(
            session,
            f"{BASE_URL}/rag-retrieve",
            payload_default,
            "RAG 檢索（預設知識庫）"
        )

async def test_api_health():
    """測試 API 健康狀態"""
    print("=== 檢查 API 健康狀態 ===")
    
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

async def test_api_docs():
    """測試 API 文檔端點"""
    print("\n=== 測試 API 文檔端點 ===")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api-docs") as response:
                if response.status == 200:
                    docs_data = await response.json()
                    print(f"✅ API 文檔端點正常")
                    print(f"   標題: {docs_data.get('title')}")
                    print(f"   描述: {docs_data.get('description')}")
                    print(f"   端點數量: {len(docs_data.get('endpoints', {}))}")
                else:
                    print(f"❌ 無法獲取 API 文檔: {response.status}, {await response.text()}")
    except Exception as e:
        print(f"❌ 無法獲取 API 文檔: {str(e)}")

async def main():
    """主測試函數"""
    print("🚀 FaceHeartAGI API 測試開始")
    print("=" * 60)
    
    # 檢查 API 健康狀態
    if not await test_api_health():
        print("\n❌ API 未運行，請先啟動 API 服務器:")
        print("   python main.py")
        return
    
    # 檢查 API 文檔
    await test_api_docs()
    
    # 測試各個端點
    try:
        # print("\n" + "=" * 60)
        # print("📡 測試串流端點")
        # print("=" * 60)
        # await test_medical_analysis_stream()
        
        print("\n" + "=" * 60)
        print("📡 測試非串流端點")
        print("=" * 60)
        await test_rag_retrieve()
        
        print("\n" + "=" * 60)
        print("✅ 所有測試完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 測試被用戶中斷")
    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    print("請確保 API 服務器正在運行（python main.py）")
    print("如果需要，請先設定 OPENROUTER_API_KEY 環境變數")
    print()
    
    asyncio.run(main()) 