from typing import Dict, Any, List
from datetime import datetime
import json


def extract_medical_documents(medical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    從醫療資料中提取文檔
    
    Args:
        medical_data: 醫療資料
        
    Returns:
        文檔列表
    """
    documents = []
    
    condition = medical_data.get("condition")
    
    # 疾病描述
    if medical_data.get("description"):
        content = f"【{condition}】疾病說明：{medical_data['description']}"
        documents.append({
            'content': content,
            'metadata': {
                'condition': condition,
                'topic_type': 'description'
            }
        })

    if medical_data.get("symptoms"):
        for symptom in medical_data["symptoms"]:
            content = f"【{condition}】症狀：{symptom}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'symptoms'
                }
            })

    if medical_data.get("diagnosis"):
        for diagnosis in medical_data["diagnosis"]:
            content = f"【{condition}】診斷：{diagnosis}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'diagnosis'
                }
            })

    # 推薦建議分類處理
    for topic, suggestions in medical_data.get("recommendations", {}).items():
        for suggestion in suggestions:
            if suggestion.strip():
                content = suggestion.strip()
                full_content = f"【{condition}】{topic} 建議：{content}"
                documents.append({
                    'content': full_content,
                    'metadata': {
                        'condition': condition,
                        'topic_type': topic
                    }
                })

    # 風險因子
    if medical_data.get("risk_factors"):
        for risk_factor in medical_data["risk_factors"]:
            content = f"【{condition}】風險因子：{risk_factor}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'risk_factors'
                }
            })

    # 併發症
    if medical_data.get("complications"):
        for complication in medical_data["complications"]:
            content = f"【{condition}】併發症：{complication}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'complications'
                }
            })

    # domestic
    if medical_data.get("domestic"):
        for sentence in medical_data.get("domestic", []):
            content = f"【{condition}】台灣資料：{sentence}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'domestic_context'
                }
            })

    # ethnic
    if medical_data.get("ethnic"):
        for sentence in medical_data.get("ethnic", []):
            content = f"【{condition}】族群資料：{sentence}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'ethnic_context'
                }
            })
    
    return documents


VITAL_SIGNS = {
    "39156-5": "BMI",
    "8867-4": "心跳",
    "9279-1": "呼吸速率",
    "59408-5": "血氧飽和度",
    "8480-6": "收縮壓",
    "8462-4": "舒張壓"
}

def get_patient_id(observation):
    ref = observation.get("subject", {}).get("reference", "")
    return ref.replace("Patient/", "") if ref else "未知"


def get_datetime(observation):
    dt = observation.get("effectiveDateTime", "")
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except:
        return dt or "未知"


def get_all_components(observation):
    results = []
    for comp in observation.get("component", []):
        coding = comp.get("code", {}).get("coding", [])
        for code_item in coding:
            code = code_item.get("code")
            label = VITAL_SIGNS.get(code)
            if label:
                value = comp.get("valueQuantity", {}).get("value")
                unit = comp.get("valueQuantity", {}).get("unit", "")
                if value is not None:
                    results.append(f"{label}: {value} {unit}")
    return results


def observation_parser(observation: Dict[str, Any]) -> str:
    lines = [
        f"病人編號: {get_patient_id(observation)}",
        f"測量時間: {get_datetime(observation)}"
    ]

    vitals = get_all_components(observation)
    if vitals:
        lines.extend(vitals)
    else:
        lines.append("（無可用的生命徵象）")

    return "\n".join(lines)