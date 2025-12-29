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
        content = f"{condition} disease description: {medical_data['description']}"
        documents.append({
            'content': content,
            'metadata': {
                'condition': condition,
                'topic_type': 'description'
            }
        })

    if medical_data.get("symptoms"):
        for symptom in medical_data["symptoms"]:
            content = f"{condition} symptom: {symptom}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'symptoms'
                }
            })

    if medical_data.get("diagnosis"):
        for diagnosis in medical_data["diagnosis"]:
            content = f"{condition} diagnosis: {diagnosis}"
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
                full_content = f"{condition} {topic} recommendation: {content}"
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
            content = f"{condition} risk factor: {risk_factor}"
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
            content = f"{condition} complication: {complication}"
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
            content = f"{condition} domestic data: {sentence}"
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
            content = f"{condition} ethnic data: {sentence}"
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
    "8867-4": "Heart rate",
    "9279-1": "Respiratory rate",
    "59408-5": "Oxygen saturation",
    "8480-6": "Systolic blood pressure",
    "8462-4": "Diastolic blood pressure"
}


def format_value(value: Any) -> str:
    """
    格式化數值為兩位小數，避免 LLM 記憶太多冗餘數據
    
    Args:
        value: 原始數值（可以是 int, float 或其他類型）
        
    Returns:
        格式化後的字符串（如果是數字則保留兩位小數，否則返回原值）
    """
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def get_patient_id(observation):
    ref = observation.get("subject", {}).get("reference", "")
    return ref.replace("Patient/", "") if ref else "Unknown"


def get_datetime(observation):
    dt = observation.get("effectiveDateTime", "")
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except:
        return dt or "Unknown"


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
                    formatted_value = format_value(value)
                    results.append(f"{label}: {formatted_value} {unit}")
    return results


def observation_parser(observation: Dict[str, Any]) -> str:
    lines = [
        f"Patient ID: {get_patient_id(observation)}",
        f"Measurement time: {get_datetime(observation)}"
    ]

    vitals = get_all_components(observation)
    if vitals:
        lines.extend(vitals)
    else:
        lines.append("(No available vital signs)")

    return "\n".join(lines)


def parser_fhir(bundle: Dict[str, Any]) -> str:
    # 1. 取得 Patient ID
    patient_id = None
    for entry in bundle.get('entry', []):
        res = entry.get('resource', {})
        if res.get('resourceType') == 'Patient':
            # 優先 identifier.value，否則用 id
            identifiers = res.get('identifier', [])
            if identifiers:
                patient_id = identifiers[0].get('value')
            else:
                patient_id = res.get('id')
            break

    # 2. 處理所有 Observation entries
    outputs = []
    for entry in bundle.get('entry', []):
        res = entry.get('resource', {})
        if res.get('resourceType') != 'Observation':
            continue

        ts = res.get('effectiveDateTime', 'N/A')
        header = f"(Patient: {patient_id}, {ts})"
        lines = []

        for comp in res['component']:
            coding = comp.get('code', {}).get('coding', [{}])[0]
            label = coding.get('display', coding.get('code', 'Unknown'))
            qty = comp.get('valueQuantity', {})
            val = qty.get('value', 'N/A')
            # 格式化數值為兩位小數
            if val != 'N/A' and isinstance(val, (int, float)):
                val = format_value(val)
            unit = qty.get('unit', '')
            lines.append(f"- {label}: {val} {unit}")

        outputs.append("\n".join([header] + lines))

    return "\n\n".join(outputs)