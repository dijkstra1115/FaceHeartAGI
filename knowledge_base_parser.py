from typing import Dict, Any, List

def extract_medical_documents(medical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    從醫療資料中提取文檔
    
    Args:
        medical_data: 醫療資料
        
    Returns:
        文檔列表
    """
    documents = []
    
    # 處理 medical_guidelines
    for guideline in medical_data.get("medical_guidelines", []):
        condition = guideline.get("condition")
        
        # 疾病描述
        if guideline.get("description"):
            content = f"【{condition}】疾病說明：{guideline['description']}"
            documents.append({
                'content': content,
                'metadata': {
                    'condition': condition,
                    'topic_type': 'description'
                }
            })

        # 推薦建議分類處理
        for topic, suggestions in guideline.get("recommendations", {}).items():
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
        if guideline.get("risk_factors"):
            for risk_factor in guideline["risk_factors"]:
                content = f"【{condition}】風險因子：{risk_factor}"
                documents.append({
                    'content': content,
                    'metadata': {
                        'condition': condition,
                        'topic_type': 'risk_factors'
                    }
                })

        # 併發症
        if guideline.get("complications"):
            for complication in guideline["complications"]:
                content = f"【{condition}】併發症：{complication}"
                documents.append({
                    'content': content,
                    'metadata': {
                        'condition': condition,
                        'topic_type': 'complications'
                    }
                })

        # domestic
        if guideline.get("domestic"):
            for sentence in guideline.get("domestic", []):
                content = f"【{condition}】台灣資料：{sentence}"
                documents.append({
                    'content': content,
                    'metadata': {
                        'condition': condition,
                        'topic_type': 'domestic_context'
                    }
                })

        # ethnic
        if guideline.get("ethnic"):
            for sentence in guideline.get("ethnic", []):
                content = f"【{condition}】族群資料：{sentence}"
                documents.append({
                    'content': content,
                    'metadata': {
                        'condition': condition,
                        'topic_type': 'ethnic_context'
                    }
                })
    
    return documents 