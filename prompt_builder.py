import json
from typing import Dict, Any


class PromptBuilder:
    """提示詞建構器，負責生成各種類型的提示詞"""
    
    # 系統提示詞
    SYSTEM_PROMPTS = {
        "retrieval": "你是一個專業的醫療資料檢索助手。你的任務是從提供的資料庫內容中找出與用戶問題最相關的資訊。請只返回相關的內容，不要添加額外的解釋。",
        "enhancement": "你是一個專業的醫療 AI 助手。請根據提供的 FHIR 資料和檢索到的相關內容，生成一個完整、準確的回答。嚴格遵守: 避免回答不存在於檢索內容中的問題。",
        "base": "你是一個專業的醫療 AI 助手，專門協助分析 FHIR 格式的醫療資料。請根據提供的醫療資料回答問題，並提供準確、有用的醫療資訊。"
    }
    
    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        """
        獲取指定類型的系統提示詞
        
        Args:
            prompt_type: 提示詞類型 ("retrieval", "enhancement", "base")
            
        Returns:
            系統提示詞
        """
        return cls.SYSTEM_PROMPTS.get(prompt_type, "")
    
    @staticmethod
    def build_retrieval_prompt(user_question: str, database_content: Dict[str, Any]) -> str:
        """
        建構檢索提示詞
        
        Args:
            user_question: 用戶問題
            database_content: 資料庫內容
            
        Returns:
            檢索提示詞
        """
        formatted_content = json.dumps(database_content, ensure_ascii=False, indent=2)
        
        prompt = f"""
用戶問題: {user_question}

請從以下資料庫內容中找出與用戶問題最相關的資訊：

資料庫內容:
{formatted_content}

請只返回與用戶問題直接相關的內容，格式如下：
- 相關資訊 1
- 相關資訊 2
- ...

如果沒有找到相關內容，請返回 "沒有檢索到相關內容"。
"""
        return prompt
    
    @staticmethod
    def build_enhancement_prompt(user_question: str, fhir_data: Dict[str, Any], 
                                retrieved_context: str) -> str:
        """
        建構增強提示詞
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            retrieved_context: 檢索到的相關內容
            
        Returns:
            增強提示詞
        """
        prompt = f"""
嚴格遵守以下規則（優先級最高）：
1. 避免回答不存在於檢索內容中的問題
2. 使用中文回答

用戶問題: {user_question}

FHIR 醫療資料:
{json.dumps(fhir_data, ensure_ascii=False, indent=2)}

從資料庫檢索到的相關內容:
{retrieved_context}

請根據以上資訊，生成一個完整、準確的回答。要求：
1. 整合 FHIR 資料和檢索到的相關內容
2. 保持專業性和準確性
3. 提供實用的醫療建議

請提供完整的回應：
"""
        return prompt
    
    @staticmethod
    def build_base_prompt(user_question: str, fhir_data: Dict[str, Any]) -> str:
        """
        建構基礎提示詞（無檢索內容時使用）
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            
        Returns:
            基礎提示詞
        """
        return f"用戶問題: {user_question}\n\nFHIR 醫療資料:\n{json.dumps(fhir_data, ensure_ascii=False, indent=2)}" 