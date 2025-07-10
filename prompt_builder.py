import json
from typing import Dict, Any, List


class PromptBuilder:
    """提示詞建構器，負責生成各種類型的提示詞"""
    
    # 系統提示詞
    SYSTEM_PROMPTS = {
        "retrieval": "你是一個專業的醫療資料檢索助手。你的任務是從提供的資料庫內容中找出與用戶問題最相關的資訊。請只返回相關的內容，不要添加額外的解釋。",
        "enhancement": "你是一個專業的醫療 AI 助手。請根據提供的 FHIR 資料和檢索到的相關內容，一步一步思考用戶問題，生成一個精簡、準確的回答。",
        "base": "你是一個專業的醫療 AI 助手，專門協助分析 FHIR 格式的醫療資料。請根據提供的醫療資料回答問題，並提供準確的醫療資訊。",
        "summary": "你是一個專業的對話摘要助手。請一步一步分析對話紀錄，生成簡潔的摘要，只包含用戶意圖和系統回應結論。"
    }
    
    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        """
        獲取指定類型的系統提示詞
        
        Args:
            prompt_type: 提示詞類型 ("retrieval", "enhancement", "base", "summary")
            
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
    def build_enhancement_prompt(user_question: str, fhir_data: str, 
                                retrieved_context: str, conversation_history: str = "") -> str:
        """
        建構增強提示詞
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            retrieved_context: 檢索到的相關內容
            conversation_history: 對話歷史（可選）
            
        Returns:
            增強提示詞
        """
        # 如果有對話歷史，加入到提示詞中
        history_section = ""
        if conversation_history:
            history_section = f"""
### 對話歷史 ###
{conversation_history}
"""

        prompt = f"""
### 回應規則 ###
1. 避免回答不存在於檢索內容中的問題
2. 若無法回答，請誠實告知用戶
3. 如果當前問題與之前的對話相關，請適當引用或延續之前的建議
4. 使用中文回答

{history_section}

### 用戶問題 ###
{user_question}

### FHIR 資料 ###
{fhir_data}

### 檢索內容 ###
{retrieved_context}

### 思考過程 ###
1. 整合 FHIR 資料和檢索內容
2. 判斷用戶問題是否在檢索內容中
3. 考慮對話歷史，確保回應的連貫性

### 目標 ###
1. 請遵循 ### 思考過程 ### 的步驟，保持專業性和準確性，生成一個精簡、準確的回答
2. 確保嚴格遵守 ### 回應規則 ###
"""
        return prompt
    
    @staticmethod
    def build_base_prompt(user_question: str, fhir_data: str, conversation_history: str = "") -> str:
        """
        建構基礎提示詞（無檢索內容時使用）
        
        Args:
            user_question: 用戶問題
            fhir_data: FHIR 資料
            conversation_history: 對話歷史（可選）
            
        Returns:
            基礎提示詞
        """
        # 如果有對話歷史，加入到提示詞中
        history_section = ""
        if conversation_history:
            history_section = f"""
### 對話歷史 ###
{conversation_history}
"""

        return f"""{history_section}\n\n### 用戶問題 ###\n{user_question}\n\n### FHIR 資料 ###\n{fhir_data}

### 目標 ###
請根據以上資訊回答問題，如果有對話歷史，請確保回應的連貫性。"""

    @staticmethod
    def build_summary_prompt(conversations: List[Dict[str, Any]]) -> str:
        """
        構建摘要生成的提示詞
        
        Args:
            conversations: 對話記錄列表
            
        Returns:
            摘要提示詞
        """
        conversation_text = ""
        for i, conv in enumerate(conversations, 1):
            
            conversation_text += f"""
[第 {conv['turn_number']} 輪對話]\n
**用戶意圖**\n{conv['user_intent']}\n
**FHIR 資料**\n{conv['fhir_data']}\n
**系統回應**\n{conv['system_response'][:200]}{'...' if len(conv['system_response']) > 200 else ''}

"""
        
        prompt = f"""
### 對話記錄 ###
{conversation_text}

### 輸出格式 ###
**用戶意圖摘要:**
- [簡潔描述用戶在這5輪對話中的主要意圖和需求]

**健康狀況變化:**
- [基於FHIR資料分析用戶健康狀況的變化趨勢]

**系統回應結論:**
- [簡潔描述系統提供的主要建議和結論]

### 目標 ###
1. 請根據 ### 對話記錄 ### 生成摘要。
2. 確保摘要符合 ### 輸出格式 ### 規則。
3. 請保持摘要簡潔明瞭，每部分不超過3個要點。
4. 請使用中文回答。
"""
        return prompt
    
 