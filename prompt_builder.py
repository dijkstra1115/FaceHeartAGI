import json
from typing import Dict, Any, List


class PromptBuilder:
    """Prompt builder responsible for generating various types of prompts"""
    
    # System prompts
    SYSTEM_PROMPTS = {
        "retrieval": "You are a professional medical data retrieval assistant. Your task is to find the most relevant information from the provided database content based on user questions. Please only return relevant content without adding additional explanations.",
        "enhancement": "You are a professional medical AI assistant. Please analyze user questions step by step based on the provided FHIR data and retrieved relevant content to generate a concise and accurate response.",
        "base": "You are a professional medical AI assistant specialized in analyzing FHIR format medical data. Please answer questions based on the provided medical data and provide accurate medical information.",
        "summary": "You are a professional conversation summary assistant. Please analyze conversation records step by step to generate concise summaries containing only user intent and system response conclusions."
    }
    
    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        """
        Get system prompt of specified type
        
        Args:
            prompt_type: Prompt type ("retrieval", "enhancement", "base", "summary")
            
        Returns:
            System prompt
        """
        return cls.SYSTEM_PROMPTS.get(prompt_type, "")
    
    @staticmethod
    def build_retrieval_prompt(user_question: str, database_content: Dict[str, Any]) -> str:
        """
        Build retrieval prompt
        
        Args:
            user_question: User question
            database_content: Database content
            
        Returns:
            Retrieval prompt
        """
        formatted_content = json.dumps(database_content, ensure_ascii=False, indent=2)
        
        prompt = f"""
User Question: {user_question}

Please find the most relevant information from the following database content:

Database Content:
{formatted_content}

Please only return content directly related to the user question in the following format:
- Relevant Information 1
- Relevant Information 2
- ...

If no relevant content is found, please return "No relevant content retrieved."
"""
        return prompt
    
    @staticmethod
    def build_enhancement_prompt(user_question: str, fhir_data: str, 
                                retrieved_context: str, conversation_history: str = "") -> str:
        """
        Build enhancement prompt
        
        Args:
            user_question: User question
            fhir_data: FHIR data
            retrieved_context: Retrieved relevant content
            conversation_history: Conversation history (optional)
            
        Returns:
            Enhancement prompt
        """
        # Add conversation history to prompt if available
        history_section = ""
        if conversation_history:
            history_section = f"""
### Conversation History ###
{conversation_history}
"""

        prompt = f"""
### Response Rules ###
1. Avoid answering questions that are not present in the retrieved content
2. If unable to answer, please honestly inform the user
3. If the current question is related to previous conversations, please appropriately reference or continue previous suggestions
4. Use English to respond

{history_section}

### User Question ###
{user_question}

### FHIR Data ###
{fhir_data}

### Retrieved Content ###
{retrieved_context}

### Thinking Process ###
1. Integrate FHIR data and retrieved content
2. Determine if the user question is within the retrieved content
3. Consider conversation history to ensure response coherence

### Objective ###
1. Please follow the steps in ### Thinking Process ###, maintain professionalism and accuracy, and generate a concise and accurate response
2. Ensure strict adherence to ### Response Rules ###
"""
        return prompt
    
    @staticmethod
    def build_base_prompt(user_question: str, fhir_data: str, conversation_history: str = "") -> str:
        """
        Build base prompt (used when no retrieved content is available)
        
        Args:
            user_question: User question
            fhir_data: FHIR data
            conversation_history: Conversation history (optional)
            
        Returns:
            Base prompt
        """
        # Add conversation history to prompt if available
        history_section = ""
        if conversation_history:
            history_section = f"""
### Conversation History ###
{conversation_history}
"""

        return f"""{history_section}\n\n### User Question ###\n{user_question}\n\n### FHIR Data ###\n{fhir_data}

### Objective ###
Please answer the question based on the above information. If there is conversation history, please ensure response coherence."""

    @staticmethod
    def build_summary_prompt(conversations: List[Dict[str, Any]]) -> str:
        """
        Build summary generation prompt
        
        Args:
            conversations: List of conversation records
            
        Returns:
            Summary prompt
        """
        conversation_text = ""
        for i, conv in enumerate(conversations, 1):
            
            conversation_text += f"""
[Turn {conv['turn_number']}]\n
**User Intent**\n{conv['user_intent']}\n
**FHIR Data**\n{conv['fhir_data']}\n
**System Response**\n{conv['system_response'][:200]}{'...' if len(conv['system_response']) > 200 else ''}
"""
        
        prompt = f"""
### Conversation Records ###
{conversation_text}

### Output Format ###
**User Intent Summary:**
- [Concise description of the user's main intent and needs in these 5 turns of conversation]

**Health Status Changes:**
- [Analysis of user's health status change trends based on FHIR data]

**System Response Conclusions:**
- [Concise description of the main recommendations and conclusions provided by the system]

### Objective ###
1. Please generate a summary based on ### Conversation Records ###.
2. Ensure the summary complies with ### Output Format ### rules.
3. Please keep the summary concise and clear, with no more than 3 points per section.
4. Please respond in English.
"""
        return prompt
    
 