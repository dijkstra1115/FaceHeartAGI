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

        prompt = f"""### System / Role ###
You are a senior clinical informatics analyst specializing in FHIR (R4/R5). Your task is to answer the user's question by analyzing structured FHIR data alongside the prior conversation and any retrieved knowledge provided below. Operate with strict evidence discipline: reference only items present in the provided contexts.

### Inputs ###
[Prior Conversation History]
{conversation_history}

[Current User Question]
{user_question}

[Current FHIR Data]
{fhir_data}

[Retrieved Knowledge]
{retrieved_context}

### Hard Rules (Strict) ###
1) Do not guess or hallucinate. If no relevant information is available to answer the question, respond exactly: I cannot answer based on the available data.
2) Base your answer solely on the four input blocks above. Do not invoke external knowledge or tools beyond what is provided here.
3) Use English only.

### Thinking & Verification (perform silently; do not reveal step-by-step reasoning) ###
A) If applicable, use pertinent information from [Retrieved Knowledge] to contextualize findings. Only use what is explicitly present there.
B) Run a self-check before finalizing:
   - Every response maps to at least one explicit datum in [Retrieved Knowledge] or [Current FHIR Data] or [Prior Conversation History].
   - Do not make up any information.

### Output Requirements ###
Produce a concise answer (typically 3–7 sentences) using the following structure when there is sufficient information:
1) Answer: Directly address the user’s question using only supported evidence.
2) Context (if used): Add 1–2 sentences of context from [Retrieved Knowledge].

If there is insufficient relevant information to answer, output only: I cannot answer based on the available data.

### Contrastive Guidance (what to do vs what to avoid) ###
Positive example (acceptable):
- Answer: This indicates worsening glycemic control relative to prior values.
- Context: ADA guidance (as provided in Retrieved Knowledge) notes targets are individualized; many adults aim for <7% when safe.

Negative example (avoid):
- Claiming “LDL is high” when no lipid panel exists in the provided FHIR.
- Inferring a diagnosis or treatment plan not present in the data.
- Using vague timing (“recently”) instead of absolute dates.

### Few-Shot Mini Cases ###
Case 1 (contextualized answer)
Inputs:
- Prior Conversation History: HbA1c 7.4% (2024-07-01)
- Current User Question: Is my diabetes control getting worse?
- Current FHIR Data: HbA1c 8.2% (2025-01-10).
- Retrieved Knowledge: ADA Standards of Care 2025 excerpt noting typical A1c target <7% for many adults.
Expected Output (style):
Answer: This pattern indicates worsening glycemic control compared with the prior measurement.
Context: 
1) Your current HbA1c 8.2% (2025-01-10) is higher than the prior HbA1c 7.4% (2024-07-01).
2) Retrieved Knowledge: ADA Standards of Care (2025) excerpt provided suggests many adults target <7%, individualized by risk.

Case 2 (contextualized answer)
Inputs:
- Current User Question: What are the symptoms of hypertension?
- Retrieved Knowledge: Hypertension symptom: headache, Hypertension symptom: Dizziness, Hypertension symptom: Fatigue, Diabetes symptom: Blurred vision
Expected Output (style):
Answer: Headache, Dizziness, Fatigue.
Context: 
1) Retrieved Knowledge: Hypertension symptom: headache, Hypertension symptom: Dizziness, Hypertension symptom: Fatigue
2) Your current user question is about the symptoms of hypertension.

Case 3 (insufficient data)
Inputs:
- Current User Question: Do I have hypertension?
- Current FHIR Data: No Condition resources for hypertension; two isolated blood pressure Observations missing diastolic component.
Expected Output:
I cannot answer based on the available data.

### Final Step ###
Now analyze the inputs and produce your response following all rules above. Do not include your intermediate reasoning or the checklist; return only the final answer in the specified structure or the default insufficiency sentence when necessary.
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
    def build_summary_prompt(conversations: list) -> str:
        """
        Build summary generation prompt

        Args:
            conversations: List of ConversationTurn ORM objects
        Returns:
            Summary prompt
        """
        conversation_text = ""
        for i, conv in enumerate(conversations, 1):
            conversation_text += f"""
[Turn {conv.turn_number}]
**User Intent**
{conv.user_intent}
**FHIR Data**
{conv.fhir_data}
**System Response**
{conv.system_response}
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
    
 