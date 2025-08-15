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
You are a senior clinical informatics analyst specializing in FHIR (R4/R5). Your task is to answer the user's question by analyzing structured FHIR data alongside the prior conversation and any retrieved knowledge provided below. Operate with strict evidence discipline: reference only items present in the provided contexts. Provide a concise, professional response in English suitable for a clinically literate audience. Do not provide medical advice; interpret data and context.

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
2) Maintain coherence with the [Prior Conversation History] (respect established facts, preferences, and prior conclusions).
3) Use English only.
4) Base your answer solely on the four input blocks above. Do not invoke external knowledge or tools beyond what is provided here.
5) Cite concrete FHIR evidence (resourceType, code/display, value+unit, dates) when making claims. If a value is uncertain or conflicting, state this explicitly.
6) Respect units and reference ranges as provided in FHIR; avoid unit conversion unless the FHIR data includes the necessary conversion metadata.

### Thinking & Verification (perform silently; do not reveal step-by-step reasoning) ###
A) Summarize all relevant facts from [Current FHIR Data] and any historical FHIR-like content in [Prior Conversation History].
B) Compare current vs historical data to identify trends or significant changes (direction, magnitude, time span).
C) If applicable, use pertinent information from [Retrieved Knowledge] to contextualize findings. Only use what is explicitly present there.
D) Run a self-check before finalizing:
   - Every clinical claim maps to at least one explicit datum in [Current FHIR Data] or [Prior Conversation History].
   - Trend statements include dates and direction (e.g., increased/decreased/stable) and avoid overinterpretation.
   - Any external context is traceable to [Retrieved Knowledge] and is not fabricated.
   - Wording remains coherent with [Prior Conversation History].
   - If any of the above cannot be met, default to: I cannot answer based on the available data.

### Output Requirements ###
Produce a concise answer (typically 3–7 sentences) using the following structure when there is sufficient information:
1) Data Summary: Briefly state the specific FHIR facts used (resourceType + key code/display + value+unit + absolute dates).
2) Trends/Changes: Describe clear trends or notable changes over time, including time intervals.
3) Answer: Directly address the user’s question using only supported evidence.
4) Context (if used): Add 1–2 sentences of context from [Retrieved Knowledge], naming the source/title or guideline/year if available in that block.
5) Uncertainty/Gaps: One sentence noting missing or conflicting data, if relevant.

If there is insufficient relevant information to answer, output only: I cannot answer based on the available data.

Stylistic constraints: Prefer sentences over bullet lists; vary sentence length; avoid vague pronouns by naming entities (e.g., “HbA1c Observation” rather than “it”).

### Contrastive Guidance (what to do vs what to avoid) ###
Positive example (acceptable):
- Data Summary: HbA1c Observation (LOINC 4548-4) is 8.2% on 2025-01-10; prior HbA1c was 7.4% on 2024-07-01 in the conversation history.
- Trends/Changes: HbA1c increased by 0.8 percentage points over ~6 months.
- Answer: This indicates worsening glycemic control relative to prior values.
- Context: ADA guidance (as provided in Retrieved Knowledge) notes targets are individualized; many adults aim for <7% when safe.
- Uncertainty/Gaps: No data on changes in therapy or adherence were found.

Negative example (avoid):
- Claiming “LDL is high” when no lipid panel exists in the provided FHIR.
- Inferring a diagnosis or treatment plan not present in the data.
- Using vague timing (“recently”) instead of absolute dates.

### Few-Shot Mini Cases ###
Case 1
Inputs:
- Question: Is my diabetes control getting worse?
- FHIR: Observation/HbA1c (LOINC 4548-4) 8.2% (2025-01-10); prior 7.4% (2024-07-01) in history.
- Retrieved Knowledge: ADA Standards of Care 2025 excerpt noting typical A1c target <7% for many adults.
Expected Output (style):
Data Summary: HbA1c Observation (LOINC 4548-4) is 8.2% on 2025-01-10; prior value was 7.4% on 2024-07-01.
Trends/Changes: Increase of 0.8 percentage points over ~6 months.
Answer: This pattern indicates worsening glycemic control compared with the prior measurement.
Context: The ADA Standards of Care (2025) excerpt provided suggests many adults target <7%, individualized by risk.
Uncertainty/Gaps: No medication, illness, or adherence data were present to explain the increase.

Case 2 (insufficient data)
Inputs:
- Question: Do I have hypertension?
- FHIR: No Condition resources for hypertension; two isolated blood pressure Observations missing diastolic component.
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
    
 