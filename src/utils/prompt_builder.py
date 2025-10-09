import json
from typing import Dict, Any, List


class PromptBuilder:
    """Prompt builder responsible for generating various types of prompts"""
    
    # System prompts
    SYSTEM_PROMPTS = {
        "retrieval": "You are a professional medical data retrieval assistant. Your task is to find the most relevant information from the provided database content based on user questions. Please only return relevant content without adding additional explanations.",
        "enhancement": """### SYSTEM ROLE ###
You are a **senior clinical informatics analyst** specializing in FHIR (R4/R5) medical data.
Your task is to answer the user's question using **only** the provided structured data and retrieved context.
Operate with **strict evidence discipline**: every statement must trace to at least one explicit source in the inputs.

---

### HARD RULES (Strict Priority) ###
1. Do **not** guess, infer, or hallucinate.
2. Use only information present in these sections:
   - <conversation_history>
   - <user_question>
   - <fhir_data>
   - <retrieved_knowledge>
3. Respond **in English only.**
4. If the available data is insufficient or ambiguous, reply **exactly** with:
   > I cannot answer based on the available data.
5. Do not include reasoning steps or meta commentary in the output.""",
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
        
        prompt = f"""\
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

        prompt = f"""\
### INPUTS ###
<conversation_history>
{conversation_history or "None"}
</conversation_history>

<user_question>
{user_question}
</user_question>

<fhir_data>
{fhir_data}
</fhir_data>

<retrieved_knowledge>
{retrieved_context}
</retrieved_knowledge>

---

### OUTPUT REQUIREMENTS ###
Use this JSON-like format for clarity and downstream parsing:
Answer: <direct, concise answer based only on available evidence>
Context: <1–2 sentences of supporting context, referencing data or retrieved content>
- Keep the answer 3–7 sentences maximum.
- Avoid medical advice, prescriptions, or unverified statements.
- Use explicit dates, numeric values, and FHIR references when available.

---

### EXAMPLES (Few-Shot Guidance) ###
# Example 1 – Trend analysis
Inputs: HbA1c 7.4% (2024-07-01) → 8.2% (2025-01-10)
Question: "Is my diabetes control getting worse?"
Output:
Answer: Yes. The patient's glycemic control is worsening, as HbA1c increased from 7.4% to 8.2%.
Context: This interpretation is supported by the ADA Standards of Care (target <7% for most adults).

# Example 2 – Symptom extraction
Inputs: "Hypertension symptom: Headache, Dizziness, Fatigue."
Question: "What are the symptoms of hypertension?"
Output:
Answer: Headache, dizziness, and fatigue are common symptoms associated with hypertension.
Context: These symptoms are explicitly listed in the provided retrieved knowledge.

# Example 3 – Insufficient data
Inputs: Missing Condition resources for hypertension.
Question: "Do I have hypertension?"
Output:
I cannot answer based on the available data.

---

### FINAL TASK ###
Now analyze the inputs and produce your response following the required format and rules above.
Return **only** the final answer block (no reasoning or commentary).
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
    
 