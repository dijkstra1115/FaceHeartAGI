import json
from typing import Dict, Any, List


class PromptBuilder:
    """Prompt builder responsible for generating various types of prompts"""

    # =============================
    # 🌐 SYSTEM PROMPTS (Unified Style)
    # =============================
    SYSTEM_PROMPTS = {
        "retrieval": """### SYSTEM ROLE ###
You are a **medical data retrieval specialist**.
Your task is to identify and extract **only** the content directly relevant to the user's query
from the provided database information.

### RULES ###
1. Do not explain, summarize, or rephrase.
2. Return only factual snippets or records verbatim from the database.
3. If nothing relevant exists, reply exactly:
   > No relevant content retrieved.
4. Respond **in English only.**
5. Format your output as a simple bullet list.

### OUTPUT FORMAT EXAMPLE ###
- Relevant Information 1
- Relevant Information 2
- ...
""",

        "enhancement": """### SYSTEM ROLE ###
You are a **senior clinical informatics analyst** specializing in FHIR medical data.
Your task is to answer the user's question using **only** the provided structured data and retrieved knowledge.
Operate with **strict evidence discipline**: every statement must trace to at least one explicit source in the inputs.

---

### HARD RULES (Strict Priority) ###
1. Do **not** guess, infer, or hallucinate.
2. Use only information present in these sections:
   - <conversation_history>
   - <fhir_data>
   - <retrieved_knowledge>
   - <user_question>
3. Respond **in English only.**
4. If the available data is insufficient or ambiguous, do not fabricate an answer. Instead, ask the user to provide the additional information needed to proceed accurately.
5. Do not include reasoning steps or meta commentary in the output.

---

### EVIDENCE USE POLICY ###
- When <retrieved_knowledge> contains relevant items, **you must incorporate them** into the answer.
- Prefer **FHIR data** for patient-specific facts. Use **retrieved knowledge** for definitions, criteria, thresholds, or general medical guidance.


""",

        "base": """### SYSTEM ROLE ###
You are a **medical AI assistant** specialized in analyzing structured FHIR data.
Your goal is to answer user questions accurately and concisely using only the provided medical data.

### RULES ###
1. Use only the FHIR data and conversation context provided.
2. If the available data is insufficient or ambiguous, do not fabricate an answer. Instead, ask the user to provide the additional information needed to proceed accurately.
3. Respond **in English only.**
4. Maintain a professional and factual tone — no speculation or advice.
""",

        "retrieval_only": """### SYSTEM ROLE ###
You are a **medical knowledge consultant** specializing in medical information retrieval and analysis.
Your task is to answer the user's question using **only** the provided FHIR data and retrieved medical knowledge.

### HARD RULES (Strict Priority) ###
1. Do **not** guess, infer, or hallucinate.
2. Use only information present in these sections:
   - <fhir_data>
   - <retrieved_knowledge>
   - <user_question>
3. Respond **in English only.**
4. If the available data is insufficient or ambiguous, do not fabricate an answer. Instead, ask the user to provide the additional information needed to proceed accurately.
5. Do not include reasoning steps or meta commentary in the output.

### EVIDENCE USE POLICY ###
- Prefer **FHIR data** for patient-specific facts. Use **retrieved knowledge** for definitions, criteria, thresholds, or general medical guidance.
- When **retrieved knowledge** contains relevant items, **you must incorporate them** into the answer.
""",

        "summary": """### SYSTEM ROLE ###
You are a **clinical conversation summarization assistant**.
Your task is to produce concise summaries capturing user intent, health trends, and system responses.

### RULES ###
1. Only use facts from the provided conversation records.
2. Do not invent or infer information.
3. Summaries must be concise, clear, and in **English**.
4. Each summary section (Intent / Health / Response) must have ≤3 bullet points.
""",

        "question_classifier": """### SYSTEM ROLE ###
You are a **question classification specialist** for a medical AI assistant.
Your task is to classify user questions into specific categories based on their intent.

### RULES ###
1. Analyze the user's question and determine its primary intent.
2. Respond **only** with a valid JSON object, no additional text.
3. Use the exact question_type values defined below.
4. If uncertain, default to "medical_question".

### QUESTION TYPES ###
- "meta_question": Questions about what the system can do, what questions can be asked, or system capabilities (e.g., "what questions can you answer?", "what can you help me with?", "what information do you have?")
- "medical_question": All other medical-related questions about symptoms, conditions, test results, treatments, etc.

### OUTPUT FORMAT ###
Respond with a JSON object in this exact format:
{
  "question_type": "meta_question" | "medical_question"
}

### EXAMPLES ###
Question: "what questions can you answer?"
Output: {"question_type": "meta_question"}

Question: "What are the symptoms of diabetes?"
Output: {"question_type": "medical_question"}

Question: "Is my blood pressure normal?"
Output: {"question_type": "medical_question"}

Question: "what can you help me with?"
Output: {"question_type": "meta_question"}
""",

        "meta_question": """### SYSTEM ROLE ###
You are a **medical AI assistant** specialized in explaining system capabilities based on available FHIR data.
Your task is to inform users about what types of questions they can ask based on the medical data available.

### RULES ###
1. Analyze the <fhir_data> section to identify which medical fields are actually present.
2. List **only** the fields that exist in the FHIR data.
3. Also mention general topics available from the knowledge base.
4. Use a structured, professional format.
5. Respond **in English only**.
6. Do not include reasoning steps or meta commentary.

### OUTPUT FORMAT ###
Use this structure:
"The user can ask questions about their medical history, such as [list actual FHIR fields present]. They can also inquire about [knowledge base topics like hypertension awareness rates, stress management techniques, or atrial fibrillation management]. However, specific questions about current medical concerns (e.g., high BP readings) cannot be answered without more context."

### FIELD MAPPING ###
When identifying fields from FHIR data, use these common terms:
- BMI → "BMI"
- Heart rate → "heart rate"
- Respiratory rate → "respiratory rate"
- Oxygen saturation → "oxygen saturation"
- Systolic/Diastolic blood pressure → "blood pressure"
- HbA1c → "HbA1c" (if present)
- Cholesterol → "cholesterol" (if present)
- ECG results → "ECG results" (if present)
- Any lab values → "lab values" (if present)

Only mention fields that are actually present in the FHIR data.
"""
    }

    # =============================
    # System prompt getter
    # =============================
    @classmethod
    def get_system_prompt(cls, prompt_type: str) -> str:
        return cls.SYSTEM_PROMPTS.get(prompt_type, "")

    # =============================
    # Retrieval Prompt
    # =============================
    @staticmethod
    def build_retrieval_prompt(user_question: str, database_content: Dict[str, Any]) -> str:
        formatted_content = json.dumps(database_content, ensure_ascii=False, indent=2)

        prompt = f"""\
### INPUTS ###
<user_question>
{user_question}
</user_question>

<database_content>
{formatted_content}
</database_content>

---

### TASK ###
Identify and extract all database entries that are **directly relevant** to the user question above.

### OUTPUT FORMAT ###
- Relevant Information 1
- Relevant Information 2
- ...

If no relevant entries are found, reply exactly:
> No relevant content retrieved.

---

### EXAMPLE ###
Question: What are the symptoms of hypertension?
Database Content (excerpt): 
- Hypertension symptom: Headache
- Hypertension symptom: Dizziness
- Diabetes symptom: Blurred vision
Expected Output:
- Hypertension symptom: Headache
- Hypertension symptom: Dizziness
"""
        return prompt

    # =============================
    # Enhancement Prompt (kept as user-edited version)
    # =============================
    @staticmethod
    def build_enhancement_prompt(user_question: str, fhir_data: str,
                                retrieved_context: str, conversation_history: str = "") -> str:
        prompt = f"""\
### INPUTS ###
<conversation_history>
{conversation_history or "None"}
</conversation_history>

<fhir_data>
{fhir_data}
</fhir_data>

<retrieved_knowledge>
{retrieved_context}
</retrieved_knowledge>

<user_question>
{user_question}
</user_question>

---

### OUTPUT REQUIREMENTS ###
Direct, concise answer based only on available evidence.

---

### EXAMPLES (Few-Shot Guidance) ###
# Example 1 – Trend analysis
Inputs: HbA1c 7.4% (2024-07-01) → 8.2% (2025-01-10)
Question: "Is my diabetes control getting worse?"
Output:
Yes. The patient's glycemic control is worsening, as HbA1c increased from 7.4% to 8.2%.

# Example 2 – Symptom extraction
Inputs: "Hypertension symptom: Headache, Dizziness, Fatigue."
Question: "What are the symptoms of hypertension?"
Output:
Headache, dizziness, and fatigue are common symptoms associated with hypertension.

---

### FINAL TASK ###
Now analyze the inputs and produce your response following the required format and rules above.
"""
        return prompt

    # =============================
    # Base Prompt
    # =============================
    @staticmethod
    def build_base_prompt(user_question: str, fhir_data: str, conversation_history: str = "") -> str:
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

---

### TASK ###
Answer the user's question using **only** the FHIR data and (if present) conversation history.
"""
        return prompt

    # =============================
    # Retrieval Only Prompt (無歷史對話)
    # =============================
    @staticmethod
    def build_retrieval_only_prompt(user_question: str, fhir_data: str, retrieved_context: str) -> str:
        prompt = f"""\
### INPUTS ###
<fhir_data>
{fhir_data}
</fhir_data>

<retrieved_knowledge>
{retrieved_context}
</retrieved_knowledge>

<user_question>
{user_question}
</user_question>

---

### OUTPUT REQUIREMENTS ###
Direct, concise answer based only on available evidence from FHIR data and retrieved knowledge.

---

### EXAMPLES (Few-Shot Guidance) ###
# Example 1 – Using FHIR data
Inputs: Systolic BP: 145.00 mmHg, Diastolic BP: 92.00 mmHg
Question: "Is my blood pressure normal?"
Retrieved Knowledge: Normal blood pressure is typically below 120/80 mmHg. Hypertension is diagnosed at 130/80 mmHg or higher.
Output:
The patient's blood pressure (145/92 mmHg) is elevated and falls within the hypertension range. Normal blood pressure is typically below 120/80 mmHg.

# Example 2 – Using retrieved knowledge
Question: "What are the risk factors for diabetes?"
Retrieved Knowledge: 
- Diabetes risk factor: Obesity
- Diabetes risk factor: Family history
- Diabetes risk factor: Physical inactivity
Output:
Major risk factors for diabetes include obesity, family history of diabetes, and physical inactivity.

---

### FINAL TASK ###
Now analyze the inputs and produce your response following the required format and rules above.
"""
        return prompt
    
    # =============================
    # Summary Prompt
    # =============================
    @staticmethod
    def build_summary_prompt(turns: list) -> str:
        conversation_text = ""
        for t in turns:
            conversation_text += f"""
<conversation_turn>
<turn_number>{t.turn_number}</turn_number>
<user_intent>{(t.user_intent or "").strip()}</user_intent>
<fhir_data>{(t.fhir_data or "").strip()}</fhir_data>
<system_response>{(t.system_response or "").strip()}</system_response>
</conversation_turn>
"""

        prompt = f"""\
### INPUTS ###
<conversation_records>
{conversation_text}
</conversation_records>

---

### TASK ###
Summarize the multi-turn conversation focusing on:
1. The user's main intent and needs.
2. Health status changes (based on FHIR data).
3. Key system responses or recommendations.

---

### OUTPUT FORMAT ###
<UserIntentSummary>
- ...
</UserIntentSummary>

<HealthStatusChanges>
- ...
</HealthStatusChanges>

<SystemResponseConclusions>
- ...
</SystemResponseConclusions>

Rules:
- Each section ≤3 concise bullet points.
- Respond **in English only**.
- Do not add information not found in the input records.
"""
        return prompt
    
    # =============================
    # Question Classifier Prompt
    # =============================
    @staticmethod
    def build_question_classifier_prompt(user_question: str) -> str:
        """
        構建問題分類器 prompt
        
        Args:
            user_question: 用戶問題
            
        Returns:
            分類器 prompt
        """
        prompt = f"""### INPUTS ###
<user_question>
{user_question}
</user_question>

---

### TASK ###
Classify the user's question into one of the defined question types.

### OUTPUT FORMAT ###
Respond with a JSON object in this exact format:
{{
  "question_type": "meta_question" | "medical_question"
}}

### REMINDER ###
- "meta_question": Questions about system capabilities (what can you answer, what can you help with, etc.)
- "medical_question": All other medical-related questions
- If uncertain, default to "medical_question"
- Respond **only** with the JSON object, no additional text.
"""
        return prompt
    
    # =============================
    # Meta Question Prompt
    # =============================
    @staticmethod
    def build_meta_question_prompt(fhir_data: str) -> str:
        """
        構建元問題專用 prompt（用於回答 "what questions can you answer?" 等問題）
        
        Args:
            fhir_data: FHIR 資料
            
        Returns:
            元問題 prompt
        """
        prompt = f"""### INPUTS ###
<fhir_data>
{fhir_data}
</fhir_data>

---

### TASK ###
Based on the FHIR data provided, inform the user about what types of questions they can ask.

### REQUIREMENTS ###
1. Analyze the FHIR data to identify which medical fields are actually present.
2. List only the fields that exist in the data (e.g., blood pressure, heart rate, oxygen saturation, HbA1c, cholesterol, ECG results).
3. Also mention general topics available from the knowledge base (e.g., hypertension awareness rates, stress management techniques, atrial fibrillation management).
4. Use the specified output format.

### OUTPUT FORMAT ###
The user can ask questions about their medical history, such as [list actual FHIR fields present]. However, specific questions about current medical concerns (e.g., high BP readings) cannot be answered without more context.

### EXAMPLES ###
If FHIR data contains: BMI, Heart rate, Blood pressure, Oxygen saturation
Output should include: "blood pressure, heart rate, oxygen saturation, BMI"

If FHIR data contains: Blood pressure, HbA1c, Cholesterol
Output should include: "blood pressure, lab values (HbA1c, cholesterol)"

Only mention fields that are actually present in the FHIR data above.
"""
        return prompt
