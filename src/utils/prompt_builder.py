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
4. If the available data is insufficient or ambiguous, reply **exactly** with:
   > I cannot answer based on the available data.
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
2. If information is insufficient, respond:
   > I cannot answer based on the available data.
3. Respond **in English only.**
4. Maintain a professional and factual tone — no speculation or advice.
""",

        "summary": """### SYSTEM ROLE ###
You are a **clinical conversation summarization assistant**.
Your task is to produce concise summaries capturing user intent, health trends, and system responses.

### RULES ###
1. Only use facts from the provided conversation records.
2. Do not invent or infer information.
3. Summaries must be concise, clear, and in **English**.
4. Each summary section (Intent / Health / Response) must have ≤3 bullet points.
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
"""
        return prompt

    # =============================
    # Base Prompt
    # =============================
    @staticmethod
    def build_base_prompt(user_question: str, fhir_data: str, conversation_history: str = "") -> str:
        history_section = f"<conversation_history>\n{conversation_history}\n</conversation_history>" if conversation_history else "<conversation_history />"

        prompt = f"""\
### INPUTS ###
{history_section}

<user_question>
{user_question}
</user_question>

<fhir_data>
{fhir_data}
</fhir_data>

---

### TASK ###
Answer the user's question using **only** the FHIR data and (if present) conversation history.

### OUTPUT FORMAT ###
Answer: <concise, factual statement>
Context: <optional, 1–2 sentence explanation>

If insufficient data is available, reply exactly:
> I cannot answer based on the available data.
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
