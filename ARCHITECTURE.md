# FaceHeartAGI 架構說明

## 重構後的架構概覽

本次重構主要針對程式可讀性和維護性進行優化，將原本複雜的 `rag_client.py` 拆分成多個專門的模組。

## 模組結構

### 1. `rag_client.py` - 主要客戶端
- **職責**: 提供 RAG 系統的主要介面
- **功能**: 
  - 初始化檢索策略
  - 協調檢索和生成流程
  - 提供 streaming 和非 streaming 介面
- **優化**: 大幅簡化，專注於流程協調

### 2. `prompt_builder.py` - 提示詞建構器
- **職責**: 專門處理各種提示詞的建構和管理
- **功能**:
  - `SYSTEM_PROMPTS`: 系統提示詞字典
  - `get_system_prompt()`: 獲取指定類型的系統提示詞
  - `build_retrieval_prompt()`: 建構檢索提示詞
  - `build_enhancement_prompt()`: 建構增強提示詞
  - `build_base_prompt()`: 建構基礎提示詞
  - `_format_database_content()`: 格式化資料庫內容
- **優化**: 將所有提示詞邏輯和系統提示詞集中管理，便於維護和調整

### 3. `config.py` - 配置管理
- **職責**: 統一管理系統配置
- **功能**:
  - API 配置（URL、模型、標頭等）
  - 請求參數（max_tokens、temperature 等）
  - 向量檢索配置
- **優化**: 將硬編碼的常數提取出來，便於配置管理

### 4. `retrieval_strategies.py` - 檢索策略
- **職責**: 實現不同的檢索策略
- **功能**:
  - `RetrievalStrategy`: 抽象基類
  - `VectorRetrievalStrategy`: 向量檢索策略
  - `LLMRetrievalStrategy`: LLM 檢索策略
- **優化**: 使用策略模式，便於擴展新的檢索方法

## 主要優化點

### 1. **單一職責原則**
- 每個模組都有明確的職責
- 降低了模組間的耦合度
- 提高了程式碼的可測試性

### 2. **策略模式**
- 檢索策略可以動態切換
- 便於添加新的檢索方法
- 提高了系統的擴展性

### 3. **配置集中化**
- 所有配置都在一個地方管理
- 便於環境切換和參數調整
- 減少了硬編碼

### 4. **提示詞管理分離**
- 提示詞邏輯和系統提示詞獨立管理
- 便於 A/B 測試和優化
- 提高了可維護性

### 5. **錯誤處理簡化**
- 使用簡單的 try-except 和 logger.error()
- 保持程式碼簡潔易懂
- 避免過度工程化

## 使用方式

### 基本使用
```python
from rag_client import RAGClient

# 初始化客戶端
rag_client = RAGClient(use_vector_search=True)

# 使用 RAG 增強回應
async for chunk in rag_client.enhance_response_with_rag_stream(
    user_question, fhir_data, database_content
):
    print(chunk, end='')
```

### 切換檢索策略
```python
# 切換到 LLM 檢索
rag_client.switch_to_llm_search()

# 切換到向量檢索
rag_client.switch_to_vector_search()
```

### 獲取統計信息
```python
# 獲取向量資料庫統計
stats = rag_client.get_vector_store_stats()
```

### 使用提示詞建構器
```python
from prompt_builder import PromptBuilder

# 獲取系統提示詞
system_prompt = PromptBuilder.get_system_prompt("retrieval")

# 建構檢索提示詞
retrieval_prompt = PromptBuilder.build_retrieval_prompt(
    user_question, database_content
)
```

## 擴展指南

### 添加新的檢索策略
1. 在 `retrieval_strategies.py` 中繼承 `RetrievalStrategy`
2. 實現 `retrieve()` 方法
3. 在 `RAGClient` 中添加切換方法

### 添加新的提示詞類型
1. 在 `prompt_builder.py` 中添加新的靜態方法
2. 在 `SYSTEM_PROMPTS` 中添加對應的系統提示詞
3. 在需要的地方調用新的提示詞建構方法

### 修改配置
1. 在 `config.py` 中修改對應的常數
2. 所有使用該配置的地方會自動更新

### 修改系統提示詞
1. 在 `prompt_builder.py` 的 `SYSTEM_PROMPTS` 中修改
2. 使用 `get_system_prompt()` 方法獲取

## 測試建議

### 單元測試
- 每個模組都應該有對應的單元測試
- 特別關注策略模式的測試
- 測試錯誤處理機制

### 整合測試
- 測試完整的 RAG 流程
- 測試策略切換功能
- 測試 streaming 和非 streaming 模式

### 性能測試
- 比較不同檢索策略的性能
- 測試向量資料庫的擴展性
- 監控記憶體使用情況 