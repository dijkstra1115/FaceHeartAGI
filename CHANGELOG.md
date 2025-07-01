# 更新日誌

## [3.0.0] - 2024-01-15

### 🚀 重大重構
- **僅保留異步串流功能**: 刪除所有非串流和同步串流功能
- **統一 API 接口**: 簡化請求模型，統一使用 `MedicalAnalysisRequest`
- **預設知識庫模板**: 內建醫療知識庫，無需額外設定
- **檢索類型切換**: 支援 `llm` 和 `vector` 檢索模式

### ✨ 新增功能
- 新增預設知識庫模板，包含高血壓和糖尿病指南
- 支援檢索類型動態切換
- 簡化的 API 端點設計
- 統一的錯誤處理機制

### 🔧 API 變更
- **主要端點**: `/analyze-stream` (原 `/enhanced-analyze-stream`)
- **檢索端點**: `/rag-retrieve` (非串流模式)
- **請求模型**: 統一使用 `MedicalAnalysisRequest`
- **參數變更**: 
  - `database_content` → `knowledge_base` (可選)
  - 新增 `retrieval_type` 參數
  - 移除 `enable_rag` 參數

### 🗑️ 移除功能
- 刪除所有非串流端點 (`/enhanced-analyze`, `/analyze`, `/health-summary`, `/rag-retrieve`)
- 刪除同步串流功能 (`generate_response_stream_sync`)
- 刪除舊版測試文件 (`test_api.py`, `example_usage.py`)
- 刪除性能比較工具 (`performance_comparison.py`)
- 刪除範例腳本 (`example_vector_rag.py`, `test_vector_rag.py`)
- 刪除啟動腳本 (`start_server.py`)

### 📁 文件變更
- 更新 `README.md` 以反映新架構
- 更新 `test_streaming_api.py` 以測試新接口
- 刪除不必要的測試和範例文件
- 簡化啟動流程，直接使用 `main.py`

### 🔄 向後兼容性
- **不向後兼容**: 這是一個重大版本更新，需要更新客戶端代碼
- 舊版 API 端點已完全移除
- 請求格式已重新設計

### 🚀 新功能
- **策略模式檢索**: 實現向量檢索和 LLM 檢索的動態切換
- **模組化重構**: 將提示詞管理、配置管理、檢索策略分離到獨立模組
- **非串流 RAG 檢索**: 新增 `/rag-retrieve` 端點，提供快速檢索結果
- **增強錯誤處理**: 簡化錯誤處理邏輯，提升穩定性
- **資料分離**: 將知識庫和示例資料分離到獨立 JSON 檔案

### 🔧 改進
- **代碼可讀性**: 大幅提升代碼結構和可維護性
- **配置管理**: 統一管理 API 配置和請求參數
- **提示詞管理**: 集中管理所有提示詞模板
- **檢索策略**: 實現策略模式，便於擴展新的檢索方法
- **資料管理**: 知識庫和示例資料獨立管理，便於維護和更新

### 📝 變更
- **檢索端點**: `/rag-retrieve` (非串流模式)
- **醫療分析**: `/analyze-stream` (保持串流模式)
- **模組結構**: 新增 `prompt_builder.py`、`config.py`、`retrieval_strategies.py`
- **資料檔案**: 新增 `default_knowledge_base.json`、`custom_knowledge_base.json`、`sample_fhir_data.json`
- **移除功能**: 刪除流式 RAG 檢索，簡化架構

### 🐛 修復
- 修復向量檢索中的潛在錯誤
- 改善錯誤處理和日誌記錄

---

## [2.1.0] - 2024-01-10

### ✨ 新增功能
- 新增 Server-Sent Events (SSE) 格式的串流回應
- 新增 `/enhanced-analyze-stream` 端點
- 新增 `/rag-retrieve` 端點（非串流模式）
- 新增 `/llm-stream` 端點
- 支援異步和同步串流模式

### 🔧 改進
- 優化串流回應的錯誤處理
- 改進 RAG 串流增強功能
- 新增串流回應格式化函數

### 📚 文檔
- 更新 API 文檔以包含串流端點
- 新增串流回應格式說明

---

## [2.0.0] - 2024-01-05

### 🚀 重大更新
- 新增 RAG (Retrieval-Augmented Generation) 功能
- 新增 FAISS 向量檢索支援
- 新增 `/enhanced-analyze` 端點
- 新增 `/rag-retrieve` 端點

### ✨ 新功能
- 支援向量檢索和傳統檢索
- 醫療知識庫檢索
- 回應增強功能
- 向量資料庫管理

### 🔧 改進
- 優化 FHIR 資料解析
- 改進提示詞建構
- 新增錯誤處理機制

### 📚 文檔
- 新增 RAG 功能說明
- 新增向量檢索使用指南
- 更新 API 文檔

### 🚀 新功能
- **雙重檢索模式**: 支援向量檢索和 LLM 檢索
- **異步串流回應**: 實現 Server-Sent Events (SSE) 串流
- **FHIR 資料整合**: 完整的 FHIR 醫療資料處理
- **自定義知識庫**: 支援動態知識庫配置

### 🔧 改進
- **效能優化**: 提升檢索速度和回應品質
- **錯誤處理**: 增強錯誤處理機制
- **API 文檔**: 完整的 Swagger UI 文檔

### 📝 變更
- **新增端點**: `/analyze-stream` 醫療分析串流
- **新增端點**: `/rag-retrieve` 端點
- **API 格式**: 統一的 JSON 回應格式

---

## [1.0.0] - 2024-01-01

### 🎉 初始版本
- 基礎 FHIR 資料解析功能
- DeepSeek LLM 整合
- 基本醫療分析功能
- FastAPI 框架支援

### 🚀 初始版本
- **基礎 RAG 功能**: 向量檢索和生成
- **FastAPI 框架**: 現代化的 Web API
- **OpenRouter 整合**: 支援多種 LLM 模型
- **向量資料庫**: ChromaDB 整合

---

## 版本命名規則

- **主版本號**: 重大功能更新或架構變更
- **次版本號**: 新功能添加
- **修訂版本號**: 錯誤修復和小幅改進

## 升級指南

### 從 v2.0.0 升級到 v2.1.0

1. **安裝新依賴**: 運行 `pip install -r requirements.txt` 安裝 FAISS 相關依賴
2. **向量資料庫初始化**: 首次使用時會自動下載句子嵌入模型
3. **功能使用**: 
   - 預設啟用向量檢索功能
   - 可通過 `use_vector_search` 參數控制
   - 支援動態切換檢索模式
4. **測試**: 運行 `python example_vector_rag.py` 查看向量檢索演示

### 從 v1.0.0 升級到 v2.1.0

1. **安裝新依賴**: 運行 `pip install -r requirements.txt`
2. **環境設定**: 確保 `.env` 檔案包含必要的 API 金鑰
3. **API 使用**: 
   - 舊版 `/analyze` 端點繼續可用
   - 新功能使用 `/enhanced-analyze` 端點
   - 可選擇性啟用 RAG 和向量檢索功能
4. **測試**: 運行 `python test_api.py` 確認功能正常

### 向後兼容性

- ✅ 所有 v1.0.0 和 v2.0.0 的 API 端點保持完全兼容
- ✅ 現有的請求格式無需修改
- ✅ 現有的回應格式保持不變
- ✅ 可以逐步遷移到新功能
- ✅ 向量檢索功能可選擇性啟用 