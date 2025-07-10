# FaceHeartAGI API v3.0

FHIR 醫療資料分析與 RAG 增強 LLM 互動 API，支援異步串流模式。

## 🚀 新版本特色

- **異步串流回應**: 僅支援異步串流，提供即時回應體驗
- **雙重檢索模式**: 支援 LLM 檢索和向量檢索
- **預設知識庫模板**: 內建醫療知識庫，無需額外設定
- **簡化 API 接口**: 統一的請求格式，易於使用
- **FastAPI 測試**: 所有測試均使用 FastAPI 接口

## 📋 功能特點

### API 接口
- **會話管理**: 使用 session_id 記錄對話歷史
- **知識庫內容**: 可選，若無提供則使用預設知識庫模板
- **用戶問題**: 醫療相關問題查詢
- **個人 FHIR**: 患者醫療資料
- **檢索類型**: LLM 檢索 (`llm`) 或向量檢索 (`vector`)

### 核心功能
- FHIR 醫療資料解析與分析
- RAG (Retrieval-Augmented Generation) 增強回應
- 異步串流回應 (Server-Sent Events)
- 醫療知識庫檢索
- 健康建議生成
- **對話歷史記錄**: 自動記錄每輪對話內容
- **智能摘要生成**: 每5輪對話自動生成摘要
- **歷史對話參考**: LLM能夠參考先前的對話內容，提供更連貫的回應

## 🛠️ 安裝與設定

### 1. 克隆專案
```bash
git clone <repository-url>
cd FaceHeartAGI
```

### 2. 安裝依賴
```bash
pip install -r requirements.txt
```

### 3. 環境設定
```bash
cp env.example .env
```

編輯 `.env` 檔案，設定您的 OpenRouter API 金鑰：
```
OPENROUTER_API_KEY=your_api_key_here
```

### 4. 啟動服務
```bash
python main.py
```

服務將在 `http://localhost:8000` 啟動。

## 📡 API 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 健康檢查 |
| `/analyze-stream` | POST | 醫療分析串流端點 |
| `/rag-retrieve` | POST | RAG 檢索端點（非串流） |
| `/conversation-history` | POST | 獲取對話歷史 |
| `/conversation-summaries` | POST | 獲取對話摘要 |
| `/export-conversation` | POST | 匯出完整對話記錄 |
| `/clear-session` | DELETE | 清除會話記錄 |
| `/api-docs` | GET | API 文檔 |
| `/docs` | GET | Swagger UI 文檔（FastAPI 自動生成） |
| `/redoc` | GET | ReDoc 文檔（FastAPI 自動生成） |

### 健康檢查
```
GET /
```

### 醫療分析串流
```
POST /analyze-stream
```

**請求參數:**
```json
{
  "session_id": "string",      // 會話ID，用於記錄對話歷史
  "knowledge_base": {},        // 可選，知識庫內容
  "user_question": "string",   // 用戶問題
  "fhir_data": {},            // FHIR 醫療資料
  "retrieval_type": "vector", // "vector" 或 "llm"
  "additional_context": {}    // 可選，額外上下文
}
```

### RAG 檢索端點
```
POST /rag-retrieve
```

**請求參數:**
```json
{
  "session_id": "string",      // 會話ID，用於記錄對話歷史
  "knowledge_base": {},        // 可選，知識庫內容
  "user_question": "string",   // 用戶問題
  "retrieval_type": "vector"   // "vector" 或 "llm"
}
```

### 對話歷史查詢
```
POST /conversation-history
```

**請求參數:**
```json
{
  "session_id": "string"      // 會話ID
}
```

### 對話摘要查詢
```
POST /conversation-summaries
```

**請求參數:**
```json
{
  "session_id": "string"      // 會話ID
}
```

### 匯出對話記錄
```
POST /export-conversation
```

**請求參數:**
```json
{
  "session_id": "string"      // 會話ID
}
```

### 清除會話記錄
```
DELETE /clear-session
```

**請求參數:**
```json
{
  "session_id": "string"      // 會話ID
}
```

### API 文檔
```
GET /api-docs
```

## 🧪 測試

### 執行測試
```bash
python test_streaming_api.py
```

### 測試內容
- API 健康狀態檢查
- 醫療分析串流（向量檢索）
- 醫療分析串流（LLM 檢索）
- 醫療分析串流（預設知識庫）
- RAG 檢索端點（向量檢索）
- RAG 檢索端點（LLM 檢索）
- RAG 檢索端點（預設知識庫）

## 📊 回應格式

### 串流回應 (Server-Sent Events)
```
event: start
data: {"type": "medical_analysis", "message": "medical_analysis 開始"}

event: chunk
data: {"content": "回應內容片段", "chunk_id": 1}

event: end
data: {"type": "medical_analysis", "message": "medical_analysis 完成", "total_chunks": 10}
```

## 💬 對話記錄功能

### 會話管理
- **會話ID**: 每個對話會話需要提供唯一的 `session_id`
- **自動記錄**: 系統自動記錄每輪對話的用戶問題和系統回應
- **時間戳記**: 每輪對話都包含精確的時間戳記
- **額外資訊**: 記錄檢索類型、知識庫使用情況等元數據

### 智能摘要
- **觸發條件**: 每5輪對話自動生成摘要
- **固定範圍**: 固定對前5輪對話進行總結摘要
- **摘要內容**: 僅包含「用戶意圖」和「系統回應結論」
- **使用 LLM**: 利用 AI 生成簡潔、準確的對話摘要
- **累積記錄**: 多個摘要累積保存，形成完整的對話脈絡
- **記憶體管理**: 最多保存10輪對話，自動移除最舊的記錄

### 歷史對話參考
- **上下文感知**: LLM能夠參考之前的對話內容
- **智能歷史管理**: 當對話超過5輪時，自動使用最近幾輪加上摘要
- **連貫性回應**: 提供與之前對話相關聯的回應
- **個人化建議**: 基於用戶的歷史問題提供更精準的建議
- **智能引用**: 能夠適當引用之前的建議和資訊

### 資料匯出
- **JSON 格式**: 支援完整對話記錄的 JSON 格式匯出
- **包含摘要**: 匯出內容包含對話歷史和所有摘要
- **時間戳記**: 匯出時間和各輪對話時間完整記錄

## 🔧 檢索類型

### 向量檢索 (Vector Search)
- 使用 FAISS 向量資料庫
- 語義相似度匹配
- 支援中文醫療文本
- 預設檢索模式
- **動態創建**：每次請求時根據用戶提供的知識庫動態創建向量資料庫
- **即時檢索**：無需預先載入，響應更快

### LLM 檢索 (Traditional Search)
- 基於關鍵字匹配
- 使用 LLM 進行內容檢索
- 適合結構化資料查詢

## 📁 專案結構

```
FaceHeartAGI/
├── main.py                      # 主要 API 服務
├── llm_client.py               # LLM 客戶端（異步串流）
├── rag_client.py               # RAG 客戶端（支援雙重檢索）
├── conversation_manager.py     # 對話管理與摘要生成
├── fhir_parser.py              # FHIR 資料解析器
├── prompt_builder.py           # 提示詞建構器
├── vector_store.py             # 向量資料庫管理
├── test_streaming_api.py       # 串流 API 測試
├── example_usage_v3.py         # 使用範例
├── example_usage_conversation.py # 對話記錄功能範例
├── example_conversation_context.py # 歷史對話上下文範例
├── example_fhir_history.py # FHIR資料歷史對話範例
├── example_new_summary_logic.py # 新摘要邏輯範例
├── requirements.txt            # 依賴套件
├── env.example                # 環境變數範例
├── README.md                  # 專案說明
└── vector_db/                 # 向量資料庫目錄（用於保存，可選）
```

## 🔄 版本變更

### v3.0.0
- ✅ 僅保留異步串流功能
- ✅ 刪除非串流和同步串流
- ✅ 統一使用 FastAPI 測試接口
- ✅ 保留 LLM 與向量檢索
- ✅ 新增預設知識庫模板
- ✅ 簡化 API 接口設計
- ✅ 支援檢索類型切換
- ✅ **新增對話歷史記錄功能**
- ✅ **新增智能摘要生成 (每5輪對話)**
- ✅ **新增會話管理系統**
- ✅ **新增對話匯出功能**
- ✅ **新增LLM歷史對話參考功能**

### 移除的功能
- ❌ 非串流回應端點
- ❌ 同步串流功能
- ❌ 舊版測試文件
- ❌ 性能比較工具
- ❌ 範例使用腳本
- ❌ 啟動腳本（直接使用 main.py）

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案採用 MIT 授權條款。

## 📞 支援

如有問題，請提交 Issue 或聯繫開發團隊。

## 📄 JSON 檔案結構

### 知識庫檔案

- **`default_knowledge_base.json`**: 預設醫療知識庫，包含高血壓和糖尿病的基本指南
- **`custom_knowledge_base.json`**: 自定義知識庫，包含更詳細的醫療指南和生活方式建議

### 示例資料檔案

- **`sample_fhir_data.json`**: 示例 FHIR 醫療資料，包含患者資訊、觀察結果、診斷和藥物請求

## 📄 更新日誌

詳細的更新日誌請參考 [CHANGELOG.md](CHANGELOG.md)。 