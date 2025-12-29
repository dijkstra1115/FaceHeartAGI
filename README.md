# FaceHeartAGI API v3.0

## 📝 ToDo List

- [ ] need gunicorn to support multiple workers
- [ ] survey text-to-speech library
- [ ] context engineering
  1. Hybrid Retrieval: Parallel Excution and Re-ranking
  2. Contextual Chunking and Embedding: Pre-process the knowledge base
  3. Dynamic Prompt Construction
  4. Conversation History Management
  5. Leveraging FHIR Structure
- [ ] need Nginx and Domain name to support HTTPS
- [ ] need a License in DB to protect the API and verify the permissions
- [ ] Fix the bug in the database where `turn_number` cannot be modified (SQLAlchemy flush sends multiple UPDATEs in its own order)
- [ ] transaction does not solve the race condition problem
- [ ] 2-layer filter for hallucination avoidance
- [ ] Prompt enhancement for flexible responses
- [ ] shared VDB or personalized VDB
- [ ] load different knowledge based on user question
- [ ] GPU memory release

FHIR 醫療資料分析與 RAG 增強 LLM 互動 API，支援異步串流模式。

## 🚀 新版本特色

- **僅支援異步串流回應**：提供即時回應體驗
- **雙重檢索模式**：支援 LLM 檢索和向量檢索
- **預設知識庫模板**：內建醫療知識庫，無需額外設定
- **簡化 API 接口**：統一的請求格式，易於使用
- **FastAPI 測試**：所有測試均使用 FastAPI 接口
- **彈性對話管理**：可透過環境變數啟用/停用歷史對話功能
- **三種回覆模式**：
  1. 有檢索資料與歷史對話
  2. 只有歷史對話
  3. 只有檢索資料（無歷史對話）

## 📋 功能特點

- FHIR 醫療資料解析與分析
- RAG (Retrieval-Augmented Generation) 增強回應
- 異步串流回應 (Server-Sent Events)
- 醫療知識庫檢索
- 健康建議生成
- 對話歷史記錄與摘要
- LLM 能參考歷史對話內容，提供連貫回應
- 可選擇性啟用歷史對話功能
- 必須檢索模式：若檢索不到資料可選擇回報錯誤

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

編輯 `.env` 檔案，設定您的 OpenRouter API 金鑰及其他設定：
```
OPENROUTER_API_KEY=your_api_key_here

# 對話管理設定
ENABLE_CONVERSATION_HISTORY=true  # 是否啟用歷史對話功能 (true/false)
REQUIRE_RETRIEVAL=false  # 是否必須檢索到資料才回應 (true/false)
```

#### 環境變數說明

**對話管理設定：**
- `ENABLE_CONVERSATION_HISTORY`: 
  - `true` (預設): 啟用歷史對話功能，LLM 會參考之前的對話內容
  - `false`: 停用歷史對話功能，每次請求獨立處理
  
- `REQUIRE_RETRIEVAL`:
  - `true`: 必須從知識庫檢索到相關資料才會回應，若檢索不到會回報錯誤
  - `false` (預設): 若檢索不到資料，LLM 仍會基於 FHIR 數據和歷史對話回應

**三種回覆模式：**
1. **有檢索資料與歷史對話** (`ENABLE_CONVERSATION_HISTORY=true`, 且成功檢索到資料)
2. **只有歷史對話** (`ENABLE_CONVERSATION_HISTORY=true`, 但未檢索到資料且 `REQUIRE_RETRIEVAL=false`)
3. **只有檢索資料** (`ENABLE_CONVERSATION_HISTORY=false`, 且成功檢索到資料)

### 4. 啟動服務
```bash
python main.py
```

## 📦 佈署時可能遇到的問題

### 1. 佈署環境
- ubuntu 24.04 LTS
- RTX 5090
- NVIDIA Driver 570.86.10
- CUDA 12.8 Toolkit

### 2. 遇過的問題
- 安裝 Ubuntu 時要先插內顯，安裝完成再插獨顯，並安裝 GPU 驅動 (`570.86.10`)
- Torch 要去官網下載最新版 `cu128` 才能與 RTX 5090 相容（`sm_120` 架構）
- 安裝/編譯 vLLM 時 OOM → 增加虛擬記憶體
- 執行 vLLM server 時遇到 `GLIBCXX_3.4.32 not found`：
  1. 安裝新版 g++ 和 libstdc++：
     ```bash
     sudo apt install g++-13 libstdc++6
     ```
  2. 指定載入新版的 `libstdc++`：
     ```bash
     export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33
     ```
- 使用 SentenceTransformer 時 CUDA error -> 下載支援 `cu128` 的最新 Torch

### 3. 佈署與運行
1. 佈署 vLLM (參考網址 https://zhuanlan.zhihu.com/p/1902008703462406116)
```bash
 #pip安装pyTorch
 pip3 install --force-reinstall torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
 #编译安装flashinfer
 git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
 cd flashinfer
 python -m pip install -v .
 #编译安装vllm：
 cd ..
 git clone https://github.com/vllm-project/vllm.git
 cd vllm
 python use_existing_torch.py 
 pip install -r requirements/build.txt 
 pip install -r requirements/common.txt
 pip install -e . --no-build-isolation (編譯會耗費大量資源，若為 OOM Error 則要增加虛擬記憶體大小，降低 workers 數量)
```
2. 運行 vLLM Server (需預先下載 `deepseek-qwen7b` 到本地)
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33
```
```bash
cd vllm
```
```bash
python -m vllm.entrypoints.openai.api_server   --model ~/llm-models/deepseek-qwen7b   --tokenizer ~/llm-models/deepseek-qwen7b   --served-model-name deepseek-qwen7b   --host 0.0.0.0   --port 8000
```
3. 運行 API (需要預先下載 `paraphrase-multilingual-MiniLM-L12-v2` 到本地)
```bash
python main.py
```

## 📡 API 端點

| 端點 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 健康檢查 |
| `/analyze-stream` | POST | 醫療分析串流端點 |
| `/clear-session` | DELETE | 清除會話記錄 |
| `/help` | GET | API 協助 |
| `/docs` | GET | Swagger UI 文檔（FastAPI 自動生成） |
| `/redoc` | GET | ReDoc 文檔（FastAPI 自動生成） |

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
  "retrieval_type": "vector" // "vector" 或 "llm"
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

## 🧪 測試

### 執行測試
```bash
python test_multi_turn_memory.py
```
（legacy/ 下的 example_usage_v3.py、test_api.py 僅供參考）

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

- 會話ID：每個對話會話需要提供唯一的 `session_id`
- 自動記錄：系統自動記錄每輪對話的用戶問題和系統回應（僅在啟用歷史對話時）
- 時間戳記：每輪對話都包含精確的時間戳記
- 智能摘要：每5輪對話自動生成摘要，僅保留最近10輪
- 歷史對話參考：LLM能夠參考之前的對話內容，提供連貫性回應
- 資料匯出：支援完整對話記錄的 JSON 格式匯出（如需可自行擴充）
- 彈性控制：可透過環境變數 `ENABLE_CONVERSATION_HISTORY` 啟用/停用此功能

## 🔧 檢索類型

- 向量檢索 (Vector Search)：使用 FAISS 向量資料庫，語義相似度匹配，支援中文醫療文本
- LLM 檢索 (Traditional Search)：基於關鍵字匹配，適合結構化資料查詢

## 📁 專案結構

```
FaceHeartAGI/
├── main.py                      # 主要 API 服務
├── llm_client.py                # LLM 客戶端（異步串流）
├── rag_client.py                # RAG 客戶端（支援雙重檢索）
├── conversation_manager.py       # 對話管理與摘要生成
├── prompt_builder.py             # 提示詞建構器
├── vector_store.py               # 向量資料庫管理
├── test_multi_turn_memory.py     # 多輪記憶測試腳本
├── requirements.txt              # 依賴套件
├── env.example                   # 環境變數範例
├── README.md                     # 專案說明
├── fhir/                         # FHIR 測試資料
├── knowledge/                    # 醫療知識庫資料
└── legacy/                       # 舊版範例與測試腳本
```

## 📄 JSON 檔案結構

### 知識庫檔案

- **`default_knowledge_base.json`**: 預設醫療知識庫，包含高血壓和糖尿病的基本指南

### 示例資料檔案

- **`fhir/fhir_1.json` ~ `fhir/fhir_6.json`**: 多組 FHIR 醫療資料範例

## 🔄 版本變更

### v3.0.0
- ✅ 僅保留異步串流功能
- ✅ 刪除非串流和同步串流
- ✅ 統一使用 FastAPI 測試接口
- ✅ 保留 LLM 與向量檢索
- ✅ 新增預設知識庫模板
- ✅ 簡化 API 接口設計
- ✅ 支援檢索類型切換
- ✅ 新增對話歷史記錄功能
- ✅ 新增智能摘要生成 (每5輪對話)
- ✅ 新增會話管理系統
- ✅ 新增對話匯出功能
- ✅ 新增LLM歷史對話參考功能
- ✅ 新增環境變數控制歷史對話功能
- ✅ 新增三種回覆模式支援
- ✅ 新增必須檢索模式 (可選)
- ✅ FHIR 數值格式化為兩位小數

### 移除的功能
- ❌ 非串流回應端點
- ❌ 同步串流功能
- ❌ 舊版測試文件
- ❌ 性能比較工具
- ❌ 範例使用腳本（僅 legacy/ 下保留參考）
- ❌ 啟動腳本（直接使用 main.py）

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本專案採用 MIT 授權條款。

## 📞 支援

如有問題，請提交 Issue 或聯繫開發團隊。