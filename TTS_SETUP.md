# 語音功能設置說明

## 安裝 Piper TTS

### 方法 1: 使用 pip 安裝
```bash
pip install piper-tts
```

### 方法 2: 從源代碼安裝
```bash
# 安裝依賴
pip install torch torchaudio

# 克隆並安裝 piper
git clone https://github.com/rhasspy/piper.git
cd piper
pip install -e .
```

## 下載語音模型

Piper TTS 需要下載語音模型才能工作。建議使用以下命令下載英文模型：

```bash
# 下載英文語音模型
python3 -m piper.download_voices --download-dir ./voices en_US-lessac-medium
```

## 測試語音功能

1. 啟動 API 服務器：
```bash
python main.py
```

2. 運行測試腳本：
```bash
cd tests
python test_multi_turn_memory.py
```

## 功能說明

- **語音生成**: 在 `/analyze-stream` 端點中設置 `generate_audio: true` 即可生成語音
- **語音播放**: 使用 `/audio/{device_id}/{audio_id}` 端點獲取語音文件
- **語音緩存**: 語音文件會保存在 `./audio_cache/` 目錄中

## 注意事項

- 首次運行時，Piper TTS 會自動下載語音模型
- 語音文件會根據 device_id 和 audio_id 進行命名
- 建議定期清理 audio_cache 目錄中的舊文件
