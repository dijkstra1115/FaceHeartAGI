import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 載入模型
encoder = SentenceTransformer("/home/llm/embedding-models/paraphrase-multilingual-MiniLM-L12-v2")

# 模擬語料庫
documents = [
    "今天是星期一",
    "我正在看醫療報告",
    "病人血糖偏高，需要注意",
    "這是一場重要的醫學研討會",
    "天氣晴朗，適合運動"
]

# 建立文件向量
doc_embeddings = encoder.encode(documents)
dimension = doc_embeddings.shape[1]

# 建立 FAISS index（使用內積為相似度）
index = faiss.IndexFlatIP(dimension)

# 正規化（讓內積 = cosine）
doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
index.add(doc_embeddings.astype('float32'))

# 搜尋範例
query = "血糖異常的處理方式"
query_embedding = encoder.encode([query])
query_embedding = query_embedding / np.linalg.norm(query_embedding)

# 搜尋前3筆
top_k = 3
scores, indices = index.search(query_embedding.astype('float32'), top_k)

# 顯示結果
print(f"查詢：{query}\n")
for i, idx in enumerate(indices[0]):
    print(f"Top {i+1}：{documents[idx]}")
    print(f"相似度分數：{scores[0][i]:.4f}")
    print()
