import os
import json
import pickle
import numpy as np
import faiss
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv
from data_parser import extract_medical_documents

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    """基於 FAISS 的向量資料庫，用於高效的相似性檢索"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化向量資料庫
        
        Args:
            model_name: 句子嵌入模型名稱
        """
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = None
        
        logger.info(f"向量資料庫初始化完成，使用模型: {model_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加文件到向量資料庫
        
        Args:
            documents: 文件列表，每個文件應包含 'content' 和 'metadata' 字段
        """
        try:
            if not documents:
                logger.warning("沒有文件需要添加")
                return
            
            # 提取文件內容和元數據
            contents = []
            for doc in documents:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                
                if content.strip():
                    contents.append(content)
                    self.documents.append(content)
                    self.metadata.append(metadata)
            
            if not contents:
                logger.warning("沒有有效的文件內容")
                return
            
            # 生成嵌入向量
            logger.info(f"正在為 {len(contents)} 個文件生成嵌入向量...")
            embeddings = self.encoder.encode(contents, show_progress_bar=True)
            
            # 初始化 FAISS 索引
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)  # 使用內積相似度
                logger.info(f"創建 FAISS 索引，維度: {self.dimension}")
            
            # 添加向量到索引
            self.index.add(embeddings.astype('float32'))
            logger.info(f"成功添加 {len(embeddings)} 個向量到索引")
            
        except Exception as e:
            logger.error(f"添加文件到向量資料庫時發生錯誤: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        搜尋相似文件
        
        Args:
            query: 查詢文本
            top_k: 返回的最相似文件數量
            threshold: 相似度閾值
            
        Returns:
            相似文件列表，包含內容、元數據和相似度分數
        """
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("向量資料庫為空，無法進行搜尋")
                return []
            
            # 生成查詢向量
            query_embedding = self.encoder.encode([query])
            
            # 搜尋相似向量
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # 過濾結果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents) and score >= threshold:
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'score': float(score),
                        'rank': i + 1
                    })
            
            logger.info(f"搜尋完成，找到 {len(results)} 個相關文件")
            return results
            
        except Exception as e:
            logger.error(f"搜尋過程中發生錯誤: {str(e)}")
            return []


class MedicalVectorStore(VectorStore):
    """專門用於醫療資料的向量資料庫"""
    
    def __init__(self):
        """初始化醫療向量資料庫，使用適合醫療文本的嵌入模型"""
        super().__init__("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    def add_medical_documents(self, medical_data: Dict[str, Any]) -> None:
        """
        添加醫療資料到向量資料庫
        
        Args:
            medical_data: 醫療資料字典
        """
        documents = extract_medical_documents(medical_data)
        self.add_documents(documents)
    
    def search_medical_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        搜尋醫療相關內容
        
        Args:
            query: 醫療查詢
            top_k: 返回結果數量
            
        Returns:
            相關醫療內容列表
        """
        # 為醫療查詢添加一些上下文
        enhanced_query = f"醫療相關: {query}"
        return self.search(enhanced_query, top_k, threshold=0.3) 