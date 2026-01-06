import numpy as np
import faiss
from typing import Dict, Any, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import logging
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from src.utils.data_parser import extract_medical_documents

# 載入環境變數
load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    """基於 FAISS 的向量資料庫，用於高效的相似性檢索"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 cache_dir: str = "./vector_cache"):
        """
        初始化向量資料庫
        
        Args:
            model_name: 句子嵌入模型名稱
            cache_dir: 向量數據庫緩存目錄
        """
        self.model_name = model_name
        self.model_path = os.getenv("EMBEDDING_MODEL_PATH")
        self.encoder = SentenceTransformer(self.model_path)
        self.index = None
        self.documents = []
        self.metadata = []
        self.dimension = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
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
    
    def save_to_disk(self, cache_name: str = "default") -> bool:
        """
        保存向量數據庫到磁盤
        
        Args:
            cache_name: 緩存名稱
            
        Returns:
            是否保存成功
        """
        try:
            if self.index is None or len(self.documents) == 0:
                logger.warning("向量資料庫為空，無法保存")
                return False
            
            # 創建緩存子目錄
            cache_path = self.cache_dir / cache_name
            cache_path.mkdir(exist_ok=True)
            
            # 保存 FAISS 索引
            index_file = cache_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            logger.info(f"FAISS 索引已保存到 {index_file}")
            
            # 保存文檔和元數據
            data_file = cache_path / "documents.pkl"
            with open(data_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'model_name': self.model_name,
                    'timestamp': datetime.now().isoformat()
                }, f)
            logger.info(f"文檔數據已保存到 {data_file}")
            
            # 保存元信息
            meta_file = cache_path / "meta.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_name': self.model_name,
                    'dimension': self.dimension,
                    'document_count': len(self.documents),
                    'timestamp': datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"元信息已保存到 {meta_file}")
            
            logger.info(f"向量數據庫已成功保存到 {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存向量數據庫時發生錯誤: {str(e)}")
            return False
    
    def load_from_disk(self, cache_name: str = "default") -> bool:
        """
        從磁盤加載向量數據庫
        
        Args:
            cache_name: 緩存名稱
            
        Returns:
            是否加載成功
        """
        try:
            cache_path = self.cache_dir / cache_name
            
            if not cache_path.exists():
                logger.info(f"緩存目錄不存在: {cache_path}")
                return False
            
            # 檢查必要文件是否存在
            index_file = cache_path / "faiss.index"
            data_file = cache_path / "documents.pkl"
            
            if not index_file.exists() or not data_file.exists():
                logger.warning(f"緩存文件不完整，無法加載")
                return False
            
            # 加載 FAISS 索引
            self.index = faiss.read_index(str(index_file))
            logger.info(f"FAISS 索引已從 {index_file} 加載")
            
            # 加載文檔和元數據
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                timestamp = data.get('timestamp', 'unknown')
            
            logger.info(f"文檔數據已從 {data_file} 加載")
            logger.info(f"成功加載向量數據庫: {len(self.documents)} 個文檔, 時間戳: {timestamp}")
            
            return True
            
        except Exception as e:
            logger.error(f"加載向量數據庫時發生錯誤: {str(e)}")
            return False


class MedicalVectorStore(VectorStore):
    """專門用於醫療資料的向量資料庫"""
    
    def __init__(self, auto_load: bool = True):
        """
        初始化醫療向量資料庫，使用適合醫療文本的嵌入模型
        
        Args:
            auto_load: 是否自動嘗試從磁盤加載緩存
        """
        super().__init__("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self._knowledge_loaded = False  # 標誌：知識庫是否已加載
        
        # 嘗試從磁盤加載
        if auto_load:
            if self.load_from_disk("medical_knowledge"):
                self._knowledge_loaded = True
                logger.info("成功從磁盤加載醫療知識庫")
            else:
                logger.info("未找到緩存，將需要從源文件構建知識庫")
    
    def load_knowledge_base(self) -> None:
        """
        從 knowledge 目錄加載醫療知識庫
        只在服務器端統一管理，不支持用戶自定義知識庫
        """
        # 如果已經加載，跳過
        if self._knowledge_loaded:
            logger.info("醫療知識庫已加載，跳過重複加載")
            return
        
        all_documents = []
        
        # 讀取 /knowledge 目錄下的所有 JSON 文件
        knowledge_dir = Path("knowledge")
        if not knowledge_dir.exists():
            logger.warning("knowledge 目錄不存在")
            return
            
        json_files = list(knowledge_dir.glob("*.json"))
        if not json_files:
            logger.warning("knowledge 目錄下沒有找到 JSON 文件")
            return
            
        logger.info(f"找到 {len(json_files)} 個 JSON 文件，開始處理醫療知識庫...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                documents = extract_medical_documents(data)
                all_documents.extend(documents)
                logger.info(f"成功處理 {json_file.name}，提取了 {len(documents)} 個文檔")
                
            except Exception as e:
                logger.error(f"處理文件 {json_file.name} 時發生錯誤: {str(e)}")
                continue
        
        if all_documents:
            logger.info(f"總共提取了 {len(all_documents)} 個文檔，開始添加到向量資料庫")
            self.add_documents(all_documents)
            self._knowledge_loaded = True
            
            # 保存到磁盤以便下次快速加載
            self.save_to_disk("medical_knowledge")
            logger.info("醫療知識庫加載完成並已保存到磁盤")
        else:
            logger.warning("沒有提取到任何文檔")
    
    def cleanup(self):
        """清理向量存儲資源"""
        try:
            # 清空索引和文檔
            self.index = None
            self.documents = []
            self.metadata = []
            self._knowledge_loaded = False  # 重置加載標誌
            logger.info("向量存儲資源已清理")
        except Exception as e:
            logger.error(f"清理向量存儲時發生錯誤: {e}")
    
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