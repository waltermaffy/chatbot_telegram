"""
Create different index with Llama-index and save it on Vector Database
Then allow querying on the index
"""

import os
from llama_index import SimpleDirectoryReader
from llama_index import LLMPredictor, VectorStoreIndex, ServiceContext
from langchain import OpenAI
from llama_index import StorageContext, load_index_from_storage


class Indexer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.input_dir = cfg["input_dir"]
        # TODO: change output from disk to vector database
        self.output_dir = cfg["output_dir"]
        llm = OpenAI(temperature=0, model_name="text-davinci-003")
        # define LLM
        self.llm_predictor = LLMPredictor(llm=llm)
    
        # configure service context
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)
        
        self.index = self.load_index()
        self.save_index()
        
    def load_index(self):
        # when loading the index from disk
        docstrore_file = os.path.join(self.output_dir, "docstore.json")
        if os.path.exists(docstrore_file):
            print(f"Loading index from {self.output_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=self.output_dir)
            index = load_index_from_storage(
                storage_context=storage_context,
                service_context=self.service_context,    
            )
        else:
            # Load documents 
            documents = SimpleDirectoryReader(self.input_dir).load_data()
            print(f"Loaded {len(documents)} documents from {self.input_dir}")
            # build index
            index = VectorStoreIndex.from_documents(
                documents, service_context=self.service_context
            )
        return index
    
    def save_index(self):
        self.index.storage_context.persist(persist_dir=self.output_dir)

    def query(self, text_query: str):
        if not text_query:
            raise ValueError("Query cannot be empty")
        if not self.index:
            raise ValueError("Index not loaded")
        
        # query index
        query_engine = self.index.as_query_engine()
        response = query_engine.query(text_query)  
        return response
    

if __name__ == "__main__":
    
    cfg = {
        "input_dir": "data",
        "output_dir": "data",
    }
    indexer = Indexer(cfg)
    
    result = indexer.query("How to open a Lightning channel?")
    
    """
    To open a Lightning channel, Alice can press the plus symbol in the LIGHTNING CHANNELS tab and select one of the four possible ways to open a channel. Once she clicks OPEN, her wallet will construct the special Bitcoin transaction that opens a Lightning channel, known as the funding transaction. The on-chain funding transaction is then sent to the Bitcoin network for confirmation.
    """
    print(result)

