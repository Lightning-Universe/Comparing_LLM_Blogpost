import lightning as L
from langchain.llms.base import LLM
from .generate_adapter import alpaca_adapter
import pytorch_lightning as pl
from typing import List, Optional
from pydantic import BaseModel, Extra  


L.seed_everything(42)

class AlpacaPipeline(LLM, BaseModel):
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self):
        super().__init__()
        global model_LLM
        model_LLM = alpaca_adapter()
        
    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        max_new_tokens = 70
        top_k = 100
        temperature  =  0.2
        text = model_LLM.generate(
            prompt=prompt, 
            max_new_tokens= max_new_tokens,
            top_k = 100,
            temperature= temperature)
        return text
