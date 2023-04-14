import torch
from langchain.llms.base import LLM
from transformers import (T5Tokenizer, 
                        T5ForConditionalGeneration,
                        AutoTokenizer, 
                        pipeline, 
                        AutoModelForCausalLM,
                        GPTNeoXForCausalLM, 
                        GPTNeoXTokenizerFast)
from typing import List, Optional
from pydantic import BaseModel, Extra  

## "balanced_low_0" 
class CustomPipeline(LLM, BaseModel):
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self, model_id):
        super().__init__()
        global model, tokenizer, model_name
        model_name = model_id 
        device_map = "auto"
        if model_id == "google/ul2" or model_id == "google/flan-t5-xxl":
            model = T5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map)      
            if  model_id == "google/ul2":
                tokenizer = AutoTokenizer.from_pretrained("google/ul2")
            else:
                tokenizer = T5Tokenizer.from_pretrained(model_id)
        elif model_id == "facebook/opt-iml-max-30b":
            model =  pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype":torch.bfloat16}, device_map=device_map)
        elif model_id == "bigscience/bloomz-7b1":
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device_map, offload_folder="offload")  
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif model_id=="EleutherAI/gpt-j-6B":
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map) 
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif model_id == "EleutherAI/gpt-neox-20b":
            model = GPTNeoXForCausalLM.from_pretrained(model_id, device_map=device_map) 
            tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_id)
        elif model_id == "Salesforce/codegen-16B-mono":
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map) 
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        elif model_id == "EleutherAI/pythia-12b-deduped":
            model = GPTNeoXForCausalLM.from_pretrained(
                        model_id,
                        revision="step3000",
                        device_map=device_map) 
            tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        revision="step3000",
                        )
    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        if model_name == "facebook/opt-iml-max-30b":
            prompt_length = len(prompt)
            response = model(prompt,max_new_tokens = 70)[0]["generated_text"]
            return response
        else:
            with torch.no_grad():
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
                outputs = model.generate(input_ids, max_new_tokens = 70)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                return response
