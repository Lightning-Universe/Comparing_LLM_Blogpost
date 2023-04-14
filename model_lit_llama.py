import lightning as L

import sys
import torch

import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Extra  

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice
from langchain.llms.base import LLM

pl.seed_everything(42)

# code taken from  https://github.com/Lightning-AI/lit-llama
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new column
        idx[:, t:] = idx_next

    return idx

# code taken from  https://github.com/Lightning-AI/lit-llama
def load_model(
    checkpoints_path: str,
    model_size: str = "7B",    
    dtype = torch.bfloat16,
    quantization: bool =  True):

    fabric = L.Fabric(accelerator="auto", devices=1)
    if quantization:
        quantization_mode = 'llm.int8'
    else:
        quantization_mode =  None
    checkpoint_path = Path(f"{checkpoints_path}/{model_size}/state_dict.pth")
    tokenizer_path = Path(f"{checkpoints_path}/tokenizer.model")

    with EmptyInitOnDevice(device=fabric.device, quantization_mode=quantization_mode):
        print("Loading model ...", file=sys.stderr)
        model = LLaMA.from_name(model_size)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model.eval()
        model = fabric.setup_module(model)
        tokenizer = Tokenizer(tokenizer_path)

        return model, tokenizer, fabric


class LitLlamaPipeline(LLM, BaseModel):
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def __init__(self,checkpoints_path):
        super().__init__()
        global model_LLM, tokenizer_LLM, fabric_LLM
        model_LLM, tokenizer_LLM, fabric_LLM = load_model(dtype=None,checkpoints_path=checkpoints_path)
        
    @property
    def _llm_type(self) -> str:
        return "custom_pipeline"
    
    def get_encoder_params(self):
        return model_LLM.transformer.wte, tokenizer_LLM, fabric_LLM.device

    def _call(self, prompt: str, stop: Optional[List[str]] = None):
        max_new_tokens = 70
        top_k = 100
        temperature  =  0.2

        encoded_prompt = tokenizer_LLM.encode(prompt, bos=True, eos=False, device=fabric_LLM.device)
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension
        y = generate(
                    model = model_LLM,
                    idx = encoded_prompt,
                    max_new_tokens = max_new_tokens,
                    max_seq_length = model_LLM.config.block_size,  
                    temperature=temperature,
                    top_k=top_k,
            )[0]  # unpack batch dimension
        text = tokenizer_LLM.decode(y)[len(prompt)::]
        del y 
        torch.cuda.empty_cache() 
        return text
