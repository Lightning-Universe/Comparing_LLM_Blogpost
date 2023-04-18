import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from .generate import generate
from lit_llama import Tokenizer
from lit_llama.adapter import LLaMA, LLaMAConfig
from lit_llama.utils import EmptyInitOnDevice
from .scripts.prepare_alpaca import generate_prompt


class alpaca_adapter():

    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned LLaMA-Adapter model.
    See `finetune_adapter.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune_adapter.py`.
        input: Optional input (Alpaca style).
        pretrained_path: The path to the checkpoint with pretrained LLaMA weights.
        tokenizer_path: The tokenizer path to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
        dtype: The dtype to use during generation.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
    """
    def __init__(self,
    adapter_path: Optional[Path] = None,
    pretrained_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    quantize: Optional[str] = None):
        if not adapter_path:
            adapter_path = Path("/home/zeus/content/Comparing_LLM_Blogpost/checkpoints/adapter/alpaca/alpaca-adapter-finetuned.pt")
        if not pretrained_path:
            pretrained_path = Path("/home/zeus/content/Comparing_LLM_Blogpost/checkpoints/lit-llama/7B/state_dict.pth")
        if not tokenizer_path:
            tokenizer_path = Path("/home/zeus/content/Comparing_LLM_Blogpost/checkpoints/lit-llama/tokenizer.model")
        
        assert adapter_path.is_file()
        assert pretrained_path.is_file()
        assert tokenizer_path.is_file()

        fabric = L.Fabric(accelerator="cuda", devices=1)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

        with EmptyInitOnDevice(
            device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            print("Loading model ...", file=sys.stderr)
            t0 = time.time()
            self._model = LLaMA(LLaMAConfig())  # TODO: Support different model sizes

            # 1. Load the pretrained weights
            pretrained_checkpoint = torch.load(pretrained_path)
            self._model.load_state_dict(pretrained_checkpoint, strict=False)
            # 2. Load the fine-tuned adapter weights
            adapter_checkpoint = torch.load(adapter_path, map_location=torch.device("cpu"))

            self._model.load_state_dict(adapter_checkpoint, strict=False)
            print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        self._model.eval()
        self._model = fabric.setup_module(self._model)
        self._tokenizer = Tokenizer(tokenizer_path)
    
    def generate(self,
            prompt:str="Generate a funny joke", 
            input_llama:str ="",
            max_new_tokens: int = 100,
            top_k: int = 200,
            temperature: float = 0.7):
        sample = {"instruction": prompt, "input": input_llama}
        prompt = generate_prompt(sample)
        encoded = self._tokenizer.encode(prompt, bos=True, eos=False)
        encoded = encoded[None, :]  # add batch dimension
        encoded = encoded.to(self._model.device)
        output = generate(
            self._model,
            idx=encoded,
            max_seq_length=max_new_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # The end of the response is where the model generates the EOS token
        output = truncate_output_to_eos(output[0].cpu(), self._tokenizer.eos_id)
        output = self._tokenizer.decode(output)
        output = output.split("### Response:")[1].strip()
        torch.cuda.empty_cache()
        return output
        
def truncate_output_to_eos(output, eos_id):
    # TODO: Make this more efficient, terminate generation early
    try:
        eos_pos = output.tolist().index(eos_id)
    except ValueError:
        eos_pos = -1

    output = output[:eos_pos]
    return output
