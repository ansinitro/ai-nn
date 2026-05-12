"""
LLM inference module with two backends:
  1. GGUF (local) via llama-cpp-python — primary
  2. OpenAI API — fallback

Automatically selects the best available backend.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GGUF_MODEL_PATH,
    LLM_CONTEXT_LENGTH,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_N_GPU_LAYERS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


class LLM:
    """Unified LLM interface supporting GGUF and OpenAI backends."""

    def __init__(self, backend: str = "auto"):
        """
        Initialize the LLM.

        Args:
            backend: "gguf", "api", or "auto" (try gguf first, then api)
        """
        self.backend = None
        self._llm = None
        self._client = None

        if backend == "auto":
            if self._try_load_gguf():
                pass
            elif self._try_load_api():
                pass
            else:
                raise RuntimeError(
                    "No LLM backend available!\n"
                    "Option 1: Download GGUF model to models/ directory\n"
                    "Option 2: Set OPENAI_API_KEY environment variable"
                )
        elif backend == "gguf":
            if not self._try_load_gguf():
                raise RuntimeError(f"GGUF model not found at {GGUF_MODEL_PATH}")
        elif backend == "api":
            if not self._try_load_api():
                raise RuntimeError("OPENAI_API_KEY not set")

    def _try_load_gguf(self) -> bool:
        """Attempt to load the local GGUF model with GPU fallback."""
        if not GGUF_MODEL_PATH.exists():
            print(f"GGUF model not found at {GGUF_MODEL_PATH}")
            return False

        try:
            from llama_cpp import Llama
        except ImportError:
            print("llama-cpp-python not installed, skipping GGUF backend")
            return False

        # Try progressively fewer GPU layers if VRAM is tight
        gpu_attempts = [LLM_N_GPU_LAYERS, 20, 15, 10, 0]
        # Remove duplicates while preserving order
        seen = set()
        gpu_attempts = [x for x in gpu_attempts if not (x in seen or seen.add(x))]

        for n_layers in gpu_attempts:
            try:
                mode = "GPU" if n_layers > 0 else "CPU-only"
                print(f"Loading GGUF model: {GGUF_MODEL_PATH.name} ({mode}, layers={n_layers})...")
                self._llm = Llama(
                    model_path=str(GGUF_MODEL_PATH),
                    n_ctx=LLM_CONTEXT_LENGTH,
                    n_gpu_layers=n_layers,
                    verbose=False,
                )
                self.backend = "gguf"
                self._gpu_layers = n_layers
                print(f"LLM ready: GGUF (local, {mode}, layers={n_layers})")
                return True
            except Exception as e:
                print(f"  Failed with {n_layers} GPU layers: {e}")
                continue

        print("All GGUF loading attempts failed")
        return False

    def _try_load_api(self) -> bool:
        """Attempt to set up OpenAI API backend."""
        api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("No OpenAI API key found")
            return False

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
            self.backend = "api"
            print(f"LLM ready: OpenAI API ({OPENAI_MODEL})")
            return True
        except ImportError:
            print("openai package not installed")
            return False

    def generate(self, prompt: str) -> str:
        """
        Generate a response given a formatted prompt.

        Args:
            prompt: Full prompt string (with system instructions and context)

        Returns:
            Generated answer text
        """
        if self.backend == "gguf":
            return self._generate_gguf(prompt)
        elif self.backend == "api":
            return self._generate_api(prompt)
        else:
            raise RuntimeError("No LLM backend loaded")

    def _generate_gguf(self, prompt: str) -> str:
        """Generate using local GGUF model."""
        output = self._llm(
            prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            stop=["</s>", "[INST]"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def _generate_api(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        # Parse the Llama 2 prompt format into messages
        system_msg = "You are a helpful assistant that answers questions based only on the provided context."
        user_msg = prompt

        # Try to extract system message from Llama 2 format
        if "<<SYS>>" in prompt and "<</SYS>>" in prompt:
            import re
            sys_match = re.search(r'<<SYS>>\s*(.*?)\s*<</SYS>>', prompt, re.DOTALL)
            if sys_match:
                system_msg = sys_match.group(1).strip()
            # Extract user part (after <</SYS>> to [/INST])
            user_match = re.search(r'<</SYS>>\s*(.*?)\s*\[/INST\]', prompt, re.DOTALL)
            if user_match:
                user_msg = user_match.group(1).strip()

        response = self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        if self.backend == "gguf":
            return f"Llama 2 7B (GGUF, local)"
        elif self.backend == "api":
            return f"OpenAI {OPENAI_MODEL} (API)"
        return "Not loaded"


if __name__ == "__main__":
    llm = LLM(backend="auto")
    print(f"\nBackend: {llm.backend_name}")
    test_prompt = "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\nWhat is deep learning? [/INST]"
    print(f"Test prompt: {test_prompt[:80]}...")
    answer = llm.generate(test_prompt)
    print(f"Answer: {answer[:300]}")
