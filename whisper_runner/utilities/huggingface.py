import os

HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"

def ensure_hf_token() -> str:
    tok = os.getenv(HF_TOKEN_ENV)
    if not tok:
        raise SystemExit(
            f"[HF] Missing {HF_TOKEN_ENV}. Set it e.g.:\n"
            f"  conda env config vars set {HF_TOKEN_ENV}=hf_xxx\n"
            f"  conda deactivate && conda activate whisperx\n"
        )
    return tok