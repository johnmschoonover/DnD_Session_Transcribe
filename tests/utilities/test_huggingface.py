import pytest

from dnd_session_transcribe.adapters import huggingface


HF_TOKEN_ENV = huggingface.HF_TOKEN_ENV
ensure_hf_token = huggingface.ensure_hf_token


def test_ensure_hf_token_missing(monkeypatch):
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)

    expected_message = (
        "[HF] Missing HUGGINGFACE_TOKEN. Set it e.g.:\n"
        "  conda env config vars set HUGGINGFACE_TOKEN=hf_xxx\n"
        "  conda deactivate && conda activate whisperx\n"
    )

    with pytest.raises(SystemExit) as exc_info:
        ensure_hf_token()

    assert str(exc_info.value) == expected_message


def test_ensure_hf_token_present(monkeypatch):
    token = "hf_dummy_token"
    monkeypatch.setenv(HF_TOKEN_ENV, token)

    assert ensure_hf_token() == token
