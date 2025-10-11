from dnd_session_transcribe.util.config import DiarizationConfig


def test_diarization_defaults_are_tuned_for_two_speakers():
    cfg = DiarizationConfig()

    assert cfg.num_speakers == 2
    assert cfg.seg_min_off == 0.05
    assert cfg.turn_min_off == 0.05
