from dnd_session_transcribe.util.config import DiarizationConfig


def test_diarization_defaults_remain_general_purpose():
    cfg = DiarizationConfig()

    assert cfg.num_speakers == 5
    assert cfg.seg_min_off == 0.10
    assert cfg.turn_min_off == 0.10


def test_diarization_config_can_be_overridden_for_alternating_dialogue():
    cfg = DiarizationConfig(num_speakers=2, seg_min_off=0.05, turn_min_off=0.05)

    assert cfg.num_speakers == 2
    assert cfg.seg_min_off == 0.05
    assert cfg.turn_min_off == 0.05
