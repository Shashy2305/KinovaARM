import numpy as np

from thesis_robot.audio_logic import COMMAND_PROMPT, Segmenter, clean_transcript

SR = 16000
CH = int(SR * 0.1)


def chunk(amp):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(CH) * amp).astype(np.float32)


def run(seg, spec):
    """spec: list of (amplitude, n_chunks); returns finished utterances."""
    out = []
    for amp, n in spec:
        for _ in range(n):
            a = seg.feed(chunk(amp))
            if a is not None:
                out.append(a)
    return out


def test_silence_produces_nothing():
    assert run(Segmenter(), [(0.002, 100)]) == []


def test_speech_burst_is_returned_with_pre_roll_and_trailing_silence():
    out = run(Segmenter(), [(0.002, 10), (0.2, 12), (0.002, 20)])
    assert len(out) == 1
    secs = len(out[0]) / SR
    assert 1.2 + 1.2 - 0.1 <= secs <= 0.3 + 1.2 + 1.2 + 0.2     # pre-roll + speech + silence tail


def test_short_click_is_dropped():
    assert run(Segmenter(), [(0.002, 10), (0.3, 1), (0.002, 20)]) == []


def test_max_length_cutoff():
    out = run(Segmenter(max_s=3.0), [(0.002, 5), (0.2, 100)])
    assert len(out) >= 1 and len(out[0]) / SR <= 3.1


def test_steady_loud_noise_raises_the_threshold_instead_of_triggering():
    seg = Segmenter()
    run(seg, [(0.01, 200)])               # steady hum below min_threshold
    assert run(seg, [(0.01, 50)]) == []
    assert seg.threshold >= seg.min_threshold


def test_two_utterances_are_separated():
    out = run(Segmenter(), [(0.002, 10), (0.2, 8), (0.002, 20), (0.2, 8), (0.002, 20)])
    assert len(out) == 2


def test_clean_transcript_drops_hallucinations_and_prompt_leaks():
    for bad in ('', ' ', 'Thank you.', 'you', '...', 'Thanks for watching!', '1 2 3 ...'.replace('1 2 3', '')):
        assert clean_transcript(bad) == ''
    assert clean_transcript(COMMAND_PROMPT) == ''
    assert clean_transcript('Robot arm commands: go home. go near the cup.') == ''


def test_clean_transcript_keeps_real_commands():
    assert clean_transcript('  go near   the cup. ') == 'go near the cup.'
    assert clean_transcript('Pick up the bottle') == 'Pick up the bottle'
    assert clean_transcript('home') == 'home'
