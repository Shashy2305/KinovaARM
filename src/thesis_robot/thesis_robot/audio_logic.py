"""
Pure-python pieces of the voice input path (no ROS, no audio hardware), so
they can be unit tested: the speech segmenter and the transcript cleaner.
"""
import collections
import re

import numpy as np

# Whisper is biased towards the words it is likely to hear. Without this a
# small model turns "cup" into "cop"/"cap" and "go near" into "gonna".
COMMAND_PROMPT = (
    'Robot arm commands: go home. go near the cup. pick up the cup. '
    'pick up the bottle. move to the left. move to the right. '
    'open the gripper. close the gripper. place it down.'
)

# Things Whisper invents on silence / fan noise. A hallucinated "thank you"
# must never reach a robot arm.
_HALLUCINATIONS = {
    '', 'you', 'thank you', 'thanks', 'thank you.', 'thanks for watching',
    'thank you for watching', 'bye', 'bye.', 'the end', 'so', 'uh', 'um',
    'hmm', 'mm', 'oh',
}


def _norm(text):
    return re.sub(r'[^a-z0-9 ]+', '', text.lower()).strip()


def clean_transcript(text, prompt=COMMAND_PROMPT):
    """Returns the cleaned command text, or '' if it should be discarded."""
    if not text:
        return ''
    t = ' '.join(text.split()).strip()
    n = _norm(t)
    if n in {_norm(h) for h in _HALLUCINATIONS}:
        return ''
    if not re.search(r'[a-zA-Z]', t):
        return ''
    # the initial prompt leaking back out on near-silence
    if len(n) > 25 and n in _norm(prompt):
        return ''
    return t


class Segmenter:
    """Energy-based voice activity detection with an adaptive noise floor.

    feed() takes fixed-size float32 chunks and returns a finished utterance
    (np.ndarray) when speech has ended, else None. A short pre-roll keeps
    the first syllable; clicks shorter than `min_speech_s` are dropped.
    """

    def __init__(self, chunk_s=0.1, min_speech_s=0.35, silence_s=1.2, max_s=10.0,
                 pre_roll_s=0.3, min_threshold=0.002, noise_mult=3.0):
        self.chunk_s = chunk_s
        self.min_speech_s = min_speech_s
        self.silence_chunks = max(1, int(round(silence_s / chunk_s)))
        self.max_chunks = max(1, int(round(max_s / chunk_s)))
        self.min_threshold = min_threshold
        self.noise_mult = noise_mult
        self.noise_floor = 0.001
        self._pre = collections.deque(maxlen=max(1, int(round(pre_roll_s / chunk_s))))
        self.reset()

    def reset(self):
        self._frames = []
        self._recording = False
        self._silent = 0
        self._speech_chunks = 0
        self._pre.clear()

    @property
    def recording(self):
        return self._recording

    @property
    def threshold(self):
        return max(self.min_threshold, self.noise_floor * self.noise_mult)

    @staticmethod
    def rms(chunk):
        return float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if len(chunk) else 0.0

    def feed(self, chunk):
        loud = self.rms(chunk) > self.threshold
        if not self._recording:
            if loud:
                self._recording = True
                self._frames = list(self._pre) + [chunk]
                self._silent = 0
                self._speech_chunks = 1
            else:
                self._pre.append(chunk)
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * self.rms(chunk)
            return None

        self._frames.append(chunk)
        if loud:
            self._speech_chunks += 1
            self._silent = 0
        else:
            self._silent += 1
        if self._silent >= self.silence_chunks or len(self._frames) >= self.max_chunks:
            audio = np.concatenate(self._frames)
            long_enough = self._speech_chunks * self.chunk_s >= self.min_speech_s
            self.reset()
            return audio if long_enough else None
        return None
