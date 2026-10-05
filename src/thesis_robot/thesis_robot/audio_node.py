#!/usr/bin/env python3
"""
Audio Node — the ears of the Audio-to-Action pipeline.

Captures the microphone on THIS machine (REAL-1), transcribes with
faster-whisper, and hands the text to the rest of the system.

Subscribes: /audio_control   (std_msgs/String)
              start   begin recording now
              stop    stop recording and transcribe
              cancel  stop recording and discard
              arm     hands-free: listen continuously (voice activity detection)
              disarm  stop hands-free listening
Publishes:  /audio_status     (String)  LOADING / IDLE / LISTENING / RECORDING / TRANSCRIBING / HEARD: ...
            /audio_level      (Float32) microphone RMS, ~10 Hz, for the dashboard meter
            /voice_transcript (String)  JSON {text, ts, audio_s, latency_s}  every accepted transcript
            /voice_command    (String)  the text, ONLY when auto_send:=true

A voice command moves a real arm, so by default (auto_send:=false) a
transcript is only SHOWN in the dashboard and is sent to the planner when
the operator confirms it (or turns on auto-send there).

Parameters: mode (dashboard|vad), model_size (small.en), device (cpu|cuda),
cpu_threads, input_device (-1 = default), max_record_seconds, silence_threshold_s,
auto_send, publish_level.
"""
import json
import queue
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

from thesis_robot.audio_logic import COMMAND_PROMPT, Segmenter, clean_transcript

CHUNK_S = 0.1


class AudioNode(Node):

    def __init__(self):
        super().__init__('audio_node')
        self.declare_parameter('mode', 'dashboard')
        self.declare_parameter('model_size', 'small.en')
        # CPU int8 by default: ~1.3 s per command on this machine, and it keeps the GPU for
        # YOLO/Ollama. GPU needs CUDA-12 cuBLAS/cuDNN, which the CUDA-13 venv doesn't have.
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('cpu_threads', 8)
        self.declare_parameter('language', 'en')
        # Input device by NAME substring (or index). The system 'default' / pulse
        # input on REAL-1 delivers digital silence; the real capture device is
        # 'ALC897 Analog'. Empty or -1 means the system default.
        self.declare_parameter('input_device', 'ALC897 Analog')
        self.declare_parameter('input_gain', 1.0)
        self.declare_parameter('max_record_seconds', 10.0)
        self.declare_parameter('silence_threshold_s', 1.2)
        self.declare_parameter('auto_send', False)
        self.declare_parameter('publish_level', True)

        p = self.get_parameter
        self.mode = p('mode').value
        self.gain = float(p('input_gain').value)
        self.max_seconds = float(p('max_record_seconds').value)
        self.auto_send = bool(p('auto_send').value)
        self.publish_level = bool(p('publish_level').value)
        self.language = p('language').value

        self.cmd_pub = self.create_publisher(String, '/voice_command', 10)
        self.transcript_pub = self.create_publisher(String, '/voice_transcript', 10)
        self.status_pub = self.create_publisher(String, '/audio_status', 10)
        self.level_pub = self.create_publisher(Float32, '/audio_level', 10)
        self.create_subscription(String, '/audio_control', self._on_control, 10)

        self.segmenter = None            # built once the device sample rate is known
        self._armed = self.mode == 'vad'
        self._manual = False            # a start..stop recording is in progress
        self._manual_frames = []
        self._busy = False              # transcription in progress
        self._lock = threading.Lock()
        self._chunks = queue.Queue(maxsize=200)
        self._running = True
        self._last_level_pub = 0.0
        self._status = 'LOADING'

        self._publish_status('LOADING — loading speech model')
        self._load_model(p('model_size').value, p('device').value, int(p('cpu_threads').value))

        import sounddevice as sd
        dev = self._resolve_device(sd, p('input_device').value)
        info = sd.query_devices(dev, 'input') if dev is not None else sd.query_devices(kind='input')
        # capture at the device's native rate (hw: devices do not resample) and
        # resample the finished utterance to 16 kHz for Whisper
        self.sample_rate = int(info['default_samplerate'])
        self.segmenter = Segmenter(
            chunk_s=CHUNK_S, silence_s=float(p('silence_threshold_s').value),
            max_s=self.max_seconds)
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype='float32',
            blocksize=int(self.sample_rate * CHUNK_S), device=dev, callback=self._on_audio)
        self._stream.start()
        self.get_logger().info(f'microphone: {info["name"]} @ {self.sample_rate} Hz')
        self.create_timer(2.0, self._heartbeat)

        threading.Thread(target=self._pump, daemon=True).start()
        self._idle_status()
        self.get_logger().info(
            f'AudioNode ready — mode={self.mode} auto_send={self.auto_send} '
            f'model={p("model_size").value}')

    # ── model ────────────────────────────────────────────────────────
    def _load_model(self, size, device, cpu_threads):
        from faster_whisper import WhisperModel
        try:
            compute = 'float16' if device == 'cuda' else 'int8'
            self.asr = WhisperModel(size, device=device, compute_type=compute,
                                    cpu_threads=cpu_threads)
        except Exception as e:
            self.get_logger().error(f'Whisper on {device} failed ({e}) — falling back to CPU int8')
            self.asr = WhisperModel(size, device='cpu', compute_type='int8',
                                    cpu_threads=cpu_threads)

    @staticmethod
    def _resolve_device(sd, value):
        v = str(value).strip()
        if v in ('', '-1'):
            return None
        if v.lstrip('-').isdigit():
            return int(v)
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0 and v.lower() in d['name'].lower():
                return i
        raise RuntimeError(f'no input device matching {v!r}')

    # ── capture ──────────────────────────────────────────────────────
    def _on_audio(self, indata, frames, time_info, status):
        # PortAudio thread: only copy and enqueue
        try:
            self._chunks.put_nowait(indata[:, 0] * self.gain)
        except queue.Full:
            pass

    def _pump(self):
        while self._running:
            try:
                chunk = self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._handle_chunk(chunk)
            except Exception as e:
                self.get_logger().error(f'audio pump error: {e}')

    def _handle_chunk(self, chunk):
        rms = Segmenter.rms(chunk)
        now = time.monotonic()
        if self.publish_level and now - self._last_level_pub >= CHUNK_S:
            self._last_level_pub = now
            self.level_pub.publish(Float32(data=rms))

        if self._manual:
            self._manual_frames.append(chunk)
            if len(self._manual_frames) * CHUNK_S >= self.max_seconds:
                self._finish_manual(transcribe=True)
            return
        if self._armed and not self._busy:
            was_recording = self.segmenter.recording
            audio = self.segmenter.feed(chunk)
            if self.segmenter.recording and not was_recording:
                self._publish_status('RECORDING — speech detected')
            if audio is not None:
                self._transcribe_async(audio)
            elif was_recording and not self.segmenter.recording:
                self._idle_status()          # a click/noise, discarded

    # ── control ──────────────────────────────────────────────────────
    def _on_control(self, msg):
        action = msg.data.strip().lower()
        if action == 'start':
            if self._busy:
                return
            self.segmenter.reset()
            self._manual_frames = []
            self._manual = True
            self._publish_status('RECORDING — press the mic again to stop')
        elif action == 'stop':
            self._finish_manual(transcribe=True)
        elif action == 'cancel':
            self._finish_manual(transcribe=False)
        elif action == 'arm':
            self._armed = True
            self.segmenter.reset()
            self._idle_status()
        elif action == 'disarm':
            self._armed = False
            self.segmenter.reset()
            self._idle_status()
        else:
            self.get_logger().warn(f'unknown /audio_control action {action!r}')

    def _finish_manual(self, transcribe):
        if not self._manual:
            return
        self._manual = False
        frames, self._manual_frames = self._manual_frames, []
        if not transcribe or not frames:
            self._idle_status()
            return
        self._transcribe_async(np.concatenate(frames))

    # ── transcription ────────────────────────────────────────────────
    def _transcribe_async(self, audio):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self._publish_status('TRANSCRIBING')
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _prepare(self, audio):
        """Resample to 16 kHz and bring a quiet microphone up to a usable level."""
        if self.sample_rate != 16000:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(16000, self.sample_rate)
            audio = resample_poly(audio, 16000 // g, self.sample_rate // g)
        audio = np.asarray(audio, dtype=np.float32)
        peak = float(np.abs(audio).max()) if len(audio) else 0.0
        if 1e-4 < peak < 0.3:
            audio = audio * (0.5 / peak)
        return audio

    def _transcribe(self, audio):
        t0 = time.time()
        duration = len(audio) / self.sample_rate
        try:
            audio = self._prepare(audio)
            segments, _info = self.asr.transcribe(
                audio, language=self.language, beam_size=3, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=COMMAND_PROMPT, condition_on_previous_text=False,
                temperature=0.0, no_speech_threshold=0.6)
            text = clean_transcript(' '.join(s.text.strip() for s in segments))
        except Exception as e:
            self.get_logger().error(f'transcription error: {e}')
            text = ''
        latency = time.time() - t0
        try:
            if not text:
                self._publish_status('NO SPEECH — nothing recognised')
            else:
                self.get_logger().info(f'Transcribed ({latency:.2f}s): "{text}"')
                self.transcript_pub.publish(String(data=json.dumps({
                    'text': text, 'ts': time.time(),
                    'audio_s': round(duration, 2),
                    'latency_s': round(latency, 2)})))
                self._publish_status(f'HEARD: {text}')
                if self.auto_send:
                    self.cmd_pub.publish(String(data=text))
        finally:
            self._busy = False
            time.sleep(1.5)                 # let the UI show the result
            if not self._busy and not self._manual and not self.segmenter.recording:
                self._idle_status()

    # ── helpers ──────────────────────────────────────────────────────
    def _idle_status(self):
        self._publish_status('LISTENING — hands-free' if self._armed else 'IDLE')

    def _publish_status(self, text):
        self._status = text
        self.status_pub.publish(String(data=text))

    def _heartbeat(self):
        # a dashboard that (re)started after our last status change would
        # otherwise never learn the current state
        self.status_pub.publish(String(data=self._status))

    def destroy_node(self):
        self._running = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = AudioNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
