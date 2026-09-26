#!/usr/bin/env python3
"""
Audio Node — the ears of the Audio-to-Action pipeline.

Listens to the microphone, transcribes speech using faster-whisper,
and publishes the transcribed command to /voice_command.

Subscribes to:  nothing (reads from microphone hardware)
Publishes to:   /voice_command  (std_msgs/String)
                /audio_status   (std_msgs/String)

Two modes:
  push_to_talk: press Enter to start recording, Enter again to stop
  vad:          voice activity detection — starts/stops automatically

Your thesis contribution: the human interface layer that converts
natural language speech into the text commands the LLM planner receives.
"""
import rclpy, threading, time, sys
from rclpy.node import Node
from std_msgs.msg import String

class AudioNode(Node):

    def __init__(self):
        super().__init__('audio_node')

        # ── parameters ───────────────────────────────────────────────
        self.declare_parameter('mode', 'push_to_talk')
        self.declare_parameter('model_size', 'large-v3')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('language', 'en')
        self.declare_parameter('sample_rate', 16000)
        self.declare_parameter('max_record_seconds', 10)
        self.declare_parameter('silence_threshold_s', 1.5)

        self.mode          = self.get_parameter('mode').value
        self.model_size    = self.get_parameter('model_size').value
        self.device        = self.get_parameter('device').value
        self.language      = self.get_parameter('language').value
        self.sample_rate   = self.get_parameter('sample_rate').value
        self.max_seconds   = self.get_parameter('max_record_seconds').value

        # ── publishers ───────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(String, '/voice_command',  10)
        self.status_pub = self.create_publisher(String, '/audio_status',   10)

        # ── load models ──────────────────────────────────────────────
        self._publish_status('LOADING — initialising faster-whisper...')
        self.get_logger().info(
            f'Loading faster-whisper {self.model_size} on {self.device}...'
        )

        try:
            from faster_whisper import WhisperModel
            self.asr = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type='float16' if self.device == 'cuda' else 'float32'
            )
            self.get_logger().info('faster-whisper loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load faster-whisper: {e}')
            self.get_logger().error(
                'Install with: pip install faster-whisper'
            )
            raise

        try:
            import sounddevice as sd
            self.sd = sd
        except Exception as e:
            self.get_logger().error(f'sounddevice not found: {e}')
            self.get_logger().error(
                'Install with: pip install sounddevice'
            )
            raise

        self._publish_status(f'READY — mode: {self.mode}')
        self.get_logger().info(
            f'AudioNode ready — mode={self.mode} '
            f'model={self.model_size} device={self.device}'
        )

        # ── start listening loop in background thread ─────────────────
        self._running = True
        t = threading.Thread(target=self._listen_loop, daemon=True)
        t.start()

    # ── MAIN LISTEN LOOP ─────────────────────────────────────────────
    def _listen_loop(self):
        """Runs in background — handles recording and transcription."""
        if self.mode == 'push_to_talk':
            self._push_to_talk_loop()
        else:
            self._vad_loop()

    # ── PUSH TO TALK MODE ────────────────────────────────────────────
    def _push_to_talk_loop(self):
        """Press Enter to start, Enter to stop. Most reliable for demos."""
        self.get_logger().info(
            '\n'
            '==========================================\n'
            '  AUDIO NODE — Push-to-Talk Mode\n'
            '  Press ENTER to start recording\n'
            '  Press ENTER again to stop\n'
            '==========================================\n'
        )

        while self._running and rclpy.ok():
            try:
                self._publish_status('READY — press Enter to speak')
                input('\n>> Press Enter to speak your command...\n')

                if not rclpy.ok():
                    break

                self._publish_status('RECORDING...')
                self.get_logger().info(
                    f'Recording... (max {self.max_seconds}s, '
                    f'press Enter to stop early)'
                )

                # Record audio
                audio = self._record()

                if audio is None or len(audio) == 0:
                    continue

                # Transcribe
                self._publish_status('TRANSCRIBING...')
                self.get_logger().info('Transcribing...')

                text = self._transcribe(audio)

                if not text:
                    self.get_logger().warn('No speech detected — try again')
                    self._publish_status('NO SPEECH — try again')
                    continue

                self.get_logger().info(f'Transcribed: "{text}"')
                self._publish_status(f'HEARD: {text}')

                # Publish to /voice_command
                msg = String()
                msg.data = text
                self.cmd_pub.publish(msg)
                self.get_logger().info(
                    f'Published to /voice_command: "{text}"'
                )

            except KeyboardInterrupt:
                break
            except EOFError:
                # stdin closed — wait and retry
                time.sleep(1.0)
            except Exception as e:
                self.get_logger().error(f'Listen loop error: {e}')
                time.sleep(1.0)

    # ── VAD MODE ─────────────────────────────────────────────────────
    def _vad_loop(self):
        """Voice activity detection — fully hands-free."""
        import numpy as np

        self.get_logger().info(
            '\n'
            '==========================================\n'
            '  AUDIO NODE — VAD Mode (hands-free)\n'
            '  Speak naturally — I will listen\n'
            '==========================================\n'
        )

        CHUNK      = int(self.sample_rate * 0.1)   # 100ms chunks
        ENERGY_THR = 0.02                           # tune for your mic
        SILENCE_S  = self.get_parameter('silence_threshold_s').value
        silence_chunks = int(SILENCE_S / 0.1)

        self._publish_status('LISTENING (VAD active)...')

        while self._running and rclpy.ok():
            try:
                frames      = []
                recording   = False
                silent_count = 0

                with self.sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=CHUNK
                ) as stream:

                    while self._running and rclpy.ok():
                        chunk, _ = stream.read(CHUNK)
                        energy = float((chunk**2).mean()**0.5)

                        if not recording:
                            if energy > ENERGY_THR:
                                recording = True
                                silent_count = 0
                                frames = [chunk.copy()]
                                self.get_logger().info(
                                    'Speech detected — recording...'
                                )
                                self._publish_status('RECORDING...')
                        else:
                            frames.append(chunk.copy())
                            if energy < ENERGY_THR:
                                silent_count += 1
                                if silent_count >= silence_chunks:
                                    # End of speech
                                    break
                            else:
                                silent_count = 0

                            if len(frames) > self.max_seconds * 10:
                                break

                if frames and recording:
                    import numpy as np
                    audio = np.concatenate(frames, axis=0).flatten()
                    self._publish_status('TRANSCRIBING...')
                    text = self._transcribe(audio)
                    if text:
                        self.get_logger().info(f'Transcribed: "{text}"')
                        msg = String()
                        msg.data = text
                        self.cmd_pub.publish(msg)
                        self._publish_status(f'HEARD: {text}')
                    else:
                        self._publish_status('LISTENING (VAD active)...')

            except Exception as e:
                self.get_logger().error(f'VAD loop error: {e}')
                time.sleep(1.0)

    # ── RECORDING ────────────────────────────────────────────────────
    def _record(self):
        """Record audio until Enter is pressed or max_seconds reached."""
        import numpy as np

        frames = []
        stop_event = threading.Event()

        def wait_for_enter():
            try:
                input()
            except Exception:
                pass
            stop_event.set()

        stopper = threading.Thread(target=wait_for_enter, daemon=True)
        stopper.start()

        try:
            with self.sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=int(self.sample_rate * 0.1)
            ) as stream:

                deadline = time.time() + self.max_seconds
                while not stop_event.is_set() and time.time() < deadline:
                    chunk, _ = stream.read(int(self.sample_rate * 0.1))
                    frames.append(chunk.copy())

        except Exception as e:
            self.get_logger().error(f'Recording error: {e}')
            return None

        if not frames:
            return None

        return np.concatenate(frames, axis=0).flatten()

    # ── TRANSCRIPTION ─────────────────────────────────────────────────
    def _transcribe(self, audio_np):
        """Run faster-whisper on audio array, return text string."""
        try:
            segments, info = self.asr.transcribe(
                audio_np,
                beam_size=5,
                language=self.language,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=700,
                    threshold=0.5
                )
            )

            text = ' '.join(
                seg.text.strip() for seg in segments
            ).strip()

            if text:
                self.get_logger().info(
                    f'Detected language: {info.language} '
                    f'({info.language_probability:.0%})'
                )

            return text

        except Exception as e:
            self.get_logger().error(f'Transcription error: {e}')
            return None

    # ── HELPERS ──────────────────────────────────────────────────────
    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def destroy_node(self):
        self._running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AudioNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'AudioNode failed to start: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
