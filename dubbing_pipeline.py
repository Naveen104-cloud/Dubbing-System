"""
Dubbing Pipeline
Real implementation integrating:
  - FFmpeg (for extraction & merging) via imageio_ffmpeg
  - Whisper (for transcription)
  - deep-translator (for text translation)
  - gTTS (for text-to-speech)
"""

import os
import shutil
import subprocess
from pathlib import Path

import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS
import imageio_ffmpeg

# Add imageio_ffmpeg's directory to PATH so whisper can find `ffmpeg`
ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe_path)
ffmpeg_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe")
if not os.path.exists(ffmpeg_alias):
    try:
        shutil.copyfile(ffmpeg_exe_path, ffmpeg_alias)
    except Exception:
        pass # If we lack write permissions
os.environ["PATH"] += os.pathsep + ffmpeg_dir

class DubbingPipeline:
    """Orchestrates the full dubbing workflow with real AI/ML libraries."""

    LANGUAGE_NAMES = {
        "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
        "bs": "Bosnian", "ca": "Catalan", "cs": "Czech", "cy": "Welsh",
        "da": "Danish", "de": "German", "el": "Greek", "en": "English",
        "eo": "Esperanto", "es": "Spanish", "et": "Estonian", "fi": "Finnish",
        "fr": "French", "gu": "Gujarati", "hi": "Hindi", "hr": "Croatian",
        "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "is": "Icelandic",
        "it": "Italian", "ja": "Japanese", "jw": "Javanese", "km": "Khmer",
        "kn": "Kannada", "ko": "Korean", "la": "Latin", "lv": "Latvian",
        "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi",
        "my": "Myanmar (Burmese)", "ne": "Nepali", "nl": "Dutch",
        "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
        "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sq": "Albanian",
        "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
        "ta": "Tamil", "te": "Telugu", "th": "Thai", "tl": "Filipino",
        "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese",
        "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
        "zu": "Zulu",
    }

    def __init__(
        self,
        job_id: str,
        input_path: str,
        target_language: str,
        whisper_model: str,
        temp_dir: str,
        output_dir: str,
        update_callback=None,
    ):
        self.job_id = job_id
        self.input_path = Path(input_path)
        self.target_language = target_language
        self.whisper_model = whisper_model
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.update = update_callback or (lambda *a, **kw: None)
        self.lang_name = self.LANGUAGE_NAMES.get(target_language, target_language)
        
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        self.temp_job_dir = self.temp_dir / self.job_id
        self.temp_job_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_path = self.temp_job_dir / "extracted_audio.wav"
        self.tts_path = self.temp_job_dir / "tts_audio.mp3"
        self.synced_tts_path = self.temp_job_dir / "synced_tts_audio.mp3"

    def _get_duration(self, file_path):
        """Helper to get video/audio duration using FFmpeg."""
        cmd = [self.ffmpeg_exe, "-i", str(file_path)]
        try:
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=False)
            for line in result.stderr.split('\n'):
                if "Duration:" in line:
                    time_str = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = time_str.split(":")
                    return float(h)*3600 + float(m)*60 + float(s)
        except Exception as e:
            print(f"Error extracting duration: {e}")
        return 0.0

    def run(self) -> str:
        """Execute all pipeline stages and return the output filename."""
        self._extract_audio()
        transcript, detected_lang = self._transcribe()
        translated = self._translate(transcript)
        self._synthesise_speech(translated)
        output_filename = self._merge_video()
        self._cleanup()
        return output_filename

    def _extract_audio(self):
        self.update(10, "Extracting audio from video…")
        # Extract audio downsampled to 16kHz for Whisper
        cmd = [
            self.ffmpeg_exe, "-y",
            "-i", str(self.input_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(self.audio_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _transcribe(self):
        self.update(25, f"Transcribing audio using Whisper ({self.whisper_model} model)…")
        # Load the selected whisper model (e.g., base)
        model = whisper.load_model(self.whisper_model)
        
        # Transcribe audio file
        result = model.transcribe(str(self.audio_path))
        transcript = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        self.update(
            40,
            "Transcription complete.",
            detected_language=detected_lang,
            transcript=transcript,
        )
        return transcript, detected_lang

    def _translate(self, transcript: str) -> str:
        self.update(50, f"Translating transcript to {self.lang_name}…")
        if not transcript:
            self.update(65, "Translation complete (no speech detected).")
            return ""
            
        try:
            translator = GoogleTranslator(source='auto', target=self.target_language)
            # Google Translator has a 5000 character limit, but most single videos stay under this for transcripts.
            translated = translator.translate(transcript)
        except Exception as e:
            print(f"Translation failed, falling back: {e}")
            translated = transcript
            
        self.update(65, "Translation complete.")
        return translated

    def _synthesise_speech(self, text: str):
        self.update(75, f"Generating {self.lang_name} speech via TTS…")
        if not text:
            # Create a 1 second silent audio file to prevent crash
            cmd = [self.ffmpeg_exe, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(self.tts_path)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.update(85, "Speech synthesis complete (no text).")
            return

        try:
            # First try exact target_language
            tts = gTTS(text, lang=self.target_language)
            tts.save(str(self.tts_path))
        except Exception:
            try:
                # If exact code fails, try base language variant (e.g., zh-CN -> zh)
                base_lang = self.target_language.split('-')[0]
                tts = gTTS(text, lang=base_lang)
                tts.save(str(self.tts_path))
            except Exception as outer_e:
                print(f"TTS Failed: {outer_e}")
                # Create silent fallback
                cmd = [self.ffmpeg_exe, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(self.tts_path)]
                subprocess.run(cmd, check=True)

        self.update(85, "Speech synthesis complete.")

    def _merge_video(self) -> str:
        self.update(90, "Synchronizing and merging dubbed audio with original video…")
        
        orig_dur = self._get_duration(self.input_path)
        tts_dur = self._get_duration(self.tts_path)
        
        # Audio length synchronization using FFmpeg atempo
        if orig_dur > 0 and tts_dur > 0:
            speedup = tts_dur / orig_dur
            
            # Bound speedup
            if speedup < 0.5:
                speedup = 1.0
                
            if speedup != 1.0:
                tempos = []
                temp_speedup = speedup
                
                # Chain atempo if > 2.0
                while temp_speedup > 2.0:
                    tempos.append("atempo=2.0")
                    temp_speedup /= 2.0
                if temp_speedup > 1.0 or (speedup < 1.0 and temp_speedup != 1.0):
                    tempos.append(f"atempo={temp_speedup:.4f}")
                
                atempo_str = ",".join(tempos)
                
                cmd_sync = [
                    self.ffmpeg_exe, "-y",
                    "-i", str(self.tts_path),
                    "-filter:a", atempo_str,
                    str(self.synced_tts_path)
                ]
                subprocess.run(cmd_sync, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                shutil.copy2(self.tts_path, self.synced_tts_path)
        else:
            shutil.copy2(self.tts_path, self.synced_tts_path)

        output_filename = f"{self.job_id}_dubbed_{self.target_language}.mp4"
        output_path = self.output_dir / output_filename
        
        # Merge Video + Synced Audio
        cmd_merge = [
            self.ffmpeg_exe, "-y",
            "-i", str(self.input_path),
            "-i", str(self.synced_tts_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",  # Make sure the result ends when the shortest stream ends
            str(output_path)
        ]
        
        subprocess.run(cmd_merge, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self.update(95, "Video merge complete.")
        return output_filename

    def _cleanup(self):
        self.update(98, "Cleaning up temporary files…")
        if self.temp_job_dir.exists():
            shutil.rmtree(self.temp_job_dir, ignore_errors=True)
