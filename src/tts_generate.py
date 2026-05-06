import asyncio
import os
import edge_tts
import soundfile as sf
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SPEAKERS = {
    "mohit": "en-IN-PrabhatNeural",
    "rahul": "en-US-GuyNeural",
    "priya": "en-IN-NeerjaNeural",
}

ENROLLMENT_SENTENCES = [
    "Hello, this is my voice sample for the smart assistant.",
    "I am speaking normally so the system can remember my voice.",
    "This sample is used only for speaker identification.",
    "My voice should be matched with future commands.",
    "The smart assistant should recognize me correctly.",
]

COMMANDS = {
    "mohit": [
        "Hey assistant, turn on my bedroom light.",
        "Hey assistant, what is my schedule today?",
        "Hey assistant, turn off the fan.",
    ],
    "rahul": [
        "Hey assistant, play my study playlist.",
        "Hey assistant, set an alarm for seven AM.",
        "Hey assistant, turn on the AC.",
    ],
    "priya": [
        "Hey assistant, remind me to call home.",
        "Hey assistant, play relaxing music.",
        "Hey assistant, turn off the room light.",
    ],
}


async def _synthesize(text: str, voice: str, out_path: str):
    communicate = edge_tts.Communicate(text, voice)
    tmp_mp3 = out_path.replace(".wav", "_tmp.mp3")
    await communicate.save(tmp_mp3)
    _convert_mp3_to_wav(tmp_mp3, out_path)
    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)


def _convert_mp3_to_wav(mp3_path: str, wav_path: str):
    """Try multiple backends to convert MP3 → 16 kHz mono WAV."""
    # 1) pydub (needs ffmpeg on PATH or ffmpeg binary)
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return
    except Exception:
        pass

    # 2) torchaudio (bundled backend, no ffmpeg needed)
    try:
        import torchaudio
        waveform, sr = torchaudio.load(mp3_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        torchaudio.save(wav_path, waveform, 16000)
        return
    except Exception:
        pass

    # 3) soundfile + audioread
    try:
        import audioread
        import soundfile as sf
        import numpy as np
        with audioread.audio_open(mp3_path) as f:
            sr = f.samplerate
            ch = f.channels
            raw = b"".join(f)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sf.write(wav_path, audio, 16000)
        return
    except Exception:
        pass

    # 4) Last resort — keep the mp3 bytes but name it .wav (imperfect but won't crash)
    import shutil
    shutil.copy(mp3_path, wav_path)
    print(f"[TTS] Warning: could not properly convert {mp3_path} — copied raw bytes as WAV.")


async def generate_enrollment(speaker: str, voice: str):
    out_dir = os.path.join(DATA_DIR, "enrolled_speakers", speaker)
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for i, sentence in enumerate(ENROLLMENT_SENTENCES):
        out_path = os.path.join(out_dir, f"enroll_{i+1}.wav")
        if not os.path.exists(out_path):
            tasks.append(_synthesize(sentence, voice, out_path))
    if tasks:
        await asyncio.gather(*tasks)
        print(f"[TTS] Enrollment audio generated for {speaker}")
    else:
        print(f"[TTS] Enrollment audio already exists for {speaker}")


async def generate_commands(speaker: str, voice: str):
    out_dir = os.path.join(DATA_DIR, "generated")
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    paths = []
    for i, cmd in enumerate(COMMANDS[speaker]):
        out_path = os.path.join(out_dir, f"{speaker}_cmd_{i+1}.wav")
        paths.append(out_path)
        if not os.path.exists(out_path):
            tasks.append(_synthesize(cmd, voice, out_path))
    if tasks:
        await asyncio.gather(*tasks)
        print(f"[TTS] Command audio generated for {speaker}")
    else:
        print(f"[TTS] Command audio already exists for {speaker}")
    return paths


async def generate_all():
    for speaker, voice in SPEAKERS.items():
        await generate_enrollment(speaker, voice)
        await generate_commands(speaker, voice)
    print("[TTS] All audio generated.")


def run_generate_all():
    asyncio.run(generate_all())


def get_command_path(speaker: str, cmd_index: int) -> str:
    return os.path.join(DATA_DIR, "generated", f"{speaker}_cmd_{cmd_index+1}.wav")


if __name__ == "__main__":
    run_generate_all()
