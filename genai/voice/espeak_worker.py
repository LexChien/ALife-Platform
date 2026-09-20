"""Offline eSpeak NG speech synthesis; isolated native library lifecycle."""
import ctypes
import ctypes.util
import json
from pathlib import Path
import sys
import wave


def synthesize(text, output, voice="en", rate=165):
    library = ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")
    if not library:
        raise RuntimeError("eSpeak NG library missing; install libespeak-ng1")
    lib = ctypes.CDLL(library)
    lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.espeak_Initialize.restype = ctypes.c_int
    sample_rate = lib.espeak_Initialize(2, 0, None, 0)
    if sample_rate <= 0:
        raise RuntimeError("eSpeak initialization failed")
    chunks = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    @callback_type
    def callback(samples, count, events):
        if samples and count:
            chunks.append(ctypes.string_at(samples, count * 2))
        return 0

    lib.espeak_SetSynthCallback.argtypes = [callback_type]
    lib.espeak_SetSynthCallback(callback)
    lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
    if lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise ValueError(f"eSpeak voice unavailable: {voice}")
    lib.espeak_SetParameter(1, int(rate), 0)
    lib.espeak_Synth.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int,
                                ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
    data = text.encode("utf-8") + b"\0"
    identifier = ctypes.c_uint()
    try:
        status = lib.espeak_Synth(data, len(data), 0, 1, 0, 1, ctypes.byref(identifier), None)
        lib.espeak_Synchronize()
        if status:
            raise RuntimeError(f"eSpeak synthesis failed: {status}")
        pcm = b"".join(chunks)
        if not pcm:
            raise RuntimeError("eSpeak produced no audio")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        return {"status": "generated", "backend": "espeak_ng", "path": str(output),
                "voice": voice, "sample_rate": sample_rate, "frames": len(pcm) // 2,
                "duration_seconds": len(pcm) / (2 * sample_rate), "synthesis": "formant"}
    finally:
        lib.espeak_Terminate()


if __name__ == "__main__":
    request = json.load(sys.stdin)
    print(json.dumps(synthesize(**request)))
