#!/usr/bin/env python3.11
# 端到端自验：用一段（合成的）参考音频驱动 TTS.run()，验证全链路能产出有效音频。
# 注意：合成音频仅用于验证代码路径，不代表任何真实音色。
import os, sys, time, numpy as np, soundfile as sf
ROOT = "/workspace/gpt-sovits-voice"; GPTS = os.path.join(ROOT, "GPT_SoVITS")
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, GPTS)

# ---- torchaudio.load 兼容补丁：torch 2.10 需 torchcodec 解码，但环境 ffmpeg 版本不匹配，
#      这里用 soundfile 直接读 wav/flac 兜底（torchaudio.load 在推理时才真正被调用） ----
import torchaudio as _ta
def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kw):
    import torch as _torch
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    wav = _torch.from_numpy(data.T.copy())  # (ch, n)
    if frame_offset or (num_frames not in (None, -1)):
        end = None if (num_frames in (None, -1)) else frame_offset + num_frames
        wav = wav[:, frame_offset:end]
    return wav, int(sr)
_ta.load = _sf_load

# ---- fast_langdetect 离线兜底：默认去下载 917MB 的 lid.176.bin（原站被墙），
#      强制改用包内自带的 lid.176.ftz small 模型，零外部依赖、语言切分够用 ----
import fast_langdetect as _fld
_orig_fld_detect = _fld.detect
def _fld_detect_small(text, model="lite", **kw):
    return _orig_fld_detect(text, model="lite", **kw)
_fld.detect = _fld_detect_small

# 1) 生成一段合成参考音频（基频+谐波+包络），仅用于触发 ref 处理流程
sr = 24000; dur = 3.0; t = np.linspace(0, dur, int(sr*dur), endpoint=False)
sig = (0.5*np.sin(2*np.pi*130*t) + 0.25*np.sin(2*np.pi*260*t) + 0.15*np.sin(2*np.pi*390*t))
sig *= np.hanning(len(sig)); sig = sig / np.max(np.abs(sig)) * 0.7
ref = "/workspace/voice_samples/_selftest_ref.wav"
sf.write(ref, sig.astype(np.float32), sr)
print("[self-test] 参考音频已生成:", ref, flush=True)

# 2) 加载模型
from TTS_infer_pack.TTS import TTS, TTS_Config
PM = "GPT_SoVITS/pretrained_models"
cfg = TTS_Config({"v2": {
    "device": "cpu", "is_half": False, "version": "v2",
    "t2s_weights_path": f"{PM}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "vits_weights_path": f"{PM}/gsv-v2final-pretrained/s2G2333k.pth",
    "cnhuhbert_base_path": f"{PM}/chinese-hubert-base",
    "bert_base_path": f"{PM}/chinese-roberta-wwm-ext-large",
}})
t0 = time.time(); tts = TTS(cfg)
print(f"[self-test] 模型加载完成 ({time.time()-t0:.1f}s)", flush=True)

# 3) 跑一次推理
t0 = time.time()
sr_out, audio = next(tts.run({
    "text": "你好，这是一段测试配音。",
    "text_lang": "zh", "ref_audio_path": ref,
    "prompt_text": "", "prompt_lang": "zh",
    "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0, "seed": -1,
}))
print(f"[self-test] 推理完成，耗时 {time.time()-t0:.1f}s", flush=True)
amp = float(np.max(np.abs(audio))) if hasattr(audio, "__len__") else 0
dur_out = (audio.shape[-1] if hasattr(audio, "shape") else len(audio)) / sr_out
print(f"[self-test] 输出 sr={sr_out} shape={getattr(audio,'shape',len(audio))} 时长={dur_out:.2f}s 最大振幅={amp:.3f}", flush=True)

out = "/workspace/voice_output/_selftest_out.wav"
_a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
print(f"[self-test] audio dtype={_a.dtype} shape={_a.shape}", flush=True)
if _a.dtype != np.int16:
    _a = np.clip((_a.astype(np.float32) * 32767), -32768, 32767).astype(np.int16)
if _a.ndim == 1:
    _a = _a.reshape(-1, 1)
import wave as _wave
with _wave.open(out, "wb") as w:
    w.setnchannels(_a.shape[1]); w.setsampwidth(2); w.setframerate(int(sr_out))
    w.writeframes(_a.tobytes())
print("[self-test] 已写出(wave):", out, flush=True)
print("[self-test] 已写出:", out, flush=True)
print("E2E_RESULT:", "PASS" if (audio is not None and dur_out > 0.1 and amp > 1e-3) else "FAIL", flush=True)
