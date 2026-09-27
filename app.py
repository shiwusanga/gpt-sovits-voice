#!/usr/bin/env python3.11
# GPT-SoVITS 配音应用：上传你的声音样本 + 输入文本 → 生成你的声音配音
# 模型常驻内存，避免每次推理重复加载。纯 CPU 推理，单句约 10-30 秒。
import os, sys, time, uuid, threading, traceback, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = "/workspace/gpt-sovits-voice"
GPTS = os.path.join(ROOT, "GPT_SoVITS")
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, GPTS)

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn

APP = FastAPI()
SAMPLE_DIR = Path("/workspace/voice_samples"); SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = Path("/workspace/voice_output"); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 兼容性补丁（已在端到端验证）----------
# 1) torchaudio.load 在 torch 2.10 需 torchcodec（本环境 ffmpeg 版本不兼容），
#    改用 soundfile 直接读 wav/flac。
import torchaudio as _ta
def _sf_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kw):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    wav = torch.from_numpy(data.T.copy())  # (ch, n)
    if frame_offset or (num_frames not in (None, -1)):
        end = None if (num_frames in (None, -1)) else frame_offset + num_frames
        wav = wav[:, frame_offset:end]
    return wav, int(sr)
_ta.load = _sf_load

# 2) fast_langdetect 默认去下载 917MB lid.176.bin（原站被墙），
#    强制改用包内自带的 lid.176.ftz "lite" 模型，离线、零外部依赖。
import torch
import fast_langdetect as _fld
_orig_fld = _fld.detect
def _fld_lite(text, model="lite", **kw):
    return _orig_fld(text, model="lite", **kw)
_fld.detect = _fld_lite

# ---------- 模型加载 ----------
TTS_MODEL = None
LOADING = True
LOAD_ERR = None
PM = "GPT_SoVITS/pretrained_models"

def load_model():
    global TTS_MODEL, LOADING, LOAD_ERR
    try:
        from TTS_infer_pack.TTS import TTS, TTS_Config
        cfg = TTS_Config({"v2": {
            "device": "cpu", "is_half": False, "version": "v2",
            "t2s_weights_path": f"{PM}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
            "vits_weights_path": f"{PM}/gsv-v2final-pretrained/s2G2333k.pth",
            "cnhuhbert_base_path": f"{PM}/chinese-hubert-base",
            "bert_base_path": f"{PM}/chinese-roberta-wwm-ext-large",
        }})
        TTS_MODEL = TTS(cfg)
        print("[app] 模型加载完成", flush=True)
    except Exception as e:
        LOAD_ERR = traceback.format_exc()
        print("[app] 模型加载失败:\n" + LOAD_ERR, flush=True)
    finally:
        LOADING = False

threading.Thread(target=load_model, daemon=True).start()
executor = ThreadPoolExecutor(max_workers=1)


def status_dict():
    return {
        "loading": LOADING, "ready": TTS_MODEL is not None, "error": LOAD_ERR,
        "samples": sorted(p.name for p in SAMPLE_DIR.glob("*") if p.is_file()),
        "outputs": sorted(p.name for p in OUTPUT_DIR.glob("*.wav") if p.is_file()),
    }


def save_wav(path, audio, sr):
    """audio: int16 np.ndarray (ch, n) 或 (n,)"""
    a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    if a.dtype != np.int16:
        a = np.clip((a.astype(np.float32) * 32767), -32768, 32767).astype(np.int16)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    import wave as _w
    with _w.open(str(path), "wb") as w:
        w.setnchannels(a.shape[1]); w.setsampwidth(2); w.setframerate(int(sr))
        w.writeframes(a.tobytes())


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPT-SoVITS 配音 · 你的声音</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--fg:#e6e6e6;--muted:#9aa0aa;--acc:#4f8cff;--ok:#3ecf8e;--warn:#f5a623}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg);padding:24px}
.wrap{max-width:880px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.card{background:var(--card);border:1px solid #23262e;border-radius:12px;padding:16px 18px;margin-bottom:16px}
.row{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:6px 0;border-bottom:1px dashed #262a33}
.row:last-child{border-bottom:none}.k{color:var(--muted);font-size:13px}.v{font-size:13px;text-align:right}
.ok{color:var(--ok)}.warn{color:var(--warn)}
.drop{border:2px dashed #2d3340;border-radius:10px;padding:18px;text-align:center;color:var(--muted);cursor:pointer;transition:.15s}
.drop:hover{border-color:var(--acc);color:var(--fg)}
input[type=file]{display:none}
button{background:var(--acc);color:#fff;border:none;border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer}
button:disabled{opacity:.5;cursor:not-allowed}
textarea,input[type=text]{width:100%;background:#0b0d11;color:var(--fg);border:1px solid #23262e;border-radius:8px;padding:10px;font-size:13px;resize:vertical}
.audio-wrap{margin-top:10px}
#status{font-size:12px;color:var(--muted)}
</style></head>
<body><div class="wrap">
<h1>🎙️ GPT-SoVITS 配音 · 用你的声音</h1>
<div class="sub">本地开源 · 数据不出本机 · 上传一段你的干音 + 输入文本 → 生成你的声音配音</div>

<div class="card"><div class="row"><span class="k">模型状态</span><span class="v" id="mstatus">加载中…</span></div>
<div class="row"><span class="k">已上传样本</span><span class="v" id="scount">0 个</span></div></div>

<div class="card">
<div class="row"><span class="k">① 上传你的声音样本（参考音色，必填）</span><span class="v warn">WAV/MP3 30s~1min 安静人声</span></div>
<label class="drop" for="up">点击选择参考音频<input id="up" type="file" accept="audio/*"></label>
<div id="uplog" style="font-size:12px;color:var(--muted);margin-top:8px"></div>
</div>

<div class="card">
<div class="row"><span class="k">② 待配音文本</span><span class="v warn">中文 / 自动</span></div>
<textarea id="text" rows="4" placeholder="把要配音成你声音的文字写在这里…"></textarea>
<div style="margin-top:8px"><label class="k">参考音频对应文字（可选，填了音色更准）：</label>
<input type="text" id="prompt_text" placeholder="例如参考音频里念的台词（可选）"></div>
<div style="margin-top:12px;display:flex;gap:10px;align-items:center">
<button id="gen" disabled>🚀 生成配音</button><span id="status"></span></div>
<div class="audio-wrap" id="audio-wrap"></div>
</div>

<div class="card"><div class="k" style="margin-bottom:4px">历史输出</div><div id="olist" style="font-size:12px;color:var(--ok)"></div></div>
</div>
<script>
const mstatus=document.getElementById('mstatus');
async function refresh(){
  try{
    const r=await fetch('/api/status');const s=await r.json();
    mstatus.textContent=s.ready?'✓ 就绪':(s.loading?'加载中…':'✗ 加载失败');
    mstatus.className='v '+(s.ready?'ok':(s.loading?'warn':''));
    document.getElementById('scount').textContent=(s.samples||[]).length+' 个';
    document.getElementById('gen').disabled=!s.ready;
    document.getElementById('olist').textContent=(s.outputs||[]).map(n=>'• '+n).join('\n');
    if(s.error) document.getElementById('status').textContent='模型错误: '+s.error.slice(0,200);
  }catch(e){}
}
document.getElementById('up').addEventListener('change',async e=>{
  const f=e.target.files[0];if(!f)return;
  const fd=new FormData();fd.append('file',f);
  document.getElementById('uplog').textContent='上传中… '+f.name;
  const r=await fetch('/upload',{method:'POST',body:fd});const j=await r.json();
  document.getElementById('uplog').textContent=j.ok?('✓ 已保存：'+j.saved):('✗ '+JSON.stringify(j));
  refresh();
});
document.getElementById('gen').addEventListener('click',async ()=>{
  const text=document.getElementById('text').value.trim();
  if(!text){alert('请先输入待配音文本');return;}
  const fd=new FormData();fd.append('text',text);
  fd.append('prompt_text',document.getElementById('prompt_text').value.trim());
  document.getElementById('status').textContent='生成中…（CPU 推理可能需要几十秒）';
  document.getElementById('gen').disabled=true;
  try{
    const r=await fetch('/api/tts',{method:'POST',body:fd});
    if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||('HTTP '+r.status));}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    document.getElementById('audio-wrap').innerHTML=
      '<audio controls src="'+url+'"></audio> <a href="'+url+'" download="voice.wav">⬇ 下载</a>';
    document.getElementById('status').textContent='✓ 生成完成';
  }catch(err){document.getElementById('status').textContent='✗ '+err.message;}
  document.getElementById('gen').disabled=false;refresh();
});
setInterval(refresh,4000);refresh();
</script></body></html>"""


@APP.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML)


@APP.get("/api/status")
def api_status():
    return status_dict()


@APP.post("/upload")
async def upload(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1].lower()
    raw = SAMPLE_DIR / f"ref_{uid}{ext or '.bin'}"
    raw.write_bytes(await file.read())
    # 统一转成 32k 单声道 wav，确保 soundfile 能读
    wav = SAMPLE_DIR / f"ref_{uid}.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-ar", "32000", "-ac", "1", str(wav)],
                      capture_output=True, check=True)
        if raw != wav:
            raw.unlink(missing_ok=True)
        return JSONResponse({"ok": True, "saved": str(wav), "name": wav.name})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"转码失败: {e}"}, status_code=400)


@APP.post("/api/tts")
async def tts(text: str = Form(...), prompt_text: str = Form(""), prompt_lang: str = Form("zh")):
    if TTS_MODEL is None:
        return JSONResponse({"ok": False, "error": "模型未就绪，请稍候或查看加载错误"}, status_code=503)
    samples = sorted(SAMPLE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not samples:
        return JSONResponse({"ok": False, "error": "请先上传你的声音样本（参考音频）"}, status_code=400)
    ref_path = str(samples[0])

    def run():
        sr_out, audio = next(TTS_MODEL.run({
            "text": text, "text_lang": "auto", "ref_audio_path": ref_path,
            "prompt_text": prompt_text, "prompt_lang": prompt_lang,
            "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0, "seed": -1,
        }))
        out = OUTPUT_DIR / f"out_{uuid.uuid4().hex}.wav"
        save_wav(out, audio, sr_out)
        return out.name
    try:
        name = executor.submit(run).result(timeout=600)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=500)
    return FileResponse(str(OUTPUT_DIR / name), media_type="audio/wav", filename=name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(APP, host="0.0.0.0", port=port, log_level="info")
