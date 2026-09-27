#!/usr/bin/env python3.11
# 最小部署状态/上传网页：验证"网页可推送"通路 + 接收用户声音样本
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

APP = FastAPI()
WS = Path("/workspace/gpt-sovits-voice")
MODEL_DIR = WS / "GPT_SoVITS" / "pretrained_models"
SAMPLE_DIR = Path("/workspace/voice_samples")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def _tail(p: Path, n: int = 4) -> str:
    try:
        lines = p.read_text(errors="ignore").strip().splitlines()
        return "\n".join(lines[-n:]) or "(暂无日志)"
    except Exception:
        return "(无日志)"


def status() -> dict:
    size = ""
    if MODEL_DIR.exists():
        try:
            size = subprocess.run(["du", "-sh", str(MODEL_DIR)],
                                  capture_output=True, text=True).stdout.strip()
        except Exception:
            size = ""
    files = []
    if MODEL_DIR.exists():
        files = sorted(str(p.relative_to(MODEL_DIR)) for p in MODEL_DIR.rglob("*")
                       if p.is_file())
    return {
        "model_size": size or "尚未开始",
        "model_files": files,
        "download_log": _tail(WS / "download_models.log"),
        "install_log": _tail(WS / "install_deps.log"),
        "samples": sorted(p.name for p in SAMPLE_DIR.glob("*") if p.is_file()),
    }


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GPT-SoVITS 声音克隆 · 部署状态</title>
<style>
  :root{--bg:#0f1115;--card:#181b22;--fg:#e6e6e6;--muted:#9aa0aa;--acc:#4f8cff;--ok:#3ecf8e;--warn:#f5a623}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,Segoe UI,Roboto,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg);padding:24px}
  .wrap{max-width:880px;margin:0 auto}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid #23262e;border-radius:12px;padding:16px 18px;margin-bottom:16px}
  .row{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:6px 0;border-bottom:1px dashed #262a33}
  .row:last-child{border-bottom:none}
  .k{color:var(--muted);font-size:13px}
  .v{font-size:13px;text-align:right}
  .ok{color:var(--ok)} .warn{color:var(--warn)}
  pre{background:#0b0d11;border-radius:8px;padding:10px;font-size:11px;color:#a9e;overflow:auto;max-height:140px;margin:8px 0 0}
  .drop{border:2px dashed #2d3340;border-radius:10px;padding:22px;text-align:center;color:var(--muted);cursor:pointer;transition:.15s}
  .drop:hover{border-color:var(--acc);color:var(--fg)}
  input[type=file]{display:none}
  button{background:var(--acc);color:#fff;border:none;border-radius:8px;padding:9px 16px;font-size:13px;cursor:pointer}
  button:disabled{opacity:.5;cursor:not-allowed}
  textarea{width:100%;height:90px;background:#0b0d11;color:var(--fg);border:1px solid #23262e;border-radius:8px;padding:10px;font-size:13px;resize:vertical}
  .pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;background:#1f2733;color:var(--muted);margin-left:6px}
</style>
</head>
<body>
<div class="wrap">
  <h1>🎙️ GPT-SoVITS 声音克隆 · 部署状态</h1>
  <div class="sub">本地开源 · 数据不出本机 · 用途：配音。模型正在后台下载，声音样本现在就能上传。</div>

  <div class="card">
    <div class="row"><span class="k">依赖安装</span><span class="v ok">✓ 已完成（gradio/transformers/funasr…）</span></div>
    <div class="row"><span class="k">模型已下载</span><span class="v" id="msize">__MODEL_SIZE__</span></div>
    <div class="row"><span class="k">已上传声音样本</span><span class="v" id="scount">__SAMPLE_COUNT__</span></div>
  </div>

  <div class="card">
    <div class="row"><span class="k">① 上传你的声音样本（干音）</span><span class="v warn">必填</span></div>
    <label class="drop" for="up">点击选择音频（WAV/MP3，建议 30 秒~1 分钟，安静无背景音）<input id="up" type="file" accept="audio/*"></label>
    <div id="uplog" style="font-size:12px;color:var(--muted);margin-top:8px"></div>
    <div id="slist" style="font-size:12px;color:var(--ok);margin-top:4px"></div>
  </div>

  <div class="card">
    <div class="row"><span class="k">② 待配音文本</span><span class="v warn">模型就绪后启用</span></div>
    <textarea placeholder="把要配音的文字写在这里（模型下完即可一键生成你的声音）" disabled></textarea>
    <div style="margin-top:10px"><button disabled>🚀 生成配音（待模型就绪）</button></div>
  </div>

  <div class="card">
    <div class="k" style="margin-bottom:4px">模型下载日志（实时）</div>
    <pre id="dlog">__DOWNLOAD_LOG__</pre>
    <div class="k" style="margin:10px 0 4px">依赖安装日志（尾部）</div>
    <pre id="ilog">__INSTALL_LOG__</pre>
  </div>
</div>
<script>
async function refresh(){
  try{
    const r=await fetch('/api/status');const s=await r.json();
    document.getElementById('msize').textContent=s.model_size||'尚未开始';
    document.getElementById('scount').textContent=(s.samples||[]).length+' 个';
    document.getElementById('dlog').textContent=s.download_log;
    document.getElementById('ilog').textContent=s.install_log;
    document.getElementById('slist').textContent=(s.samples||[]).map(n=>'• '+n).join('\n');
  }catch(e){}
}
document.getElementById('up').addEventListener('change',async e=>{
  const f=e.target.files[0];if(!f)return;
  const fd=new FormData();fd.append('file',f);
  document.getElementById('uplog').textContent='上传中… '+f.name;
  const r=await fetch('/upload',{method:'POST',body:fd});
  const j=await r.json();
  document.getElementById('uplog').textContent=j.ok?('✓ 已保存：'+j.saved):('✗ '+JSON.stringify(j));
  refresh();
});
setInterval(refresh,5000);refresh();
</script>
</body></html>"""


@APP.get("/", response_class=HTMLResponse)
def index():
    s = status()
    html = (HTML
            .replace("__MODEL_SIZE__", s["model_size"])
            .replace("__SAMPLE_COUNT__", f'{len(s["samples"])} 个')
            .replace("__DOWNLOAD_LOG__", s["download_log"])
            .replace("__INSTALL_LOG__", s["install_log"]))
    return HTMLResponse(html)


@APP.get("/api/status")
def api_status():
    return status()


@APP.post("/upload")
async def upload(file: UploadFile = File(...)):
    dest = SAMPLE_DIR / file.filename
    data = await file.read()
    dest.write_bytes(data)
    return JSONResponse({"ok": True, "saved": str(dest), "bytes": len(data)})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(APP, host="0.0.0.0", port=port, log_level="info")
