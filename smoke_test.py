#!/usr/bin/env python3.11
# 模型加载冒烟测试：验证依赖/路径/权重能正常载入（不依赖用户声音样本）
import os, sys, time

ROOT = "/workspace/gpt-sovits-voice"
GPTS = os.path.join(ROOT, "GPT_SoVITS")

# 运行前置：cwd=仓库根，且把 GPT_SoVITS 加入 sys.path（裸模块 AR/BigVGAN/feature_extractor/module 在此）
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, GPTS)

PM = "GPT_SoVITS/pretrained_models"

def log(msg):
    print(msg, flush=True)

log(">>> [1/5] 导入 TTS 推理模块 ...")
t0 = time.time()
try:
    from TTS_infer_pack.TTS import TTS, TTS_Config
    import torch
    log(f"    导入成功 ({time.time()-t0:.1f}s) | torch={torch.__version__} cuda={torch.cuda.is_available()}")
except Exception as e:
    log(f"    [FAIL] 导入失败: {e}")
    raise

log(">>> [2/5] 构造 TTS_Config (v2 / cpu / fp32) ...")
configs = {
    "v2": {
        "device": "cpu",
        "is_half": False,
        "version": "v2",
        "t2s_weights_path": f"{PM}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "vits_weights_path": f"{PM}/gsv-v2final-pretrained/s2G2333k.pth",
        "cnhuhbert_base_path": f"{PM}/chinese-hubert-base",
        "bert_base_path": f"{PM}/chinese-roberta-wwm-ext-large",
    }
}
try:
    cfg = TTS_Config(configs)
    log(f"    配置解析成功 | version={cfg.version} device={cfg.device} is_half={cfg.is_half}")
    log(f"    t2s={cfg.t2s_weights_path}")
    log(f"    vits={cfg.vits_weights_path}")
except Exception as e:
    log(f"    [FAIL] 配置失败: {e}")
    raise

log(">>> [3/5] 构造 TTS 并加载所有权重（CPU，约 1-2 分钟）...")
t0 = time.time()
try:
    tts = TTS(cfg)
    log(f"    全部权重加载完成 ({time.time()-t0:.1f}s)")
except Exception as e:
    import traceback; traceback.print_exc()
    log(f"    [FAIL] 模型加载失败: {e}")
    raise

log(">>> [4/5] 校验各底座已就绪 ...")
checks = {
    "t2s_model (GPT 语义底模)": tts.t2s_model is not None,
    "vits_model (SoVITS 声码器)": tts.vits_model is not None,
    "bert_model (中文 BERT)": tts.bert_model is not None,
    "cnhuhbert_model (CNHuBERT)": tts.cnhuhbert_model is not None,
    "text_preprocessor": tts.text_preprocessor is not None,
}
allok = True
for k, v in checks.items():
    log(f"    [{'OK' if v else 'FAIL'}] {k}")
    allok = allok and v

log(">>> [5/5] 校验 G2PW 中文前端权重可加载（onnxruntime 直读）...")
g2pw_path = os.path.join(ROOT, PM, "G2PWModel/g2pW.onnx")
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(g2pw_path, providers=["CPUExecutionProvider"])
    log(f"    [OK] G2PW onnx 加载成功 | inputs={[i.name for i in sess.get_inputs()]}")
    g2pw_ok = True
except Exception as e:
    log(f"    [WARN] G2PW 加载失败（不影响底模，可在推理时按需排查）: {e}")
    g2pw_ok = False

log("")
log("=" * 48)
log("冒烟测试结论: " + ("✅ 全部通过 —— 模型链路可正常载入" if (allok and g2pw_ok) else
                         ("⚠️ 底模通过，G2PW 需关注" if allok else "❌ 存在失败")))
log("=" * 48)
