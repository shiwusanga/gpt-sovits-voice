#!/bin/bash
# GPT-SoVITS 依赖安装（精简中文版，跳过日/韩/粤编译坑，复用系统已装 torch 2.10 CPU）
set -e
python3.11 -m pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
python3.11 -m pip config set global.trusted-host mirrors.tencent.com

python3.11 -m pip install --no-input --upgrade pip wheel setuptools

# 注意：不重装 torch/torchaudio（系统已是 2.10.0+cu128，CPU 推理可用）
python3.11 -m pip install --no-input \
  "numpy<2.0" scipy tensorboard "librosa==0.10.2" numba \
  pytorch-lightning "gradio<5" ffmpeg-python tqdm funasr cn2an pypinyin g2p_en \
  modelscope sentencepiece "transformers>=4.51,<5" "peft<0.18.0" chardet PyYAML psutil \
  jieba_fast jieba split-lang "fast_langdetect>=0.3.1" wordsegment rotary_embedding_torch \
  fastapi "x_transformers" "torchmetrics<=1.5" "pydantic<=2.10.6" "ctranslate2>=4.0,<5" \
  "av>=11" huggingface_hub soundfile onnxruntime

echo "=== 依赖安装完成 ==="
python3.11 -c "import torch,librosa,gradio,transformers,funasr;print('torch',torch.__version__,'libs ok')"
