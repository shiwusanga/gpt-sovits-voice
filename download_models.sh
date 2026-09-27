#!/bin/bash
# GPT-SoVITS 模型下载脚本
# 适用：受限网络（已通过 /etc/hosts 直连 + hf-mirror/modelscope 镜像验证）
# 警告：模型权重体积大（约 2.3G），【不入库】，请克隆后用本脚本下载。
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/GPT_SoVITS/pretrained_models"
TEXT_DIR="$SCRIPT_DIR/GPT_SoVITS/text"
mkdir -p "$OUT/chinese-hubert-base" "$OUT/chinese-roberta-wwm-ext-large" "$OUT/gsv-v2final-pretrained"

dl() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo "[skip] $dest"; return; fi
  echo "[get ] $url"
  curl -L --retry 5 --retry-all-errors -C - -o "$dest" "$url" || { echo "FAILED: $url"; exit 1; }
}

BASE="https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main"

dl "$BASE/chinese-hubert-base/config.json"                       "$OUT/chinese-hubert-base/config.json"
dl "$BASE/chinese-hubert-base/preprocessor_config.json"          "$OUT/chinese-hubert-base/preprocessor_config.json"
dl "$BASE/chinese-hubert-base/pytorch_model.bin"                 "$OUT/chinese-hubert-base/pytorch_model.bin"
dl "$BASE/chinese-roberta-wwm-ext-large/config.json"            "$OUT/chinese-roberta-wwm-ext-large/config.json"
dl "$BASE/chinese-roberta-wwm-ext-large/pytorch_model.bin"       "$OUT/chinese-roberta-wwm-ext-large/pytorch_model.bin"
dl "$BASE/chinese-roberta-wwm-ext-large/tokenizer.json"         "$OUT/chinese-roberta-wwm-ext-large/tokenizer.json"
dl "$BASE/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" "$OUT/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
dl "$BASE/gsv-v2final-pretrained/s2D2333k.pth"                   "$OUT/gsv-v2final-pretrained/s2D2333k.pth"
dl "$BASE/gsv-v2final-pretrained/s2G2333k.pth"                   "$OUT/gsv-v2final-pretrained/s2G2333k.pth"

# G2PWModel（中文 G2P，modelscope 源，已验证 200 OK）
G2PW_ZIP="$OUT/G2PWModel_1.1.zip"
if [ ! -d "$OUT/G2PWModel" ]; then
  echo "[get ] G2PWModel_1.1.zip (modelscope)"
  curl -L --retry 5 -o "$G2PW_ZIP" "https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip"
  ( cd "$OUT" && (unzip -o -q G2PWModel_1.1.zip || true) )
  [ -d "$OUT/G2PWModel_1.1" ] && mv -f "$OUT/G2PWModel_1.1" "$OUT/G2PWModel"
fi

# 同时放到 text/ 下（代码运行时会从这里读取 g2pW.onnx）
if [ -d "$OUT/G2PWModel" ] && [ ! -e "$TEXT_DIR/G2PWModel/g2pW.onnx" ]; then
  mkdir -p "$TEXT_DIR/G2PWModel"
  cp -r "$OUT/G2PWModel/." "$TEXT_DIR/G2PWModel/"
  echo "[ok ] G2PWModel 已同步到 $TEXT_DIR/G2PWModel"
fi

echo "=== 模型下载完成 ==="
du -sh "$OUT" 2>/dev/null
find "$OUT" -type f | sort
