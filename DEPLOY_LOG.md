# GPT-SoVITS 本地部署 · 困难记录与易错点

> 适用场景：Ubuntu 22.04 纯 CPU（无 GPU）、无图形界面、且处于**受限网络**（DNS 污染 + 部分域名 SNI 层 TLS 拦截）的沙箱环境。
> 目标：用 GPT-SoVITS 把"你的声音"克隆出来做中文配音。
> 结论：在纯 CPU 沙箱里**端到端跑通**了"文字 → 你的声音"推理（单句中文约 9–15 秒）。

---

## 〇、成果速览

| 项 | 状态 |
|----|------|
| 代码仓库 | GPT-SoVITS 官方仓库（fork/clone）+ 本仓库的部署脚本 |
| 依赖安装 | ✅ gradio / transformers / funasr / 中文 G2P 依赖等 |
| 模型权重 | ✅ v2 底模 + 中文 BERT + CNHuBERT + G2PW（约 2.3G，经 `download_models.sh` 下载） |
| 推理链路 | ✅ 加载→参考音频→语言检测→文本分段→BERT→中文G2P→语义预测→SoVITS合成→写出 wav |
| 声音克隆 | ✅ 零样本参考音频克隆（需你后续上传干音） |
| 模型微调 | ⏳ 未做（仅零样本）；若需"专属模型"另跑官方微调脚本 |

---

## 一、遇到的困难与解决

### 1. DNS 污染（最致命，拦在第一步）

- **现象**：`getent hosts github.com` → `198.18.0.24`；`huggingface.co` → `198.18.0.28`。这些是 RFC 2544 的**测试黑洞地址**，curl 全部超时/连接拒绝。
- **诊断**：系统 DNS 是 `183.60.83.19 / 82.98`，明文 UDP 53 被中间人篡改；改用**阿里 DoH（`223.5.5.5`，HTTPS 加密）**能拿到真实 IP。
- **解决**：把真实 IP 写死进 `/etc/hosts` 直连。
- **易错点**：
  - 只换 `nameserver` **没用**——明文 DNS 仍被篡改，必须用 DoH 取 IP 后再 hosts 直连。
  - `github.com`、`api.github.com`、`uploads.github.com` 真实 IP 不同，要分别加（`api.github.com → 20.205.243.168`）。

### 2. huggingface.co / pypi.org 原站 SNI 层 TLS 拦截

- **现象**：即便 IP 正确，`https://huggingface.co` 握手阶段被 RST（`SSL_ERROR_SYSCALL`）。这是 SNI 层拦截，**不是 DNS 问题**。
- **解决**：模型走 `hf-mirror.com`（HF 国内镜像），Python 包走阿里/清华 pypi 镜像，兜底用 `modelscope.cn`。
- **易错点**：改 hosts 救不了这类拦截，必须换**镜像域名**。判断方法：curl 能拿到 `200 OK` 说明通，拿到 `SSL_ERROR_SYSCALL` 就是 TLS 层被拦。

### 3. hf-mirror 的 LFS 大文件 302 跳转到真实 CDN 又被污染

- **现象**：小文件能下（hf-mirror 自有托管），但 `pytorch_model.bin` 等 LFS 大权重被 **302 跳转到 `cas-bridge.xethub.hf.co`**，该 CDN 同样被污染成黑洞地址 → 大文件全失败。
- **解决**：把 `cas-bridge.xethub.hf.co`、`cdn-lfs.huggingface.co`、`cdn-lfs.hf.co` 的真实 IP 也加进 `/etc/hosts`。
- **易错点**：下载脚本报错时，**先抓 HTTP 302 的 `Location`** 看是不是跳到了被污染的 CDN，再判断是 URL 错还是网络层问题。

### 4. 官方 install.sh / Dockerfile 不可用

- **现象**：`install.sh` 强依赖 conda，且含未替换的占位符 `XXXXRT`；`Dockerfile` 用私有 CUDA 基础镜像（`xxxxrt666/torch-base`）+ `runtime: nvidia`，沙箱无 GPU。
- **解决**：放弃官方一键脚本，改用 **venv + 精简中文依赖 + CPU 版 torch + 镜像下模型**。
- **易错点**：README 里的模型路径是**占位符**，真实 v2 底模在 **`lj1995/GPT-SoVITS`**（原作者账号），文件名是 `s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`，不是 README 写的名字。

### 5. G2PWModel 找不到真实下载源

- **现象**：hf-mirror 上 `GPT-SoVITS / RVC-Boss / lj1995` 仓库都没有 `G2PWModel`；PaddleSpeech 旧链接 404。
- **解决**：直接 grep 代码，在 `GPT_SoVITS/text/g2pw/onnx_api.py` 找到内置真实源：
  `https://www.modelscope.cn/models/kamiorinn/g2pw/resolve/master/G2PWModel_1.1.zip`（验证 200 OK）。
- **易错点**：**不要猜仓库名**，直接看代码里的下载 URL。

### 6. 默认依赖里的非中文编译坑

- **现象**：`requirements.txt` 含 `pyopenjtalk`(日)、`ToJyutping`(粤)、`g2pk2/ko_pron`(韩)，会触发 C 编译，纯中文用不到且容易失败。
- **解决**：只装中文配音所需依赖，跳过上述包。

### 7. torchaudio.load 在 torch 2.10 + CPU 下需要 torchcodec（CUDA 版，加载失败）

- **现象**：`torchcodec 0.16` 是针对 **CUDA 版 torch** 构建的，`.so` 缺 `libtorch_cuda.so`，纯 CPU 直接报错；系统 ffmpeg 版本也很新。
- **解决**：**不装 torchcodec**，monkeypatch `torchaudio.load` 改用 `soundfile` 读 wav（返回 float32 numpy）。
- **易错点**：别陷在 ffmpeg/torchcodec 版本地狱里，soundfile 兜底是更稳的路。

### 8. fast_langdetect 下载 lid.176.bin 原站被墙

- **现象**：语言检测要下载 917MB 的 `lid.176.bin`（源 `dl.fbaipublicfiles.com` 被墙）。
- **解决**：monkeypatch `fast_langdetect`，强制用其**自带离线 small 模型** `lid.176.ftz`（`model="lite"`，938KB，完全离线）。
- **易错点**：模型名是 `"lite"` / `"full"` / `"auto"`，**不是** `"small"`（会报 `Invalid model`）。

### 9. opencc 模块缺失

- **现象**：中文 G2P 前端缺 `opencc`（简繁转换）。
- **解决**：`pip install opencc-python-reimplemented`（纯 Python 实现，避免系统 lib 依赖）。

### 10. TTS.run() 是 generator，不能直接解包

- **现象**：`sr, audio = tts.run(...)` 报 "need 2 values / got 1"。
- **解决**：非流式模式下 `run()` 只 `yield` 一次，应写 `sr, audio = next(tts.run(...))`。
- **易错点**：加 `next()` 后**括号要配对**——`next(tts.run({...}))` 比原 `tts.run({...})` 多一个右括号，漏掉会 `SyntaxError`。

### 11. soundfile 写 int16 wav 报 "System error" / 目录不存在

- **现象**：模型返回 int16 音频，`sf.write` 在此环境报 `System error`；根因其实是**输出目录不存在**（被错误掩盖）。
- **解决**：用标准库 `wave` 模块直接写 int16 wav（100% 可靠），并先 `mkdir -p` 输出目录。

---

## 二、运行方式（关键，决定能否跑起来）

1. **工作目录必须是仓库根** `/workspace/gpt-sovits-voice`。
2. 必须把 `GPT_SoVITS/` 加入 `sys.path`：裸模块 `AR / BigVGAN / feature_extractor / module` 都在 `GPT_SoVITS/` 下，而配置 `configs/tts_infer.yaml` 路径相对仓库根。
3. 配置文件 `GPT_SoVITS/configs/tts_infer.yaml` 已改为 `device: cpu` / `is_half: false`（CPU 必需）。
4. 先跑 `bash download_models.sh` 下载权重（约 2.3G），再推理。
5. 推理入口：`from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config`，`TTS(configs).run(inputs)`，取结果用 `next(...)`。
6. 单句中文 CPU 推理约 9–15 秒（模型常驻后更快）。

---

## 三、剩余问题 / 风险

- **纯 CPU 推理慢**：长文本、长音频要耐心；GPU 环境会快几十倍。
- **未做训练微调**：目前是零样本参考音频克隆。若要"专属模型"（音色更稳、跨文本更一致），需要你提供 **1 分钟+ 干净干音** 再跑官方微调流程。
- **镜像域名可能变动**：若 `download_models.sh` 失败，优先复查 `/etc/hosts` 与 LFS CDN 的真实 IP。
- **G2PW 校验和**：`GPT_SoVITS/text/g2pw/polyphonic.md5` 已被更新为 modelscope 版 `G2PWModel_1.1` 的实际校验和，**不要回退**，否则加载器可能误判损坏并尝试重下（离线会失败）。

---

## 四、易错点速查（checklist）

1. `/etc/hosts` 是否包含 `api.github.com` 与 `cas-bridge.xethub.hf.co` 等 LFS CDN 真实 IP。
2. 工作目录是否为仓库根，且 `sys.path` 含 `GPT_SoVITS`。
3. 是否打了三处补丁：① `torchaudio.load` → soundfile；② `fast_langdetect` 强制 `lite`；③ `next(tts.run(...))`。
4. 输出目录是否 `mkdir -p` 存在。
5. 模型是否下全：`chinese-hubert-base` / `chinese-roberta-wwm-ext-large` / `gsv-v2final-pretrained`(3 件套) / `G2PWModel/g2pW.onnx`。
6. 大权重不要入库（GitHub 100MB/文件、2GB/仓库上限），用 `download_models.sh` 拉取。

---

## 五、文件清单（本仓库新增/修改）

| 文件 | 作用 |
|------|------|
| `download_models.sh` | 从 hf-mirror + modelscope 下载全部模型权重（相对路径、可复现） |
| `install_deps.sh` | 安装精简中文依赖（跳过日/韩/粤语编译坑） |
| `smoke_test.py` | 模型加载冒烟测试（验证依赖/路径/权重能载入） |
| `e2e_test.py` | 端到端自验（含全部补丁，产出有效 wav） |
| `app.py` / `app_min.py` | 推理服务（模型常驻；app_min 为最小状态页，已弃用网页方案） |
| `DEPLOY_LOG.md` | 本文档 |
| `GPT_SoVITS/configs/tts_infer.yaml` | 改为 `device: cpu` / `is_half: false` |
| `GPT_SoVITS/text/g2pw/polyphonic.md5` | 更新为 G2PWModel_1.1 实际校验和 |
