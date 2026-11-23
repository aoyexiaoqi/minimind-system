# InstaBox 阶段 A 操作手册（供浏览器操作员执行）

目标机器：agent-browser-hzy9（Debian 13，HOME=/data/home）。经仪表盘唤醒并打开终端操作。
注意：网页终端有吞空格怪癖，优先用桌面自带终端应用；命令块整段粘贴。
**红线：绝不触碰 3000 端口的 AIHOT 服务；不删除任何文件。**
令牌：由主代理通过安全渠道单独交给你（不在本手册内）。拿到后写到 InstaBox 的
/data/home/minimind-system/secrets.env，内容一行 `SERVICE_TOKEN=<令牌>`，并 chmod 600。

## 步骤 0：侦察（先做，结果记入回报）
```bash
ps -p 1 -o comm=
cat /proc/1/comm
ls /data/home/minimind
find /data/home/minimind -name "pretrain_768.pth" 2>/dev/null
python3 --version; which python3 pip3
df -h /data; free -m; nproc
ls /data/home/minimind/model
```
- PID 1 若不是 systemd → 用 PID 文件方式托管（步骤 5b）；若 supervisord 已装（`which supervisord`）可用 5a。
- 找到的 pretrain_768.pth 即 Phase B 检查点候选（600 步、loss 终点约 5.1380）。
  若 find 无结果：再 `find /data/home/minimind -name "*.pth" | head -20` 看清单，
  仍判断不了就停下回报，**不要擅自换权重来源**。

## 步骤 1：建目录与 venv
```bash
mkdir -p /data/home/minimind-system/{versions,logs,serving,deploy,eval,configs}
cd /data/home/minimind-system
python3 -m venv venv || python3 -m venv --without-pip venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r /dev/stdin <<'EOF'
fastapi
uvicorn
pydantic
safetensors
transformers
EOF
```
torch：若机器上已有可用的 torch 环境（复跑时装过，先查
`find /data/home -maxdepth 4 -name "torch" -type d 2>/dev/null | head`），
优先复用其版本并在回报中注明；否则装 CPU 版：
```bash
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 步骤 2：下载服务文件（公开链接，2026-11-25 07:42 UTC 前有效）
```bash
cd /data/home/minimind-system
curl -L -o serving/app.py "https://muse.ai/files/1358788220656773/2239542186907774/mqkhaw3olvfq6kbzue1867e0/app.py"
curl -L -o serving/requirements.txt "https://muse.ai/files/1358788220656773/1721982582683756/xdvj17r22ioy15ig8defa4nz/requirements.txt"
curl -L -o deploy/export_to_safetensors.py "https://muse.ai/files/1358788220656773/3530089037171151/xxo5o3slum7r2wdubg3o3cfh/export_to_safetensors.py"
curl -L -o deploy/validate_package.py "https://muse.ai/files/1358788220656773/1176380634900031/l5g6yh38dtpxc7lx95gsm5by/validate_package.py"
curl -L -o deploy/start_service.sh "https://muse.ai/files/1358788220656773/4832085083728021/425hh5omw6auht6eo5qrw3lf/start_service.sh"
curl -L -o deploy/stop_service.sh "https://muse.ai/files/1358788220656773/1480889897484072/fssxie5vh7r3whqzg35k1z19/stop_service.sh"
curl -L -o deploy/supervisord.conf "https://muse.ai/files/1358788220656773/1501435515423560/b35fqmgw9i46xj3r8vc7jzwq/supervisord.conf"
curl -L -o eval/prompts.jsonl "https://muse.ai/files/1358788220656773/1146094248349974/z1qcja5ooxf0g7ty6896oy9s/prompts.jsonl"
curl -L -o configs/model.json "https://muse.ai/files/1358788220656773/1730116962065392/b3tb4rf3wlkn0x31v0q3t2nj/model.json"
curl -L -o configs/generation.json "https://muse.ai/files/1358788220656773/2039083510830108/arx37qe8x0e4h5mjd872i0tm/generation.json"
chmod +x deploy/start_service.sh deploy/stop_service.sh
wc -c serving/app.py deploy/export_to_safetensors.py eval/prompts.jsonl
```
字节数应分别约为 5906 / 3514 / 9889，对不上就停下回报。

## 步骤 3：核验检查点并导出发布包
把 <CKPT> 换成步骤 0 找到的 pretrain_768.pth 实际路径：
```bash
cd /data/home/minimind-system
./venv/bin/python deploy/export_to_safetensors.py \
  --checkpoint <CKPT> \
  --repo /data/home/minimind \
  --out /data/home/minimind-system/versions/20261123-pretrain-run01 \
  --version 20261123-pretrain-run01 --step 600 --loss 5.1380 \
  --code-head 3d85c71e5cd86b8622107eb1f5745bdb4cc5590a
```
脚本会校验参数量在 60–68M 区间（期望约 63.91M），不合格会 FAIL 退出——
若 FAIL，停下并把输出原样回报，不要改参数硬闯。
然后离线冒烟：
```bash
./venv/bin/python deploy/validate_package.py \
  --package /data/home/minimind-system/versions/20261123-pretrain-run01 \
  --repo /data/home/minimind
ls -la /data/home/minimind-system/versions/20261123-pretrain-run01
```
应看到 PASS 与生成样例。再建 current 符号链接：
```bash
ln -sfn /data/home/minimind-system/versions/20261123-pretrain-run01 /data/home/minimind-system/current
ls -la /data/home/minimind-system/current
```

## 步骤 4：写入令牌
```bash
printf 'SERVICE_TOKEN=%s\n' '<主代理单独给你的令牌>' > /data/home/minimind-system/secrets.env
chmod 600 /data/home/minimind-system/secrets.env
```

## 步骤 5：启动服务（二选一）
5a 有 supervisord 时：
```bash
cd /data/home/minimind-system
set -a; . ./secrets.env; set +a
./venv/bin/supervisord -c deploy/supervisord.conf
./venv/bin/supervisorctl -c deploy/supervisord.conf status
```
（supervisord 需先 pip install supervisor 到 venv。）
5b PID 文件方式（默认）：
```bash
bash /data/home/minimind-system/deploy/start_service.sh
sleep 5; tail -30 /data/home/minimind-system/logs/service.log
```

## 步骤 6：本机自测（在 InstaBox 上）
```bash
curl -s http://127.0.0.1:8080/health/live
curl -s http://127.0.0.1:8080/health/ready
curl -s http://127.0.0.1:8080/v1/models
curl -s -X POST http://127.0.0.1:8080/v1/generate -H 'Content-Type: application/json' -d '{"prompt":"中国的首都是","max_new_tokens":32,"greedy":true}'
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://127.0.0.1:8080/v1/generate -H "Authorization: Bearer $(sed -n 's/^SERVICE_TOKEN=//p' /data/home/minimind-system/secrets.env)" -H 'Content-Type: application/json' -d '{"prompt":"中国的首都是","max_new_tokens":32,"greedy":true}'
```
最后一行应返回 200；不带令牌的 POST 应返回 401。
把以上输出原样回报给主代理，由主代理再从 VM 侧做外部验收（20 次生成等）。
