#!/bin/bash
# 使用koudai48 conda环境运行代码-证据映射CSV生成脚本
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
# 加载.env中的环境变量
source .env 2>/dev/null || true
export DEEPSEEK_API_KEY
# 使用koudai48环境的python运行
exec /home/mingg/miniconda3/envs/koudai48/bin/python generate_code_evidence_csv.py
