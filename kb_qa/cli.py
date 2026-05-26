from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import KB_QA_DEFAULTS
from .qa import VideoKnowledgeQA

# ── Script directory detection ──────────────────────────────────────────────
# 将默认数据文件/目录解析到 transcript_analyze/ 目录下
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_RECORDS = str(_SCRIPT_DIR / "download_records.json")
_DEFAULT_SUBTITLE_ROOT = str(_SCRIPT_DIR / "firered_output_batch")
_DEFAULT_KB_DIR = str(_SCRIPT_DIR / "video_knowledge_db")


def setup_logger(debug: bool = False):
    """设置日志系统，使用loguru或回退到简单日志"""
    level = "DEBUG" if debug else "INFO"
    try:
        from loguru import logger
        # 移除默认的处理器，添加我们自己的格式
        logger.remove()
        # 添加控制台输出
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level=level
        )
        # 添加文件输出，带轮转
        logger.add(
            "kb_qa.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level=level,
            rotation="10 MB",
            # retention="30 days"
        )
        return logger
    except ImportError:
        # 简单的日志类（回退方案）
        class SimpleLogger:
            def __init__(self):
                self.level_colors = {
                    "INFO": "\033[94m",      # 蓝色
                    "SUCCESS": "\033[92m",   # 绿色
                    "WARNING": "\033[93m",   # 黄色
                    "ERROR": "\033[91m",     # 红色
                    "RESET": "\033[0m"       # 重置
                }
            
            def _log(self, message, level="INFO"):
                color = self.level_colors.get(level, self.level_colors["RESET"])
                reset = self.level_colors["RESET"]
                print(f"{color}[{level}] {message}{reset}")
            
            def info(self, message):
                self._log(message, "INFO")
            
            def success(self, message):
                self._log(message, "SUCCESS")
            
            def warning(self, message):
                self._log(message, "WARNING")
            
            def error(self, message):
                self._log(message, "ERROR")
            
            def debug(self, message):
                # 简单日志类中debug和info一样
                self._log(message, "INFO")
            
            def critical(self, message):
                self._log(message, "ERROR")
        
        return SimpleLogger()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="直播视频知识库问答系统")
    p.add_argument("--records", default=_DEFAULT_RECORDS, help="下载记录JSON路径")
    p.add_argument("--subtitle-root", default=_DEFAULT_SUBTITLE_ROOT, help="字幕输出根目录")
    p.add_argument("--kb-dir", default=_DEFAULT_KB_DIR, help="知识库持久化目录")
    p.add_argument("--embedding-model", default="shibing624/text2vec-base-chinese", help="向量模型")
    p.add_argument("--llm-model", default="deepseek-v4-flash", help="问答LLM模型名")
    p.add_argument("--api-base", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), help="LLM API base url")
    p.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"), help="LLM API key")
    default_bg_dir = str(_SCRIPT_DIR / "docs" / "Background")
    p.add_argument("--background-knowledge-dir", default=default_bg_dir,
                    help="背景知识 markdown 文件目录（默认: docs/Background）")
    p.add_argument("--debug", action="store_true", help="开启调试日志，打印更多内部信息")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="构建或增量更新知识库")

    d = KB_QA_DEFAULTS
    ask = sub.add_parser("ask", help="执行问答")
    ask.add_argument("--question", required=True, help="用户问题")
    ask.add_argument("--vector-top-k", type=int, default=d.vector_top_k, help="向量检索的候选数（低内存服务器建议<1000）")
    ask.add_argument("--bm25-top-k", type=int, default=d.bm25_top_k, help="BM25检索的候选数（低内存服务器建议<1000）")
    ask.add_argument("--vector-score-threshold", type=float, default=d.vector_score_threshold, help="向量检索相关性阈值 [0-1]")
    ask.add_argument("--bm25-score-threshold", type=float, default=d.bm25_score_threshold, help="BM25检索相关性阈值")
    ask.add_argument("--context-window", type=int, default=d.context_window, help="上下文扩展窗口大小")
    ask.add_argument("--analysis-batch-size", type=int, default=d.analysis_batch_size, help="逐批分析候选片段时每批的最大数量")
    ask.add_argument("--synthesis-context-window", type=int, default=d.synthesis_context_window, help="合成阶段局部上下文窗口大小")
    ask.add_argument("--synthesis-batch-trigger-count", type=int, default=d.synthesis_batch_trigger_count, help="触发分批合成的有用段阈值")
    ask.add_argument("--synthesis-batch-size", type=int, default=d.synthesis_batch_size, help="分批合成时每批大小")
    return p


def _load_env_file() -> bool:
    """从多个候选位置依次尝试加载 .env 文件。

    查找顺序（优先级由高到低）：
      1. 当前工作目录 (os.getcwd())
      2. 脚本所在目录 (transcript_analyze/)
      3. 项目根目录 (snh48_web/)

    返回 True 表示至少成功加载了一个 .env 文件。
    """
    candidates = [
        Path.cwd() / ".env",
        _SCRIPT_DIR / ".env",
        _SCRIPT_DIR.parent / ".env",       # snh48_web/.env
    ]
    # 去重（如果多个路径指向同一个文件则只加载一次）
    seen: set[Path] = set()
    loaded = False
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            load_dotenv(resolved, override=False)
            loaded = True
    return loaded


def main() -> None:
    # ── 自动加载 .env 文件 ──────────────────────────────────────────────
    # 从当前工作目录 / transcript_analyze/ / snh48_web/ 依次查找 .env
    _load_env_file()

    args = build_parser().parse_args()
    logger = setup_logger(debug=args.debug)

    # ── API Key 安全检查 ──────────────────────────────────────────────
    final_api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not final_api_key:
        logger.error(
            "未配置 DeepSeek API Key！\n\n"
            "请通过以下任一方式配置：\n"
            f"  1. 复制 {_SCRIPT_DIR / '.env.example'} 为 {_SCRIPT_DIR / '.env'}，"
            "填入你的 DEEPSEEK_API_KEY\n"
            "  2. 设置环境变量: export DEEPSEEK_API_KEY=sk-xxx\n"
            "  3. 命令行参数: --api-key sk-xxx\n"
        )
        sys.exit(1)

    # ── 背景知识目录 ──
    bg_dir: Optional[Path] = None
    if args.background_knowledge_dir:
        bg_dir = Path(args.background_knowledge_dir)

    qa = VideoKnowledgeQA(
        records_path=Path(args.records),
        subtitle_root=Path(args.subtitle_root),
        kb_dir=Path(args.kb_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        api_base=args.api_base,
        api_key=final_api_key,
        logger=logger,
        background_knowledge_dir=bg_dir,
    )

    if args.command == "build":
        logger.info("开始执行build命令")
        stat = qa.build_or_update()
        # print(json.dumps(stat, ensure_ascii=False, indent=2))
        logger.success("build命令执行完成")
        return

    if args.command == "ask":
        logger.info(f"开始执行ask命令: {args.question}")
        out = qa.ask(
            question=args.question,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            context_window=args.context_window,
            vector_score_threshold=args.vector_score_threshold,
            bm25_score_threshold=args.bm25_score_threshold,
            analysis_batch_size=args.analysis_batch_size,
            synthesis_context_window=args.synthesis_context_window,
            synthesis_batch_trigger_count=args.synthesis_batch_trigger_count,
            synthesis_batch_size=args.synthesis_batch_size,
        )
        logger.success(f"ask命令执行完成，归档文件: {out.get('archive_path', 'N/A')}")
        return


if __name__ == "__main__":
    main()

