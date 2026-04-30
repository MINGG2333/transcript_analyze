from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .qa import VideoKnowledgeQA


def setup_logger():
    """设置日志系统，使用loguru或回退到简单日志"""
    try:
        from loguru import logger
        # 移除默认的处理器，添加我们自己的格式
        logger.remove()
        # 添加控制台输出
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO"
        )
        # 添加文件输出，带轮转
        logger.add(
            "kb_qa.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="INFO",
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
    p.add_argument("--records", default="download_records.json", help="下载记录JSON路径")
    p.add_argument("--subtitle-root", default="firered_output_batch", help="字幕输出根目录")
    p.add_argument("--kb-dir", default="video_knowledge_db", help="知识库持久化目录")
    p.add_argument("--embedding-model", default="shibing624/text2vec-base-chinese", help="向量模型")
    p.add_argument("--llm-model", default="gpt-4o-mini", help="问答LLM模型名")
    p.add_argument("--api-base", default=os.getenv("OPENAI_BASE_URL"), help="LLM API base url")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="LLM API key")

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="构建或增量更新知识库")

    ask = sub.add_parser("ask", help="执行问答")
    ask.add_argument("--question", required=True, help="用户问题")
    ask.add_argument("--vector-top-k", type=int, default=40)
    ask.add_argument("--bm25-top-k", type=int, default=40)
    ask.add_argument("--context-window", type=int, default=3)
    return p


def main() -> None:
    logger = setup_logger()
    args = build_parser().parse_args()
    qa = VideoKnowledgeQA(
        records_path=Path(args.records),
        subtitle_root=Path(args.subtitle_root),
        kb_dir=Path(args.kb_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        api_base=args.api_base,
        api_key=args.api_key,
        logger=logger,
    )

    if args.command == "build":
        logger.info("开始执行build命令")
        stat = qa.build_or_update()
        print(json.dumps(stat, ensure_ascii=False, indent=2))
        logger.success("build命令执行完成")
        return

    if args.command == "ask":
        logger.info(f"开始执行ask命令: {args.question}")
        out = qa.ask(
            question=args.question,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            context_window=args.context_window,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        logger.success("ask命令执行完成")
        return


if __name__ == "__main__":
    main()

