from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .qa import VideoKnowledgeQA


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
    args = build_parser().parse_args()
    qa = VideoKnowledgeQA(
        records_path=Path(args.records),
        subtitle_root=Path(args.subtitle_root),
        kb_dir=Path(args.kb_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    if args.command == "build":
        stat = qa.build_or_update()
        print(json.dumps(stat, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        out = qa.ask(
            question=args.question,
            vector_top_k=args.vector_top_k,
            bm25_top_k=args.bm25_top_k,
            context_window=args.context_window,
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()

