"""检索逻辑：查询改写、向量检索、BM25 检索、合并、上下文扩展。"""

from __future__ import annotations

import gc
import random
from typing import Any, Optional

from .indexer import BM25Index
from .models import Segment
from .name_normalizer import normalize_text, normalize_segments_text
from .parsers import collect_segments
from .qa_prompts import build_bm25_refinement_prompt, build_vector_refinement_prompt
from .qa_utils import call_llm_json


def get_bm25(self) -> BM25Index:
    """惰性获取 BM25 索引，仅在首次调用时构建。"""
    if self._bm25 is None:
        if self.logger:
            self.logger.info("BM25索引未就绪，首次构建（惰性加载）...")
        self._bm25 = BM25Index(self.store.segments)
        if self.logger:
            self.logger.info("BM25索引构建完成")
    return self._bm25


def generate_kb_description(self) -> str:
    """从片段样本中生成知识库描述。"""
    all_segments = list(self.store.segments.values())
    if not all_segments:
        return "未知数据库"
    total = len(all_segments)

    # ---- 统计信息 ----
    speech_count = sum(1 for seg in all_segments if seg.source_type == "speech")
    danmaku_count = total - speech_count
    live_ids = sorted({seg.live_id for seg in all_segments})
    video_count = len(live_ids)

    # 仅从 speech 片段提取主播名（danmaku 的 anchor_name 是弹幕发送者ID，不是主播）
    speech_segments = [seg for seg in all_segments if seg.source_type == "speech"]
    anchor_names = sorted({seg.anchor_name for seg in speech_segments if seg.anchor_name})
    streamer_name = anchor_names[0] if len(anchor_names) == 1 else (", ".join(anchor_names[:3]) + "等" if anchor_names else "未知")

    # ---- 构建完整条目示例 ----
    sample_segments = random.sample(all_segments, min(20, len(all_segments)))
    # 按时间排序
    sample_segments.sort(key=lambda s: (s.video_datetime or "", s.video_title or s.live_id, s.live_id, s.start_time))
    sample_lines = ["以下是数据库中的条目示例（按时间排序）："]
    for i, seg in enumerate(sample_segments, 1):
        text_clean = seg.text.replace("\n", " ").strip()[:100]
        sample_lines.append(
            f"  [{i}] 时间={seg.video_datetime} | "
            f"标题={seg.video_title} | "
            f"偏移={seg.hhmmss} | "
            f"类型={seg.source_label} | "
            f"用户={seg.anchor_name} | "
            f"内容={text_clean}"
        )
    sample_text = "\n".join(sample_lines)

    # ---- 构建统计说明 ----
    stats_text = (
        f"数据库规模与分布：\n"
        f"  - 总片段数：{total}\n"
        f"  - 主播讲话片段：{speech_count}（{speech_count/total*100:.1f}%）\n"
        f"  - 观众弹幕片段：{danmaku_count}（{danmaku_count/total*100:.1f}%）\n"
        f"  - 直播视频数：{video_count}\n"
        f"  - 涉及的参与者/主播（仅来自主播讲话数据）：{', '.join(anchor_names) if anchor_names else '未知'}"
    )

    prompt = [
        {
            "role": "user",
            "content": (
                "请根据下面的信息，用一句简洁的话（20~50字）描述这个数据库的内容。\n\n"
                f"{stats_text}\n\n"
                f"数据条目示例：\n"
                f"{sample_text}\n\n"
                "请输出JSON对象：{\"description\":\"...\"}\n"
                "仅输出JSON，不要额外文本。"
            ),
        }
    ]

    # 调试：打印完整的 prompt 供检查
    if self.logger:
        self.logger.debug("=== 知识库描述生成完整 Prompt ===")
        self.logger.debug(prompt[0]["content"])
        self.logger.debug("=== 知识库描述生成 Prompt 结束 ===")

    try:
        self._ensure_client()
        parsed, _ = call_llm_json(
            self.client, self.llm_model, prompt, "知识库描述生成", logger=self.logger
        )
        desc = (parsed.get("description") or "").strip()
        if desc:
            return desc
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"LLM生成知识库描述失败: {exc}")

    # Fallback: 基于已知信息构建简单描述
    name = streamer_name.replace("SNH48-", "") if streamer_name else "未知"
    return f"这是一个包含{name}的直播回放视频的转录文本（主播讲话+观众弹幕）的数据库，共{total}条记录。"


def load_kb_description(self) -> None:
    """从文件加载知识库描述（仅加载，不生成）。"""
    if self.kb_description_path.exists():
        self.kb_description = self.kb_description_path.read_text(encoding="utf-8").strip()
        if self.logger:
            self.logger.info(f"已加载知识库描述: {self.kb_description[:80]}...")
    else:
        self.kb_description = None
        if self.logger:
            self.logger.info("知识库描述文件不存在，将在 build 时生成")


def build_or_update(self) -> dict[str, int]:
    """构建或增量更新知识库。"""
    if self.logger:
        self.logger.info("开始构建或更新知识库")
        self.logger.info(f"记录文件路径: {self.records_path}")
        self.logger.info(f"字幕根目录: {self.subtitle_root}")
        self.logger.info(f"知识库目录: {self.kb_dir}")

    all_segments = list(collect_segments(self.records_path, self.subtitle_root))
    parsed_count = len(all_segments)
    if self.logger:
        self.logger.info(f"解析得到 {parsed_count} 个片段")

    # ── 数据层归一化：入库前对片段文本做同音异形词替换 ──
    normalize_segments_text(all_segments)
    if self.logger:
        self.logger.info("已完成片段文本的同音异形词归一化")

        # 动态统计不同类型的片段
        participant_types = {}
        for seg in all_segments:
            participant_types[seg.source_type] = participant_types.get(seg.source_type, 0) + 1
        self.logger.info(f"片段类型统计: {participant_types}")

        self.logger.info("=== 示例片段（随机挑选） ===")
        sample_segments = []
        if all_segments:
            sample_segments = random.sample(all_segments, min(3, len(all_segments)))
        for seg in sample_segments:
            self.logger.info(f"[{seg.source_label}] 完整信息:")
            self.logger.info(f"  segment_id: {seg.segment_id}")
            self.logger.info(f"  source_type: {seg.source_type}")
            self.logger.info(f"  file_path: {seg.file_path}")
            self.logger.info(f"  video_path: {seg.video_path}")
            self.logger.info(f"  video_title: {seg.video_title}")
            self.logger.info(f"  anchor_name: {seg.anchor_name}")
            self.logger.info(f"  live_id: {seg.live_id}")
            self.logger.info(f"  video_datetime: {seg.video_datetime}")
            self.logger.info(f"  start_time: {seg.start_time} → end_time: {seg.end_time}")
            self.logger.info(f"  video_offset: {seg.hhmmss}")
            self.logger.info(f"  text: {seg.text!r}")

        if not all_segments:
            self.logger.warning("未解析到任何片段，请检查字幕文件和记录格式是否正确")

    changed = self.store.upsert_many(all_segments)
    if self.logger:
        self.logger.info(f"检测到 {len(changed)} 个新或更新的片段")

    if changed:
        if self.logger:
            self.logger.info("开始更新向量索引（这可能需要几分钟，请耐心等待...）")
        # 释放解析结果的内存，让向量索引过程有更多可用内存
        del all_segments
        gc.collect()
        if self.logger:
            self.logger.info("已释放解析结果内存")
        self.vector.upsert(changed, logger=self.logger)
        if self.logger:
            self.logger.info("开始保存片段存储")
        self.store.save()
        if self.logger:
            self.logger.success("片段存储保存完成")
            self.logger.info("重建BM25索引")
        self._bm25 = BM25Index(self.store.segments)
        # 重建后回收一次
        gc.collect()
    else:
        if self.logger:
            self.logger.info("没有新的片段，无需更新索引")

    # 构建完成后重新生成知识库描述
    self.kb_description = generate_kb_description(self)
    if not self.kb_description_path.parent.exists():
        self.kb_description_path.parent.mkdir(parents=True, exist_ok=True)
    self.kb_description_path.write_text(self.kb_description, encoding="utf-8")
    if self.logger:
        self.logger.info(f"知识库描述已更新: {self.kb_description[:80]}...")

    stat = {
        "parsed_segments": parsed_count,
        "updated_segments": len(changed),
        "total_segments": len(self.store.segments),
    }
    if self.logger:
        self.logger.success(f"构建完成: {stat}")

    return stat


def retrieve(
    self,
    question: str,
    vector_top_k: int = 40,
    bm25_top_k: int = 40,
    context_window: int = 3,
    vector_score_threshold: float = 0.3,
    bm25_score_threshold: float = 15.0,
    max_base_segments: Optional[int] = None,
    max_expanded_segments: Optional[int] = None,
    vector_survey_top_k: int = 1000,
    bm25_survey_top_k: int = 1000,
) -> tuple[list[Segment], dict[str, Any]]:
    if self.logger:
        self.logger.info("[1/6] 开始向量查询改写，辅助检索语句更聚焦语义")
    vector_query, vector_refinement = refine_vector_query(self, question)
    if self.logger:
        self.logger.info(f"[1/6] 向量查询改写完成，使用查询：{vector_query}")

    # ── 调查阶段：以较大 top_k 检索，评估相关性较高的段的总数 ──
    if self.logger:
        self.logger.info(
            f"[2/6·调查] 从向量索引通过Survey检索评估相关片段总量，"
            f"survey_top_k={vector_survey_top_k}, score_threshold={vector_score_threshold}"
        )
    survey_vector_ids, survey_vector_scores = self.vector.retrieve(vector_query, top_k=vector_survey_top_k)
    survey_vector_filtered = [
        (sid, score) for sid, score in zip(survey_vector_ids, survey_vector_scores)
        if score >= vector_score_threshold
    ]
    survey_vector_total = len(survey_vector_filtered)
    if self.logger:
        self.logger.info(
            f"[2/6·调查] 向量Survey完成：{len(survey_vector_ids)}个候选中 "
            f"{survey_vector_total}个超过阈值，实际将使用前{vector_top_k}个"
        )

    # ── 实际使用阶段：只取前 vector_top_k 个用于后续处理 ──
    if self.logger:
        self.logger.info(f"[2/6·使用] 从Survey结果中取前{vector_top_k}个，score_threshold={vector_score_threshold}")
    vector_ids = survey_vector_ids[:vector_top_k] if vector_top_k < len(survey_vector_ids) else survey_vector_ids
    vector_scores = survey_vector_scores[:vector_top_k] if vector_top_k < len(survey_vector_scores) else survey_vector_scores
    vector_filtered = [(sid, score) for sid, score in zip(vector_ids, vector_scores) if score >= vector_score_threshold]

    # 构建向量检索超清单（所有过滤后的命中，用于存档 Debug）
    vector_hits_all = []
    for rank, (sid, score) in enumerate(sorted(vector_filtered, key=lambda x: x[1], reverse=True), start=1):
        seg = self.store.segments.get(sid)
        text_snippet = seg.text.replace("\n", " ").strip() if seg else "<missing segment>"
        vector_hits_all.append({
            "rank": rank,
            "segment_id": sid,
            "score": round(score, 6),
            "text_snippet": text_snippet[:120],
        })
        if rank <= 200 and self.logger:
            self.logger.debug(
                f"  {rank:03d}. {sid} score={score:.6f} text={text_snippet}"
            )
    if self.logger:
        self.logger.info(f"[2/6] 向量检索完成，得到 {len(vector_ids)} 个候选段，过滤后 {len(vector_filtered)} 个")

    if self.logger:
        self.logger.info("[3/6] 开始BM25查询改写，辅助检索语句更聚焦")
    bm25_query, bm25_refinement = refine_bm25_query(self, question)
    if self.logger:
        self.logger.info(f"[3/6] BM25查询改写完成，使用查询：{bm25_query}")

    # ── 调查阶段：以较大 top_k 检索，评估相关性较高的段的总数 ──
    if self.logger:
        self.logger.info(
            f"[4/6·调查] 从BM25索引通过Survey检索评估相关片段总量，"
            f"survey_top_k={bm25_survey_top_k}, score_threshold={bm25_score_threshold}"
        )
    survey_bm25_ids, survey_bm25_scores = get_bm25(self).retrieve(bm25_query, top_k=bm25_survey_top_k)
    survey_bm25_filtered = [
        (sid, score) for sid, score in zip(survey_bm25_ids, survey_bm25_scores)
        if score >= bm25_score_threshold
    ]
    survey_bm25_total = len(survey_bm25_filtered)
    if self.logger:
        self.logger.info(
            f"[4/6·调查] BM25 Survey完成：{len(survey_bm25_ids)}个候选中 "
            f"{survey_bm25_total}个超过阈值，实际将使用前{bm25_top_k}个"
        )

    # ── 实际使用阶段：只取前 bm25_top_k 个用于后续处理 ──
    if self.logger:
        self.logger.info(f"[4/6·使用] 从Survey结果中取前{bm25_top_k}个，score_threshold={bm25_score_threshold}")
    bm25_ids = survey_bm25_ids[:bm25_top_k] if bm25_top_k < len(survey_bm25_ids) else survey_bm25_ids
    bm25_scores = survey_bm25_scores[:bm25_top_k] if bm25_top_k < len(survey_bm25_scores) else survey_bm25_scores
    bm25_filtered = [(sid, score) for sid, score in zip(bm25_ids, bm25_scores) if score >= bm25_score_threshold]

    # 构建 BM25 检索超清单（所有过滤后的命中，用于存档 Debug）
    bm25_hits_all = []
    for rank, (sid, score) in enumerate(sorted(bm25_filtered, key=lambda x: x[1], reverse=True), start=1):
        seg = self.store.segments.get(sid)
        text_snippet = seg.text.replace("\n", " ").strip() if seg else "<missing segment>"
        bm25_hits_all.append({
            "rank": rank,
            "segment_id": sid,
            "score": round(score, 6),
            "text_snippet": text_snippet[:120],
        })
        if rank <= 200 and self.logger:
            self.logger.debug(
                f"  {rank:03d}. {sid} score={score:.6f} text={text_snippet[:120]}"
            )
    if self.logger:
        self.logger.info(f"[4/6] BM25检索完成，得到 {len(bm25_ids)} 个候选段，过滤后 {len(bm25_filtered)} 个")

    if self.logger:
        self.logger.info(f"[5/6] 合并向量和BM25结果")

    bm25_max_score = max((score for _, score in bm25_filtered), default=0.0)
    merged_dict: dict[str, dict[str, float]] = {}

    for index, (sid, score) in enumerate(vector_filtered):
        merged_dict.setdefault(sid, {"vector_score": 0.0, "bm25_score": 0.0, "source": "vector"})
        merged_dict[sid]["vector_score"] = score
        merged_dict[sid]["vector_rank"] = index + 1

    for index, (sid, score) in enumerate(bm25_filtered):
        if sid not in merged_dict:
            merged_dict[sid] = {"vector_score": 0.0, "bm25_score": 0.0, "source": "bm25"}
        merged_dict[sid]["bm25_score"] = score
        merged_dict[sid]["bm25_rank"] = index + 1

    for sid, values in merged_dict.items():
        vector_norm = values.get("vector_score", 0.0)
        bm25_norm = values.get("bm25_score", 0.0)
        if bm25_max_score > 0:
            bm25_norm = bm25_norm / bm25_max_score
        # combined_score 仅用于 BM25-only 片段的排序，向量片段保证优先保留
        values["combined_score"] = vector_norm + bm25_norm * 0.3

    all_sorted = sorted(
        merged_dict.keys(),
        key=lambda x: merged_dict[x]["combined_score"],
        reverse=True,
    )
    raw_merged_count = len(all_sorted)
    if self.logger:
        self.logger.info(f"[5/6] 合并完成，共 {raw_merged_count} 个唯一段")

    # 关键策略：向量检索结果是语义匹配的，**全部保留**。
    vector_ids_set = {sid for sid, _ in vector_filtered}
    merged_ids: list[str] = []
    added: set[str] = set()

    # 第一步：全部向量片段无条件保留（按 combined_score 排序）
    for sid in all_sorted:
        if sid in vector_ids_set and sid not in added:
            merged_ids.append(sid)
            added.add(sid)

    # 第二步：补充 BM25 片段，但严格控制数量。
    bm25_supplement_limit = (
        80 if len(merged_ids) == 0          # 向量未找到：多补充一些 BM25
        else min(30, (max_base_segments or 200) - len(merged_ids))  # 向量已找到：最多补充 30 条
    )
    bm25_supplement_count = 0
    for sid in all_sorted:
        if bm25_supplement_count >= bm25_supplement_limit:
            break
        if sid not in added:
            merged_ids.append(sid)
            added.add(sid)
            bm25_supplement_count += 1

    if self.logger:
        self.logger.info(
            f"[5/6] 合并策略: 向量保留 {len(vector_filtered)} 个, "
            f"BM25补充 {bm25_supplement_count} 个, "
            f"总计 {len(merged_ids)} 个基础段"
        )

    if self.logger:
        self.logger.info(f"[6/6] 开始上下文扩展，context_window={context_window}")
    candidates = self.store.expand_context(merged_ids, context_window=context_window, logger=self.logger)
    if self.logger:
        self.logger.info(f"[6/6] 上下文扩展完成，得到 {len(candidates)} 个扩展后的片段")

    # max_expanded_segments 已弃用（不再截断），所有扩展片段全部保留
    merged_ids_set = set(merged_ids)

    # ── 计算回答全面性比率 ──
    survey_all_relevant_ids: set[str] = set()
    for sid, score in zip(survey_vector_ids, survey_vector_scores):
        if score >= vector_score_threshold:
            survey_all_relevant_ids.add(sid)
    for sid, score in zip(survey_bm25_ids, survey_bm25_scores):
        if score >= bm25_score_threshold:
            survey_all_relevant_ids.add(sid)
    survey_total_relevant = len(survey_all_relevant_ids)
    if survey_total_relevant > 0:
        comprehensiveness_ratio = len(merged_ids) / survey_total_relevant
    else:
        comprehensiveness_ratio = 1.0

    if self.logger:
        self.logger.info(
            f"[全面性评估] 向量Survey相关={survey_vector_total}, "
            f"BM25 Survey相关={survey_bm25_total}, "
            f"预估相关段总数={survey_total_relevant}, "
            f"实际使用基段数={len(merged_ids)}, "
            f"全面性比率={comprehensiveness_ratio:.2%}"
        )

    stats = {
        "vector_hits_raw": len(vector_ids),
        "vector_hits_filtered": len(vector_filtered),
        "vector_score_threshold": vector_score_threshold,
        "vector_query": vector_query,
        "vector_refinement": vector_refinement,
        "vector_hits_all": vector_hits_all,
        "bm25_query": bm25_query,
        "bm25_refinement": bm25_refinement,
        "bm25_hits_all": bm25_hits_all,
        "bm25_hits_raw": len(bm25_ids),
        "bm25_hits_filtered": len(bm25_filtered),
        "bm25_score_threshold": bm25_score_threshold,
        "raw_merged_ids": raw_merged_count,
        "used_base_ids": len(merged_ids),
        "merged_ids_set": list(merged_ids_set),
        "merged_dict_scores": {
            sid: {"vector_score": v["vector_score"], "bm25_score": v["bm25_score"], "source": v["source"]}
            for sid, v in merged_dict.items()
        },
        "candidate_count": len(candidates),
        "context_window": context_window,
        "max_base_segments": max_base_segments,
        "max_expanded_segments": max_expanded_segments,
        "truncated": False,
        # 全面性评估相关字段
        "comprehensiveness": {
            "survey_vector_total": survey_vector_total,
            "survey_bm25_total": survey_bm25_total,
            "survey_total_relevant": survey_total_relevant,
            "used_base_count": len(merged_ids),
            "ratio": round(comprehensiveness_ratio, 4),
            "bm25_query": bm25_query,
        },
    }

    return candidates, stats


def refine_vector_query(self, question: str) -> tuple[str, dict[str, Any]]:
    """向量查询改写。"""
    if self.logger:
        self.logger.debug(f"向量查询改写input: {question}")
    prompt_messages = build_vector_refinement_prompt(question, self.kb_description or "未知数据库")

    if self.logger:
        self.logger.debug("=== 向量查询改写完整 Prompt ===")
        self.logger.debug(prompt_messages[0]["content"])
        self.logger.debug("=== 向量查询改写 Prompt 结束 ===")

    try:
        parsed, llm_metadata = call_llm_json(
            self.client, self.llm_model, prompt_messages, "向量查询改写", logger=self.logger
        )
        refined = (parsed.get("refined_query") or "").strip()
        if not refined:
            raise ValueError("LLM未返回 refined_query")
        return refined, llm_metadata
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"向量查询改写失败，使用原始问题: {exc}")
        return question, {
            "description": "向量查询改写",
            "success": False,
            "error": str(exc),
            "refined_query": question,
            "prompt": prompt_messages[0]["content"],
            "response": "",
        }


def refine_bm25_query(self, question: str) -> tuple[str, dict[str, Any]]:
    """BM25 查询改写。"""
    if self.logger:
        self.logger.debug(f"BM25查询改写input: {question}")
    prompt_messages = build_bm25_refinement_prompt(question, self.kb_description or "未知数据库")

    if self.logger:
        self.logger.debug("=== BM25 查询改写完整 Prompt ===")
        self.logger.debug(prompt_messages[0]["content"])
        self.logger.debug("=== BM25 查询改写 Prompt 结束 ===")

    try:
        parsed, llm_metadata = call_llm_json(
            self.client, self.llm_model, prompt_messages, "BM25 查询改写", logger=self.logger
        )
        refined = (parsed.get("refined_query") or "").strip()
        if not refined:
            raise ValueError("LLM未返回 refined_query")
        return refined, llm_metadata
    except Exception as exc:
        if self.logger:
            self.logger.warning(f"BM25查询改写失败，使用原始问题: {exc}")
        return question, {
            "description": "BM25 查询改写",
            "success": False,
            "error": str(exc),
            "refined_query": question,
            "prompt": prompt_messages[0]["content"],
            "response": "",
        }
