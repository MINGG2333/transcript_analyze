from __future__ import annotations

import gc
from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import Any, Optional
import uuid

from .indexer import BM25Index, SegmentStore, VectorIndex
from .models import Segment
from .name_normalizer import normalize_text, normalize_segments_text
from .parsers import collect_segments


class VideoKnowledgeQA:
    def __init__(
        self,
        records_path: Path,
        subtitle_root: Path,
        kb_dir: Path,
        embedding_model: str,
        llm_model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        logger=None,
    ):
        self.records_path = records_path
        self.subtitle_root = subtitle_root
        self.kb_dir = kb_dir
        self.llm_model = llm_model
        self.logger = logger

        self.store = SegmentStore(kb_dir / "segment_store.json")
        self.store.load()
        self.vector = VectorIndex(kb_dir / "chroma_db", embedding_model=embedding_model)
        # BM25 延迟加载：构建时仅在 upsert 后重建，问答时通过 get_bm25() 惰性获取
        self._bm25: Optional[BM25Index] = None
        self.api_base = api_base
        self.api_key = api_key
        self.client = None
        self.kb_description_path = kb_dir / "kb_description.txt"
        self.kb_description: Optional[str] = None
        self._load_kb_description()

    def get_bm25(self) -> BM25Index:
        """惰性获取 BM25 索引，仅在首次调用时构建。"""
        if self._bm25 is None:
            if self.logger:
                self.logger.info("BM25索引未就绪，首次构建（惰性加载）...")
            self._bm25 = BM25Index(self.store.segments)
            if self.logger:
                self.logger.info("BM25索引构建完成")
        return self._bm25

    def _generate_kb_description(self) -> str:
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
                    # "例如：'这是一个包含主播陈嘉仪的直播回放视频的转录文本（主播讲话+观众弹幕）的数据库，共30万+条记录。'\n"
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
            parsed, _ = self._call_llm_json(prompt, "知识库描述生成")
            desc = (parsed.get("description") or "").strip()
            if desc:
                return desc
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"LLM生成知识库描述失败: {exc}")

        # Fallback: 基于已知信息构建简单描述
        name = streamer_name.replace("SNH48-", "") if streamer_name else "未知"
        return f"这是一个包含{name}的直播回放视频的转录文本（主播讲话+观众弹幕）的数据库，共{total}条记录。"

    def _load_kb_description(self) -> None:
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
        # 确保向量索引和 BM25 索引基于规范名称，避免「陈佳怡」和「陈嘉仪」
        # 被当作不同实体处理。
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
        self.kb_description = self._generate_kb_description()
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
        vector_query, vector_refinement = self._refine_vector_query(question)
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
        bm25_query, bm25_refinement = self._refine_bm25_query(question)
        if self.logger:
            self.logger.info(f"[3/6] BM25查询改写完成，使用查询：{bm25_query}")

        # ── 调查阶段：以较大 top_k 检索，评估相关性较高的段的总数 ──
        if self.logger:
            self.logger.info(
                f"[4/6·调查] 从BM25索引通过Survey检索评估相关片段总量，"
                f"survey_top_k={bm25_survey_top_k}, score_threshold={bm25_score_threshold}"
            )
        survey_bm25_ids, survey_bm25_scores = self.get_bm25().retrieve(bm25_query, top_k=bm25_survey_top_k)
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
        # BM25 检索结果是词频匹配的（易被高频泛义词如主播名淹没），仅作为对向量结果的少量补充。
        # 这样保证语义相关的片段不会被 BM25 的噪声冲掉。
        vector_ids_set = {sid for sid, _ in vector_filtered}
        merged_ids: list[str] = []
        added: set[str] = set()

        # 第一步：全部向量片段无条件保留（按 combined_score 排序）
        for sid in all_sorted:
            if sid in vector_ids_set and sid not in added:
                merged_ids.append(sid)
                added.add(sid)

        # 第二步：补充 BM25 片段，但严格控制数量。
        # 向量已找到相关片段时，BM25 仅补充少量（避免高频泛义词如主播名淹没结果）。
        # 如果向量找到 0 条，则允许补充更多 BM25 片段兜底。
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
        # 对两种Survey方式中超过阈值的片段取并集，得到「预估相关段总数」
        # 注意：survey_vector_filtered / survey_bm25_filtered 的数据在后续合并中
        # 使用的是截断后的（取前vector_top_k/bm25_top_k），但这里的并集用的是
        # 完整的survey结果（各1000条），因此能更全面反映实际存在的相关片段总量
        survey_all_relevant_ids: set[str] = set()
        for sid, score in zip(survey_vector_ids, survey_vector_scores):
            if score >= vector_score_threshold:
                survey_all_relevant_ids.add(sid)
        for sid, score in zip(survey_bm25_ids, survey_bm25_scores):
            if score >= bm25_score_threshold:
                survey_all_relevant_ids.add(sid)
        survey_total_relevant = len(survey_all_relevant_ids)
        # 实际使用的唯一基础段数（合并后、上下文扩展前）
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

    def _build_judge_prompt(self, question: str, segments: list[Segment]) -> str:
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 用户名={s.anchor_name}; 内容={normalize_text(s.text)}"
            )
        context = "\n".join(lines)
        return (
            "你是严谨的证据型问答助手。请根据候选片段回答用户问题，不能臆造。\n"
            "注意区分不同来源的角色：\n"
            "- 「主播讲话」类型：内容是主播本人说的，但主播可能在说话中**引用/转述他人话语**，需要结合上下文分辨。\n"
            "- 「观众弹幕」类型：内容是观众/粉丝发的弹幕，不是主播说的。\n\n"
            "⚠️ 重要提示——如何区分主播「自述」与「转述/引用」：\n"
            "由于转录文本没有标点符号，同一段「主播讲话」中可能混合自述和转述。请结合**同一直播相邻片段的上下文**判断：\n"
            "  • 如果主播讲话片段**紧跟在一条观众弹幕之后**，且内容相似或直接回应了该弹幕，则可能是在**念/转述那条弹幕**，而非主播本人的观点。\n"
            "  • 如果主播讲话中出现\"有人说\"\"有弹幕说\"\"刚才有人说\"\"这条弹幕说\"\"有个朋友说\"\"xx说\"（如\"佳佳说\"）等标志词，则后面的内容是**转述**。\n"
            "  • 如果主播讲话中出现\"我念一下\"\"我读一下\"\"他/她说\"\"粉丝说\"等口吻，其后内容多为转述。\n"
            "  • 如果主播连续说出一段完整的个人观点/叙述（无转述标志词），则为**主播自己的话**。\n\n"
            f"用户问题：{normalize_text(question)}\n\n"
            "候选片段：\n"
            f"{context}\n\n"
            "请输出JSON对象，格式为：\n"
            '{"answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}\n'
            "要求：\n"
            "1) answer字段中如果列出事实或时间，请尽量用 [#1]、[#2] 这样的引用标记对应 evidence 条目；\n"
            "2) 在evidence的reason字段中，请注明该段内容是「主播自述」还是「主播转述」；\n"
            "3) evidence必须只使用给定segment_id；\n"
            "4) evidence列表应按时间顺序排列；\n"
            "5) 尽量覆盖所有相关证据；\n"
            "6) 如果证据不足，answer里明确说明不确定。\n"
            "仅输出JSON，不要额外文本。"
        )

    def _build_bm25_refinement_prompt(
        self,
        question: str,
        kb_description: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "你是一个专业的BM25检索关键词优化助手。你的任务是生成一组BM25查询关键词，"
                    "使得检索结果**精准覆盖到可能包含答案的片段**。\n\n"
                    f"数据库描述：{kb_description}\n\n"
                    "核心原则：\n"
                    "1) **排除高频无区分力的词**——主播名/嘉宾名在数据库中几乎每条都会出现，"
                    "对缩小搜索范围**毫无帮助**，不要放入关键词。\n"
                    "2) **优先用名词/名物词**，尽量避免通用动词——只有动词在数据库中属于低频率、"
                    "高区分度的专用场景时（如\"考研\"\"退团\"）才保留；"
                    "通用动词（如\"是\"\"有\"\"做\"\"说\"\"去\"\"来\"等）不会缩小搜索范围，不要放入。\n"
                    "3) **大胆联想，越多越好**——请沿着以下思路，充分发挥想象力：\n"
                    "   ① 同边形扩展：问题核心概念的同义词、近义词、口语/方言说法有哪些？\n"
                    "   ② 下位扩展：如果核心概念是一个抽象类别，它的具体实例/成员有哪些？（比如\"性别\"包含哪些具体值）\n"
                    "   ③ 场景扩展：与核心概念经常一起出现的人、事、物、地点、场景有哪些？\n"
                    "   ④ 取值扩展：如果问题是询问某个属性的值，把所有可能的取值都列出来。\n"
                    "   ⑤ 如果问题是针对某个具体非知名对象（如宠物的名字），则仅用通用词，不要猜具体名字。\n"
                    "   ⑥ 如果问题涉及具体地点，请联想可能的方言口音、特色食物、风俗文化等领域词。\n"
                    "4) 基于你训练数据中的可靠常识进行联想（如\"男/女\"是性别取值、\"四川/重庆/上海/广东\"是常见地域），"
                    "放心输出，不要自我审查。但不要编造你无法确认的**具体事实**（如特定宠物的具体名字）。\n"
                    "5) 最终输出**多词组合的查询字符串**（词之间用空格分隔），"
                    "词越多越好，覆盖越全面越好，不限制数量。\n\n"
                    f"用户问题：{question}\n"
                    "请输出JSON对象：{\"refined_query\":\"...\"}。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _build_vector_refinement_prompt(
        self,
        question: str,
        kb_description: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    "你是一个专业的向量检索查询优化助手。你的任务是将用户的自然语言问题改写为"
                    "更适合向量（语义）检索的查询文本。\n\n"
                    f"数据库描述：{kb_description}\n\n"
                    "核心原则：\n"
                    "1) **排除高频无区分力的词**——主播名/嘉宾名在数据库中几乎每条都会出现，"
                    "应弱化或移除，让向量搜索更关注区分度高的语义成分。\n"
                    "2) **保留核心语义，并用常识进行语义扩展**——根据问题类型，凭你的知识联想出"
                    "相关的概念/场景，融入改写后的句子中，让向量检索能匹配到更丰富的相关片段。\n"
                    "   例如：\n"
                    "   - 问\"性别\" → 改写成\"这个人的性别是什么，男的还是女的，是哥还是姐还是娘\"\n"
                    "   - 问\"祖籍/老家\" → 改写成\"是哪里人，老家在何处，曾在或不是来自某地\"\n"
                    "   - 问\"年龄\" → 改写成\"某人多大年纪，几岁，什么年龄\"\n"
                    "   - 问\"感情状况\" → 改写成\"恋爱关系，有男朋友或女朋友，单身，结婚\"\n"
                    "   - 问\"职业/学业\" → 改写成\"大学专业，工作公司，毕业什么学校，做什么的\"\n"
                    "   - 问\"宠物名字\" → **不要乱猜具体名字**，仅写\"宠物狗猫叫什么名字\"\n"
                    "3) **输出完整的语义句子**——不要只输出关键词，而是输出一个或多个完整的、"
                    "语义清晰的查询句子（多个句子用分号连接）。\n"
                    "4) **不要编造你无法确认的具体信息**——比如不确定宠物的名字、不确定去过什么城市，"
                    "就不要写入。宁可泛化，不要错写。\n\n"
                    f"用户问题：{question}\n"
                    "请输出JSON对象：{\"refined_query\":\"...\"}。\n"
                    "仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _refine_vector_query(self, question: str) -> tuple[str, dict[str, Any]]:
        if self.logger:
            self.logger.debug(f"向量查询改写input: {question}")
        prompt_messages = self._build_vector_refinement_prompt(question, self.kb_description or "未知数据库")

        # 调试：打印完整的 prompt 供检查
        if self.logger:
            self.logger.debug("=== 向量查询改写完整 Prompt ===")
            self.logger.debug(prompt_messages[0]["content"])
            self.logger.debug("=== 向量查询改写 Prompt 结束 ===")

        try:
            parsed, llm_metadata = self._call_llm_json(prompt_messages, "向量查询改写")
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

    def _refine_bm25_query(self, question: str) -> tuple[str, dict[str, Any]]:
        if self.logger:
            self.logger.debug(f"BM25查询改写input: {question}")
        prompt_messages = self._build_bm25_refinement_prompt(question, self.kb_description or "未知数据库")

        # 调试：打印完整的 prompt 供检查
        if self.logger:
            self.logger.debug("=== BM25 查询改写完整 Prompt ===")
            self.logger.debug(prompt_messages[0]["content"])
            self.logger.debug("=== BM25 查询改写 Prompt 结束 ===")

        try:
            parsed, llm_metadata = self._call_llm_json(prompt_messages, "BM25 查询改写")
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

    def _build_kb_background_text(self) -> str:
        """构建知识库背景说明，供分析/合成提示使用。"""
        if self.kb_description:
            return f"【数据库背景】{self.kb_description}"
        return ""

    def _format_segment_with_local_context(
        self,
        segment: Segment,
        context_window: int = 6,
    ) -> str:
        """格式化单个片段及其局部上下文，供合成阶段使用。"""
        key = segment.live_id
        seq = self.store.by_live_source.get(key, [])
        if not seq:
            return (
                f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
                f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
                f"核心片段内容：{normalize_text(segment.text)}"
            )

        try:
            pos = seq.index(segment.segment_id)
        except ValueError:
            return (
                f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
                f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
                f"核心片段内容：{normalize_text(segment.text)}"
            )

        start = max(0, pos - context_window)
        end = min(len(seq), pos + context_window + 1)
        local_lines: list[str] = []
        for idx in range(start, end):
            sid = seq[idx]
            local_seg = self.store.segments.get(sid)
            if not local_seg:
                continue
            marker = "核心片段" if sid == segment.segment_id else "上下文片段"
            local_lines.append(
                f"  - [{marker}] ({local_seg.hhmmss}) [{local_seg.source_label}] 用户名={local_seg.anchor_name}; {normalize_text(local_seg.text)}"
            )


        local_context = "\n".join(local_lines)
        return (
            f"[{segment.segment_id}] 类型={segment.source_label}; 直播时间={segment.video_datetime}; "
            f"视频内时间={segment.hhmmss}; 标题={segment.video_title}; 用户名={segment.anchor_name};\n"
            f"局部上下文（同一直播同一来源，窗口={context_window}）：\n{local_context}"
        )

    def _call_llm_json(
        self,
        messages: list[dict[str, str]],
        description: str,
        max_retries: int = 5,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """调用LLM API 并解析JSON响应，包含重试机制（指数退避）。"""
        last_raw = None

        for attempt in range(max_retries):
            try:
                if self.logger:
                    self.logger.info(f"调用LLM: {description} (尝试 {attempt+1}/{max_retries})")

                resp = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                last_raw = content

                llm_metadata = {
                    "model": self.llm_model,
                    "description": description,
                    "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prompt": messages[0]["content"] if messages and messages[0].get("role") == "user" else "",
                    "response": content,
                }

                if self.logger:
                    self.logger.info(f"  tokens: input={llm_metadata['input_tokens']}, output={llm_metadata['output_tokens']}, total={llm_metadata['total_tokens']}")

                try:
                    parsed = json.loads(content)
                    llm_metadata["success"] = True
                    return parsed, llm_metadata
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.warning(f"LLM返回JSON解析失败，尝试提取内容: {description}")
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        try:
                            parsed = json.loads(content[start : end + 1])
                            llm_metadata["success"] = True
                            llm_metadata["note"] = "通过字符串提取成功解析"
                            return parsed, llm_metadata
                        except json.JSONDecodeError:
                            pass
                    if attempt < max_retries - 1:
                        wait_time = 5 ** attempt
                        if self.logger:
                            self.logger.warning(f"JSON解析失败 (第 {attempt+1} 次)，等待 {wait_time} 秒后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        llm_metadata["success"] = False
                        llm_metadata["error"] = f"无法将LLM响应解析为JSON: {content}"
                        raise RuntimeError(llm_metadata["error"])

            except Exception as e:
                if self.logger:
                    self.logger.warning(f"LLM请求异常 (第 {attempt+1} 次): {e}")

                if attempt < max_retries - 1:
                    wait_time = 5 ** attempt
                    if self.logger:
                        self.logger.warning(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"LLM调用在 {max_retries} 次重试后仍失败"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _build_analysis_prompt(
        self,
        question: str,
        segments: list[Segment],
        batch_index: int,
        total_batches: int,
        kb_description: str = "",
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; 直播时间={s.video_datetime}; "
                f"视频内时间={s.hhmmss}; 标题={s.video_title}; 用户名={s.anchor_name}; 内容={normalize_text(s.text)}"
            )
        context = "\n".join(lines)

        bg_text = ""
        if kb_description:
            bg_text = f"【数据库背景】{kb_description}"

        return [
            {
                "role": "user",
                "content": (
                    "你是严谨的证据分析助手。注意区分不同来源的角色：\n"
                    "- 「主播讲话」类型：内容是主播本人说的，但主播可能在说话中**引用/转述他人话语**，需要结合上下文分辨。\n"
                    "- 「观众弹幕」类型：内容是观众/粉丝发的弹幕，不是主播说的。\n\n"
                    "⚠️ 重要提示——如何区分主播「自述」与「转述/引用」：\n"
                    "由于转录文本没有标点符号，请结合**同一直播相邻片段的上下文**判断：\n"
                    "  • 如果主播讲话中带有\"有人说\"\"有弹幕说\"\"刚才有人说\"\"xx说\"等标志词，后面内容可能是**转述**。\n"
                    "  • 如果主播讲话内容与附近的弹幕内容相似或直接回应了弹幕，则可能是在**念/转述弹幕**。\n"
                    "  • 请将你的判断写入每条有用片段的 reason 字段，注明「主播自述」或「主播转述」。\n\n"
                    f"{bg_text}\n\n"
                    "🧠 通用推理原则：\n"
                    "1) 请充分利用你训练数据中的**背景常识**（如公众人物的公开身份、团体的公开性质等已知信息），"
                    "结合片段上下文来判断主播话语的真实含义。\n"
                    "2) 如果主播说的话与你所知的常识存在**明显矛盾**（例如一个公开身份为女性的人说\"我是男生\"），"
                    "请结合上下文判断：这很可能是主播在**转述弹幕、回应观众、开玩笑或玩梗**，而非其真实自述。\n"
                    "3) 你给出的回答会被公开发布，请确保回答客观、负责任，避免对主播造成不当误导或负面形象。\n"
                    "4) 请在 reason 字段中写明你的判断依据，包括你使用了什么背景常识和上下文线索。\n"
                    "5) 即使是\"矛盾\"的内容，如果它有助于理解上下文，也可以标记为 useful。\n\n"
                    "🔑 重要：片段元数据（如「用户名」字段中标注的组织团体前缀格式）本身也是有效的证据信息。"
                    "对于询问身份的问题，如果用户名中明确标注了提问对象的身份信息，"
                    "则该片段可以直接用作回答该问题的证据，因为用户名本身就是对提问对象的身份说明。"
                    "即使片段的文字内容是日常闲聊，其元数据信息仍可用于回答问题。\n\n"
                    f"用户问题：{question}\n"
                    f"这是第 {batch_index}/{total_batches} 批候选片段，请逐条判断是否与问题相关。\n"
                    "请输出JSON对象，格式为：\n"
                    '{"useful":[{"segment_id":"...","reason":"..."}]}\n'
                    "仅将与问题直接相关且可用于回答问题的片段放入 useful。\n"
                    "不要输出多余文本。\n"
                    "候选片段：\n"
                    f"{context}" 
                ),
            }
        ]

    def _analyze_candidates(
        self,
        question: str,
        candidates: list[Segment],
        analysis_batch_size: int = 20,
        kb_description: str = "",
    ) -> tuple[list[dict[str, Any]], list[Segment], dict[str, Any], list[dict[str, Any]]]:
        analysis: list[dict[str, Any]] = []
        useful_segments: list[Segment] = []
        total_batches = max(1, (len(candidates) + analysis_batch_size - 1) // analysis_batch_size)
        useful_ids: set[str] = set()
        batch_stats: list[dict[str, Any]] = []
        llm_calls: list[dict[str, Any]] = []

        for batch_index in range(total_batches):
            start = batch_index * analysis_batch_size
            batch = candidates[start : start + analysis_batch_size]
            if self.logger:
                self.logger.info(
                    f"分析候选批次 {batch_index + 1}/{total_batches}，包含 {len(batch)} 个片段"
                )

            if batch_index == 0 and self.logger:
                self.logger.info("=== 批次1详细信息 ===")
                for i, seg in enumerate(batch[:min(3, len(batch))], 1):
                    self.logger.info(f"  片段{i}: [{seg.segment_id}] {seg.source_label} {seg.hhmmss} {seg.text[:80]}")
                if len(batch) > 3:
                    self.logger.info(f"  ... 还有 {len(batch) - 3} 个片段")

            try:
                prompt_messages = self._build_analysis_prompt(question, batch, batch_index + 1, total_batches, kb_description=kb_description)
                parsed, llm_metadata = self._call_llm_json(
                    prompt_messages,
                    f"候选分析 {batch_index + 1}/{total_batches}",
                )

                if batch_index == 0 and self.logger:
                    self.logger.info("=== 批次1 LLM Prompt ===")
                    self.logger.info(prompt_messages[0]["content"])
                    self.logger.info("=== 批次1 LLM Response ===")
                    self.logger.info(json.dumps(parsed, ensure_ascii=False))

                llm_calls.append(llm_metadata)
            except Exception as exc:
                if self.logger:
                    self.logger.error(f"分析候选批次失败，视为全部片段候选: {exc}")
                parsed = {"useful": []}
                llm_calls.append({
                    "description": f"候选分析 {batch_index + 1}/{total_batches}",
                    "success": False,
                    "error": str(exc),
                    "prompt": prompt_messages[0]["content"],
                    "response": "",
                })

            useful_items = parsed.get("useful", []) or []
            useful_batch_ids = {item.get("segment_id") for item in useful_items if item.get("segment_id")}
            useful_count = 0
            for seg in batch:
                item = next(
                    (item for item in useful_items if item.get("segment_id") == seg.segment_id),
                    None,
                )
                is_useful = item is not None
                reason = item.get("reason", "").strip() if item else ""
                analysis.append(
                    {
                        "segment_id": seg.segment_id,
                        "source_label": seg.source_label,
                        "video_title": seg.video_title,
                        "anchor_name": seg.anchor_name,
                        "video_offset": seg.hhmmss,
                        "absolute_time": seg.absolute_time,
                        "text": seg.text,
                        "useful": is_useful,
                        "reason": reason,
                    }
                )
                if is_useful and seg.segment_id not in useful_ids:
                    useful_ids.add(seg.segment_id)
                    useful_segments.append(seg)
                    useful_count += 1

            batch_stats.append(
                {
                    "batch_index": batch_index + 1,
                    "batch_size": len(batch),
                    "useful_count": useful_count,
                    "useful_ids": sorted(useful_batch_ids),
                }
            )
            if self.logger:
                self.logger.info(
                    f"批次 {batch_index + 1} 分析完成: useful={useful_count} / {len(batch)}"
                )

        summary = {
            "total_candidates": len(candidates),
            "useful_segment_count": len(useful_segments),
            "analysis_batches": batch_stats,
            "analysis_batch_size": analysis_batch_size,
        }
        return analysis, useful_segments, summary, llm_calls

    def _build_group_synthesis_prompt(
        self,
        question: str,
        segments: list[Segment],
        group_info: str = "",
        batch_label: str = "",
    ) -> list[dict[str, str]]:
        """构建分组合成的prompt。

        与旧的 `_build_synthesis_prompt` 不同：
        - 不添加每条片段的局部上下文（因为候选片段已经是上下文扩展后的完整序列）
        - 片段按时间顺序简单列出即可
        """
        lines: list[str] = []
        for s in segments:
            lines.append(
                f"[{s.segment_id}] 类型={s.source_label}; "
                f"视频内时间={s.hhmmss}; "
                f"内容={normalize_text(s.text)}"
            )
        context = "\n".join(lines)
        bg_text = self._build_kb_background_text()

        header_parts = []
        if group_info:
            header_parts.append(f"直播视频信息：{group_info}")
        if batch_label:
            header_parts.append(f"批次：{batch_label}")
        header = "\n".join(header_parts) + "\n\n" if header_parts else ""

        return [
            {
                "role": "user",
                "content": (
                    "你是一名熟知这名主播的粉丝，现在请根据下面提供的直播片段来回答问题。\n\n"
                    "注意区分不同来源的角色：\n"
                    "- 「主播讲话」类型：内容是主播本人说的，但主播可能在说话中**引用/转述他人话语**，需要结合上下文分辨。\n"
                    "- 「观众弹幕」类型：内容是观众/粉丝发的弹幕，不是主播说的。\n\n"
                    "⚠️ 重要提示——如何区分主播「自述」与「转述/引用」：\n"
                    "由于转录文本没有标点符号，请结合同一视频内相邻片段的上下文判断：\n"
                    "  • 如果主播讲话中带有\"有人说\"\"有弹幕说\"\"刚才有人说\"\"xx说\"\"我念一下\"等标志词，后面内容可能是**转述**。\n"
                    "  • 如果主播讲话内容与附近的弹幕内容相似或直接回应了弹幕，则可能是在**念/转述弹幕**。\n"
                    "  • 请在 evidence 的 reason 字段中注明每段内容是「主播自述」还是「主播转述」。\n\n"
                    f"{bg_text}\n\n"
                    "🧠 通用推理原则：\n"
                    "1) 请充分利用你训练数据中的**背景常识**（如公众人物的公开身份、团体的公开性质等已知信息），结合片段上下文来判断主播话语的真实含义。\n"
                    "2) 如果主播说的话与你所知的常识存在**明显矛盾**（例如一个公开身份为女性的人说\"我是男生\"），请结合上下文判断：这很可能是主播在**转述弹幕、回应观众、开玩笑或玩梗**，而非其真实自述。\n"
                    "3) 你给出的回答会被公开发布，请确保回答客观、负责任，避免对主播造成不当误导或负面形象。\n"
                    "4) 请在 evidence 的 reason 字段中写明你的判断依据，包括你使用了什么背景常识和上下文线索。\n"
                    "5) 即使是\"矛盾\"的内容，如果它有助于理解上下文，也可以保留为证据。\n\n"
                    "💬 回答风格要求：\n"
                    "请用自然、亲切的口吻回答，就像在跟朋友介绍一样。在提到片段的证据时，用 [#1]、[#2] 这样的标记引用对应的证据条目。\n"
                    "不需要在回答末尾列出所有引用的编号，只需要在回答中自然地插入引用标记即可。\n"
                    "例如：\"根据片段中的账号名 [#1]，陈嘉仪是SNH48的成员。\"\n\n"
                    "⚠️ 引用格式要求（重要）：\n"
                    "一个中括号内**只能有一个引用编号**，格式如 [#1][#5] 等。\n"
                    "禁止使用任何其他格式，包括但不限于：\n"
                    "  - ❌ [#N-#M]（不允许区间写法，需要逐一列出 [#N]...[#M]）\n"
                    "  - ❌ [#N, #M]（逗号分隔）\n"
                    "  - ❌ [#N, #M, #K]（多个逗号分隔）\n"
                    "  - ❌ [#N-M]（缺少#号）\n"
                    "请严格遵守，否则系统无法正确识别引用标记。\n\n"
                    f"{header}"
                    f"用户问题：{question}\n\n"
                    "片段列表（按视频内时间排序）：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"answer":"...","evidence":[{"segment_ids":["...","..."],"citation_type":"...","reason":"..."}]}\n'
                    "要求：\n"
                    "1) answer用自然的语言回答，在引用证据时插入 [#N] 标记；\n"
                    "2) evidence必须包含所有**为答案提供了独特信息**的片段，即使是**间接的、部分的线索**也要收录；\n"
                    "3) 每个evidence条目可以引用**一条或多条片段**（`segment_ids`数组），并给出该组作为一个整体的理由（`reason`）。"
                    "对于由多个片段的组合才完整表达的意思，请将它们归入同一条evidence。\n"
                    "4) `citation_type`字段填写该条evidence的类型，可选值：\n"
                    "   - 「主播讲话」：仅包含主播讲话片段\n"
                    "   - 「观众弹幕」：仅包含观众弹幕片段\n"
                    "   - 「互动对话」：同时包含主播讲话和观众弹幕（显示对话互动关系）\n"
                    "5) evidence列表按时间顺序排列；\n"
                    "6) **即使无法给出确定答案**：如果在片段中找到了**任何相关的间接线索**，也必须将这些片段放入evidence[]并说明其相关性；"
                    "只有真正**毫无关联**的片段集合才返回空的 evidence[]。\n"
                    "7) **仅输出JSON，不要额外文本**。"
                ),
            }
        ]

    def _build_batch_synthesis_prompt(
        self,
        question: str,
        segments: list[Segment],
        batch_index: int,
        total_batches: int,
        synthesis_context_window: int,
    ) -> list[dict[str, str]]:
        lines: list[str] = []
        for s in segments:
            lines.append(self._format_segment_with_local_context(s, context_window=synthesis_context_window))
        context = "\n".join(lines)
        bg_text = self._build_kb_background_text()
        return [
            {
                "role": "user",
                "content": (
                    "你是一名熟知这名主播的粉丝，请根据下面的片段及其局部上下文总结与问题相关的关键信息。\n"
                    "注意区分不同来源的角色：\n"
                    "- 「主播讲话」类型：内容是主播本人说的，但主播可能在说话中**引用/转述他人话语**，需要结合上下文分辨。\n"
                    "- 「观众弹幕」类型：内容是观众/粉丝发的弹幕，不是主播说的。\n\n"
                    "⚠️ 重要提示——如何区分主播「自述」与「转述/引用」：\n"
                    "由于转录文本没有标点符号，请结合**同一直播相邻片段的上下文**判断：\n"
                    "  • 如果主播讲话中带有\"有人说\"\"有弹幕说\"\"刚才有人说\"\"xx说\"\"我念一下\"等标志词，后面内容可能是**转述**。\n"
                    "  • 如果主播讲话内容与附近的弹幕内容相似或直接回应了弹幕，则可能是在**念/转述弹幕**。\n"
                    "  • 请在 key_segments 的 reason 字段中注明每段是「主播自述」还是「主播转述」。\n\n"
                    f"{bg_text}\n\n"
                    "🧠 通用推理原则：\n"
                    "1) 请充分利用你训练数据中的**背景常识**（如公众人物的公开身份、团体的公开性质等已知信息），"
                    "结合片段上下文来判断主播话语的真实含义。\n"
                    "2) 如果主播说的话与你所知的常识存在**明显矛盾**（例如一个公开身份为女性的人说\"我是男生\"），"
                    "请结合上下文判断：这很可能是主播在**转述弹幕、回应观众、开玩笑或玩梗**，而非其真实自述。\n"
                    "3) 你给出的回答会被公开发布，请确保回答客观、负责任，避免对主播造成不当误导或负面形象。\n"
                    "4) 请在 key_segments 的 reason 字段中写明你的判断依据，包括你使用了什么背景常识和上下文线索。\n"
                    "5) 即使是\"矛盾\"的内容，如果它有助于理解上下文，也可以保留为关键段。\n\n"
                    f"用户问题：{question}\n"
                    f"这是第 {batch_index}/{total_batches} 批片段。\n\n"
                    "片段列表（每条含核心片段和局部上下文）：\n"
                    f"{context}\n\n"
                    "请输出JSON对象，格式为：\n"
                    '{"summary":"...","key_segments":[{"segment_id":"...","reason":"..."}]}\n'
                    "要求：\n"
                    "1) summary中简洁地总结这批片段与问题的相关信息；\n"
                    "2) key_segments列出该批最关键的5-10个segment_id及其原因；\n"
                    "3) 仅输出JSON，不要额外文本。"
                ),
            }
        ]

    def _build_citations_from_evidence(
        self,
        evidence: list[dict[str, Any]],
        useful_segments: list[Segment],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        id_to_seg = {s.segment_id: s for s in useful_segments}
        normalized_evidence = list(evidence)
        citations: list[dict[str, Any]] = []

        for idx, item in enumerate(normalized_evidence, start=1):
            # 支持新格式 segment_ids 数组，以及向后兼容旧格式 segment_id 字符串
            sids: list[str] = item.get("segment_ids", []) or []
            if not sids:
                sid = item.get("segment_id")
                if sid:
                    sids = [sid]
            if not sids:
                continue

            segs = [id_to_seg.get(sid) for sid in sids if sid]
            segs = [s for s in segs if s is not None]
            if not segs:
                continue

            citation: dict[str, Any] = {
                "citation_id": f"#{idx}",
                "segments": [],
                "citation_type": item.get("citation_type", ""),
                "reason": item.get("reason", ""),
            }
            # 向后兼容：保留旧的 segment_id / quoted_text 等顶层字段
            first = segs[0]
            citation["segment_id"] = first.segment_id
            citation["source_type"] = first.source_label
            citation["quoted_text"] = normalize_text(first.text)
            citation["video_offset"] = first.hhmmss
            citation["absolute_time"] = first.absolute_time
            citation["source_file"] = first.file_path
            citation["video_path"] = first.video_path
            citation["video_title"] = first.video_title
            citation["anchor_name"] = first.anchor_name
            citation["live_id"] = first.live_id

            # 完整的多段信息
            for s in segs:
                citation["segments"].append(
                    {
                        "segment_id": s.segment_id,
                        "source_type": s.source_label,
                        "quoted_text": normalize_text(s.text),
                        "video_offset": s.hhmmss,
                        "absolute_time": s.absolute_time,
                        "source_file": s.file_path,
                        "video_path": s.video_path,
                        "video_title": s.video_title,
                        "anchor_name": s.anchor_name,
                        "live_id": s.live_id,
                    }
                )

            citations.append(citation)
        return citations, normalized_evidence

    def _synthesize_with_batches(
        self,
        question: str,
        useful_segments: list[Segment],
        batch_size: int = 200,
        synthesis_context_window: int = 6,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """用分组合成的方式处理大量有用段。返回 (answer_text, final_evidence, llm_metadata)"""
        total_segments = len(useful_segments)
        total_batches = max(1, (total_segments + batch_size - 1) // batch_size)

        if self.logger:
            self.logger.info(
                f"准备分批合成最终回答，共 {total_segments} 个有用片段，分 {total_batches} 批处理"
            )

        batch_summaries: list[dict[str, Any]] = []
        key_segment_ids: set[str] = set()
        all_llm_calls: list[dict[str, Any]] = []

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total_segments)
            batch = useful_segments[start:end]

            if self.logger:
                self.logger.info(f"处理第 {batch_idx + 1}/{total_batches} 批，包含 {len(batch)} 个片段")

            try:
                prompt = self._build_batch_synthesis_prompt(
                    question,
                    batch,
                    batch_idx + 1,
                    total_batches,
                    synthesis_context_window=synthesis_context_window,
                )
                parsed, llm_metadata = self._call_llm_json(
                    prompt,
                    f"批次合成 {batch_idx + 1}/{total_batches}",
                )
                summary = parsed.get("summary", "").strip()
                key_segs = parsed.get("key_segments", []) or []

                batch_summaries.append({
                    "batch_index": batch_idx + 1,
                    "summary": summary,
                    "segment_count": len(batch),
                })

                for seg_info in key_segs:
                    if seg_info.get("segment_id"):
                        key_segment_ids.add(seg_info["segment_id"])

                all_llm_calls.append(llm_metadata)

                if self.logger:
                    self.logger.info(
                        f"批次 {batch_idx + 1} 合成完成，共提取 {len(key_segs)} 个关键段"
                    )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"批次 {batch_idx + 1} 合成失败: {exc}，使用原始片段")
                batch_summaries.append({
                    "batch_index": batch_idx + 1,
                    "summary": f"该批包含 {len(batch)} 个相关片段，无法生成摘要",
                    "segment_count": len(batch),
                    "error": str(exc),
                })
                all_llm_calls.append({
                    "description": f"批次合成 {batch_idx + 1}/{total_batches}",
                    "success": False,
                    "error": str(exc),
                })
                for seg in batch:
                    key_segment_ids.add(seg.segment_id)

        if self.logger:
            self.logger.info(f"第一阶段合成完成，提取了 {len(key_segment_ids)} 个关键段，准备生成最终答案")

        key_segments = [seg for seg in useful_segments if seg.segment_id in key_segment_ids]

        final_evidence: list[dict[str, Any]] = []
        final_answer = ""
        final_llm_metadata = {}

        try:
            batch_summary_text = "\n".join([
                f"[第{s['batch_index']}批] {s['summary']}"
                for s in batch_summaries
            ])

            lines: list[str] = []
            for s in key_segments:
                lines.append(
                    self._format_segment_with_local_context(
                        s,
                        context_window=synthesis_context_window,
                    )
                )
            context = "\n".join(lines)

            bg_text = self._build_kb_background_text()
            final_prompt = [
                {
                    "role": "user",
                    "content": (
                        "你是一名熟知这名主播的粉丝，基于下面的批次摘要和关键片段，生成一个全面的最终答案。\n"
                        "注意区分不同来源的角色：\n"
                        "- 「主播讲话」类型：内容是主播本人说的，但主播可能在说话中**引用/转述他人话语**，需要结合上下文分辨。\n"
                        "- 「观众弹幕」类型：内容是观众/粉丝发的弹幕，不是主播说的。\n\n"
                        "⚠️ 重要提示——如何区分主播「自述」与「转述/引用」：\n"
                        "由于转录文本没有标点符号，请结合**同一直播相邻片段的上下文**判断：\n"
                        "  • 如果主播讲话中带有\"有人说\"\"有弹幕说\"\"刚才有人说\"\"xx说\"\"我念一下\"等标志词，后面内容可能是**转述**。\n"
                        "  • 如果主播讲话内容与附近的弹幕内容相似或直接回应了弹幕，则可能是在**念/转述弹幕**。\n"
                        "  • 请在 evidence 的 reason 字段中注明每段内容是「主播自述」还是「主播转述」。\n\n"
                        f"{bg_text}\n\n"
                        "🧠 通用推理原则：\n"
                        "1) 请充分利用你训练数据中的**背景常识**（如公众人物的公开身份、团体的公开性质等已知信息），"
                        "结合片段上下文来判断主播话语的真实含义。\n"
                        "2) 如果主播说的话与你所知的常识存在**明显矛盾**（例如一个公开身份为女性的人说\"我是男生\"），"
                        "请结合上下文判断：这很可能是主播在**转述弹幕、回应观众、开玩笑或玩梗**，而非其真实自述。\n"
                        "3) 你给出的回答会被公开发布，请确保回答客观、负责任，避免对主播造成不当误导或负面形象。\n"
                        "4) 请在 evidence 的 reason 字段中写明你的判断依据，包括你使用了什么背景常识和上下文线索。\n\n"
                        " 回答风格要求：\n"
                        "请用自然、亲切的口吻回答，就像在跟朋友介绍一样。在提到片段的证据时，用 [#1]、[#2] 这样的标记引用对应的证据条目。\n"
                        "不需要在回答末尾列出所有引用的编号，只需要在回答中自然地插入引用标记即可。\n\n"
                        f"用户问题：{question}\n\n"
                        "批次摘要：\n"
                        f"{batch_summary_text}\n\n"
                        "关键片段列表（每条含核心片段和局部上下文）：\n"
                        f"{context}\n\n"
                        "请输出JSON对象，格式为：\n"
                        '{"answer":"...","evidence":[{"segment_id":"...","reason":"..."}]}\n'
                        "要求：\n"
                        "1) answer必须是一个全面的、综合所有批次的答案，涵盖所有重要信息；\n"
                        "2) evidence应包含所有关键segment_id，按时间顺序排列；\n"
                        "3) 每个evidence条目说明该片段如何支持答案，并注明是「主播自述」还是「主播转述」；\n"
                        "4) 力求完整性和全面性；\n"
                        "5) 不遗漏任何关键片段。\n"
                        "仅输出JSON，不要额外文本。"
                    ),
                }
            ]

            parsed, final_llm_metadata = self._call_llm_json(
                final_prompt,
                "最终答案合成",
            )

            final_evidence = parsed.get("evidence", []) or []
            final_answer = parsed.get("answer", "").strip() or "模型未返回有效答案。"

            if self.logger:
                self.logger.info(f"最终答案生成成功，包含 {len(final_evidence)} 个引用")
        except Exception as exc:
            if self.logger:
                self.logger.error(f"最终答案合成失败: {exc}")
            final_answer = "模型在生成最终答案时发生错误。"
            final_llm_metadata = {"success": False, "error": str(exc)}
            for seg in key_segments:
                final_evidence.append({
                    "segment_id": seg.segment_id,
                    "reason": f"该片段包含相关信息"
                })

        return final_answer, final_evidence, {
            "batch_synthesis": {
                "total_segments": total_segments,
                "batch_size": batch_size,
                "synthesis_context_window": synthesis_context_window,
                "total_batches": total_batches,
                "batch_summaries": batch_summaries,
                "batch_llm_calls": all_llm_calls,
            },
            "final_synthesis": final_llm_metadata,
        }

    def ask(
        self,
        question: str,
        vector_top_k: int = 1000,
        bm25_top_k: int = 1000,
        context_window: int = 3,
        vector_score_threshold: float = 0.3,
        bm25_score_threshold: float = 15.0,
        analysis_batch_size: int = 20,
        synthesis_context_window: int = 6,
        synthesis_batch_trigger_count: int = 100,
        synthesis_batch_size: int = 50,
    ) -> dict[str, Any]:
        self._ensure_client()
        if self.logger:
            self.logger.info(
                f"开始检索: vector_top_k={vector_top_k}, bm25_top_k={bm25_top_k}, context_window={context_window}"
            )
            candidates, stats = self.retrieve(
                question,
                vector_top_k,
                bm25_top_k,
                context_window,
                vector_score_threshold=vector_score_threshold,
                bm25_score_threshold=bm25_score_threshold,
                max_base_segments=200,
                # max_expanded_segments 不设置（None），不截断，所有扩展片段全部保留
                max_expanded_segments=None,
            )
        if self.logger:
            self.logger.info(
                f"检索结果: vector_hits_raw={stats['vector_hits_raw']}, vector_hits_filtered={stats['vector_hits_filtered']}, "
                f"bm25_hits_raw={stats['bm25_hits_raw']}, bm25_hits_filtered={stats['bm25_hits_filtered']}, "
                f"unique_base_ids={stats['raw_merged_ids']}, used_base_ids={stats['used_base_ids']}, "
                f"candidate_count={stats['candidate_count']}, truncated={stats['truncated']}"
            )

        merged_ids_set = set(stats.get("merged_ids_set", []))
        retrieval_segments = [
            {
                "segment_id": seg.segment_id,
                "source_label": seg.source_label,
                "video_title": seg.video_title,
                "anchor_name": seg.anchor_name,
                "video_offset": seg.hhmmss,
                "absolute_time": seg.absolute_time,
                "text": seg.text,
                "source_type": "基段" if seg.segment_id in merged_ids_set else "上下文扩展",
                "vector_score": stats.get("merged_dict_scores", {}).get(seg.segment_id, {}).get("vector_score", None),
                "bm25_score": stats.get("merged_dict_scores", {}).get(seg.segment_id, {}).get("bm25_score", None),
            }
            for seg in candidates
        ]

        if not candidates:
            result = {
                "question": question,
                "answer": "未检索到可用片段，无法回答该问题。",
                "citations": [],
                "retrieved_count": 0,
                "retrieval": stats,
                "analysis_summary": {
                    "total_candidates": 0,
                    "useful_segment_count": 0,
                    "analysis_batches": [],
                    "analysis_batch_size": analysis_batch_size,
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            archive_data = {**result, "retrieval_segments": [], "analysis": [], "useful_segments": []}
            archive_path = self._archive(archive_data)
            result["archive_path"] = str(archive_path)
            return result

        if self.logger:
            self.logger.info(f"候选片段数: {len(candidates)}，跳过有用性分析，全部视为有用。")
        # [方案C] 跳过LLM分析有用性阶段，全部候选片段保留供合成阶段自行判断。
        # 这样避免了LLM分析的不稳定性（同样片段每次判断可能不同），
        # 让最终合成LLM自己决定哪些片段有价值。
        analysis = [
            {
                "segment_id": seg.segment_id,
                "source_label": seg.source_label,
                "video_title": seg.video_title,
                "anchor_name": seg.anchor_name,
                "video_offset": seg.hhmmss,
                "absolute_time": seg.absolute_time,
                "text": seg.text,
                "useful": True,
                "reason": "跳过分析阶段，全部保留",
            }
            for seg in candidates
        ]
        useful_segments = candidates  # 全部保留
        analysis_summary = {
            "total_candidates": len(candidates),
            "useful_segment_count": len(candidates),
            "analysis_batches": [],
            "analysis_batch_size": analysis_batch_size,
            "skipped": True,
        }
        analysis_llm_calls = []
        if self.logger:
            self.logger.info(f"跳过分析，全部 {len(candidates)} 个候选片段均进入合成阶段。")

        if not useful_segments:
            if self.logger:
                self.logger.info("未找到有用片段，快速返回结果。")
            result = {
                "question": question,
                "answer": "未找到与问题直接相关的片段，无法给出确定答案。",
                "citations": [],
                "retrieved_count": len(candidates),
                "retrieval": stats,
                "analysis_summary": analysis_summary,
                "useful_segment_count": 0,
                "video_results": [],
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            archive_data = {
                **result,
                "retrieval_segments": retrieval_segments,
                "analysis": analysis,
                "useful_segments": [],
                "llm_calls": {
                    "analysis_batches": analysis_llm_calls,
                    "synthesis": {"description": "skip_synthesis_no_useful_segments"},
                },
            }
            archive_path = self._archive(archive_data)
            result["archive_path"] = str(archive_path)
            return result

        # ---- 按直播视频分组合成 ----
        # 将候选片段按 live_id 分组，每组内按 start_time 排序
        video_groups: dict[str, list[Segment]] = {}
        for seg in useful_segments:
            video_groups.setdefault(seg.live_id, []).append(seg)
        for live_id in video_groups:
            video_groups[live_id].sort(key=lambda s: s.start_time)

        # 获取视频元信息
        video_meta: dict[str, dict[str, str]] = {}
        for seg in self.store.segments.values():
            if seg.live_id not in video_meta:
                video_meta[seg.live_id] = {
                    "live_id": seg.live_id,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_datetime": seg.video_datetime,
                }

        # 每个视频分批处理（每批最多500条），不使用局部上下文（候选片段已是完整序列）
        per_video_batch_size = 500
        total_videos = len(video_groups)
        if self.logger:
            self.logger.info(f"开始合成：共 {total_videos} 个视频，{len(useful_segments)} 个片段")
        all_evidence: list[dict[str, Any]] = []
        video_results: list[dict[str, Any]] = []
        synthesis_llm_calls: list[dict[str, Any]] = []

        for video_idx, (live_id, segs) in enumerate(sorted(video_groups.items(), key=lambda x: x[1][0].start_time if x[1] else 0), start=1):
            meta = video_meta.get(live_id, {})
            group_info = (
                f"标题={meta.get('video_title', '')}, "
                f"直播时间={meta.get('video_datetime', '')}, "
                f"主播={meta.get('anchor_name', '')}"
            )
            total = len(segs)
            n_batches = max(1, (total + per_video_batch_size - 1) // per_video_batch_size)

            if self.logger:
                self.logger.info(
                    f"[{video_idx}/{total_videos}] 处理直播 {live_id}（{meta.get('video_title', '')}），"
                    f"共 {total} 个片段，分 {n_batches} 批（每批最多 {per_video_batch_size} 条）"
                )

            video_evidence: list[dict[str, Any]] = []
            video_answer_parts: list[str] = []

            for batch_idx in range(n_batches):
                start = batch_idx * per_video_batch_size
                batch = segs[start:start + per_video_batch_size]
                batch_label = f"{batch_idx + 1}/{n_batches}" if n_batches > 1 else ""

                # Debug: 记录第一个视频中第一组的LLM输入输出
                is_first_group = (video_idx == 1 and batch_idx == 0)

                try:
                    prompt_messages = self._build_group_synthesis_prompt(
                        question,
                        batch,
                        group_info=group_info,
                        batch_label=batch_label,
                    )
                    if is_first_group and self.logger:
                        self.logger.debug("=== 分组处理首个 LLM 调用 - Prompt ===")
                        self.logger.debug(prompt_messages[0].get("content", ""))

                    parsed, llm_meta = self._call_llm_json(
                        prompt_messages,
                        f"直播 {live_id} 合成 {batch_idx + 1}/{n_batches}" if n_batches > 1 else f"直播 {live_id} 合成",
                    )

                    if is_first_group and self.logger:
                        self.logger.debug("=== 分组处理首个 LLM 调用 - Response ===")
                        self.logger.debug(json.dumps(parsed, ensure_ascii=False))
                    batch_evidence = parsed.get("evidence", []) or []
                    batch_answer = parsed.get("answer", "").strip()
                    synthesis_llm_calls.append(llm_meta)

                    video_evidence.extend(batch_evidence)
                    if batch_answer:
                        video_answer_parts.append(batch_answer)

                    if self.logger:
                        self.logger.info(
                            f"  批次 {batch_idx + 1}/{n_batches} 完成，"
                            f"evidence={len(batch_evidence)} 条"
                        )
                except Exception as exc:
                    if self.logger:
                        self.logger.warning(f"  批次 {batch_idx + 1}/{n_batches} 合成失败: {exc}")

            # 合并同一视频的多个批次的答案
            if len(video_answer_parts) == 1:
                video_answer = video_answer_parts[0]
            elif len(video_answer_parts) > 1:
                video_answer = "\n---\n".join(video_answer_parts)
            else:
                video_answer = ""

            video_citations, _ = self._build_citations_from_evidence(video_evidence, segs)
            video_results.append({
                **meta,
                "answer": video_answer,
                "citations": video_citations,
                "useful_segment_count": len(segs),
                "batch_count": n_batches,
            })
            all_evidence.extend(video_evidence)

            if self.logger:
                self.logger.info(
                    f"直播 {live_id} 合成完成: 共 {len(video_evidence)} 条 evidence"
                )

        # ---- 最终答案生成 ----
        # 方案：
        #   1. 无答案 → 提示未找到
        #   2. 一个分组有答案 → 直接用（citations 随之生成）
        #   3. 多个分组有答案 →
        #      a. 先算出全局最终 citations 表（按 segment_id 全局编号）
        #      b. 将每个分组答案中的旧 [#N] 映射为最终 [#N]
        #      c. 将完整答案（不截断）喂给 LLM 合并
        #      d. LLM只负责合并文本，引用序号已在输入层保证正确
        answer_videos = [vr for vr in video_results if vr["answer"]]
        if len(answer_videos) == 0:
            answer_text = "未找到与问题直接相关的片段，无法给出确定答案。"
            citations, final_evidence = [], []
        elif len(answer_videos) == 1:
            answer_text = answer_videos[0]["answer"]
            # 只用这个分组的 evidence 生成 citations
            video_evidence = all_evidence  # 只有一个分组会写入 all_evidence
            citations, final_evidence = self._build_citations_from_evidence(
                [ev for vr in answer_videos for ev in (vr.get("citations", []) or [])],
                useful_segments,
            )
        else:
            # 多个分组有答案 → 先计算出全局最终 citations
            if self.logger:
                self.logger.info(f"开始最终汇总合并，共 {len(answer_videos)} 个视频分组有答案")

            # Step 1: 从所有分组的 evidence 生成最终 citation 表，建立映射
            citations, final_evidence = self._build_citations_from_evidence(
                all_evidence, useful_segments
            )
            # 多段引用：每个 citation 可能包含多个 segment，需要建立 segment_id → citation_id 映射
            seg_to_final_citation: dict[str, str] = {}
            for c in citations:
                c_id = c["citation_id"]
                segments_list = c.get("segments", []) or []
                for seg_info in segments_list:
                    seg_to_final_citation[seg_info["segment_id"]] = c_id
                # 也保留旧的 segment_id 顶层字段映射（向后兼容）
                if c.get("segment_id"):
                    seg_to_final_citation.setdefault(c["segment_id"], c_id)

            # Step 2: 为每个分组重写其答案中的引用编号
            video_summaries = []
            for idx, vr in enumerate(answer_videos, start=1):
                # 替换答案中的旧 citation_id 为最终编号
                raw_answer = vr["answer"]
                remapped_answer = raw_answer
                for old_c in vr.get("citations", []):
                    old_id = old_c["citation_id"]
                    # 尝试用旧 citation 引用的每个 segment_id 查找新的编号
                    new_id = None
                    segments_list = old_c.get("segments", []) or []
                    for seg_info in segments_list:
                        nid = seg_to_final_citation.get(seg_info["segment_id"])
                        if nid:
                            new_id = nid
                            break
                    if not new_id and old_c.get("segment_id"):
                        new_id = seg_to_final_citation.get(old_c["segment_id"])
                    if new_id and new_id != old_id:
                        remapped_answer = remapped_answer.replace(old_id, new_id)

                # 列出该分组实际引用的片段及其最终编号
                evidence_lines = []
                for c in vr.get("citations", []):
                    # 找到该旧 citation 对应的最终编号
                    final_id = None
                    segments_list = c.get("segments", []) or []
                    for seg_info in segments_list:
                        nid = seg_to_final_citation.get(seg_info["segment_id"])
                        if nid:
                            final_id = nid
                            break
                    if not final_id and c.get("segment_id"):
                        final_id = seg_to_final_citation.get(c["segment_id"], c["citation_id"])
                    text_snippet = c.get("quoted_text", "")
                    evidence_lines.append(f"  [{final_id}] (类型={c.get('citation_type', c.get('source_type', ''))}, 视频={c['video_title']}) {text_snippet}")
                evidence_text = "\n".join(evidence_lines) if evidence_lines else "  无直接引用片段"

                video_summaries.append(
                    f"[视频 {idx}] 标题={vr.get('video_title', '')}, "
                    f"直播时间={vr.get('video_datetime', '')}\n"
                    f"  答案：\n{remapped_answer}\n"
                    f"  引用片段：\n{evidence_text}"
                )
            summaries_text = "\n\n".join(video_summaries)

            # Step 3: 调用 LLM 合并（引用序号已在输入中正确）
            merge_prompt = [
                {
                    "role": "user",
                    "content": (
                        "你是一名熟知这名主播的粉丝，现在需要将多个视频分组对同一个问题的分析结果"
                        "合并成一个连贯、完整的答案。\n\n"
                        "每个视频分组的分析结果包含该组的初步答案和引用的片段（引用编号已经是全局统一的最终编号）。\n"
                        "你需要在保持信息完整性的前提下，去除重复内容，组织成一个结构清晰的统一答案。\n"
                        "不要在答案中包含组号和分组的标记，而是自然地整合信息。\n"
                        "引用编号 [#N] 已在输入中使用最终编号，输出时继续使用这些编号即可。\n\n"
                        f"问题：{question}\n\n"
                        "各视频分组分析结果：\n"
                        f"{summaries_text}\n\n"
                        "请输出JSON对象，格式为：\n"
                        '{"answer":"..."}\n'
                        "要求：\n"
                        "1) answer 是一个连贯的、完整的最终答案，不要有分组标记；\n"
                        "2) 引用编号已在输入中给出，直接沿用即可（无需创建新的引用）；\n"
                        "3) 如果某些视频分组的答案只是重复背景常识（如\"SNH48是女子团体\"），"
                        "可以合并成一句，不要每个分组都单独说一遍；\n"
                        "4) 引用精简：仅当不同引用的论证角度高度相似时才去重保留最具代表性的几个即可；"
                        "但如果某个引用提供了独特的信息角度或表达方式，即使都为同一结论服务，也应当保留——独特的表达本身就是亮点。\n"
                        "5) 以自然、亲切的口吻回答，就像在跟朋友介绍一样。\n"
                        "6) 回答应客观、负责任，避免对主播造成不当误导或负面形象。\n"
                        "7) 引用格式必须严格遵守：一个中括号内只能有一个引用编号，格式如 [#1]、[#5]。"
                        "禁止使用 [#N-#M]（区间）、[#N, #M]（逗号）、[#N-M]（缺少#号）等非标准格式。\n"
                        "仅输出JSON，不要额外文本。"
                    ),
                }
            ]
            try:
                if self.logger:
                    self.logger.debug("=== 最终答案合并 LLM 调用 - Prompt ===")
                    self.logger.debug(merge_prompt[0].get("content", ""))
                parsed, merge_llm_meta = self._call_llm_json(merge_prompt, "最终答案合并")
                if self.logger:
                    self.logger.debug("=== 最终答案合并 LLM 调用 - Response ===")
                    self.logger.debug(json.dumps(parsed, ensure_ascii=False))
                answer_text = (parsed.get("answer") or "").strip()
                if not answer_text:
                    raise ValueError("LLM返回空答案")
                synthesis_llm_calls.append(merge_llm_meta)
                if self.logger:
                    self.logger.info(f"最终答案合并完成，答案长度={len(answer_text)}")

                # 校验引用格式，不通过则让 LLM 修正
                invalid_refs = self._validate_answer_citations(answer_text)
                if invalid_refs:
                    correction_msg = (
                        "⚠️ 答案中的以下引用格式不符合要求，请修正：\n"
                        + "\n".join(f"  ❌ {r}" for r in invalid_refs)
                        + "\n\n引用格式必须为 [#N]（如 [#1]、[#5]），一个中括号内只能有一个引用编号。"
                        "禁止使用 [#N-#M]（区间）、[#N, #M]（逗号）等非标准格式。"
                    )
                    if self.logger:
                        self.logger.warning(f"最终答案合并引用格式不通过，重试: {invalid_refs}")
                    retry_prompt = list(merge_prompt)
                    retry_prompt.append({"role": "assistant", "content": json.dumps({"answer": answer_text})})
                    retry_prompt.append({"role": "user", "content": correction_msg})
                    parsed, merge_llm_meta = self._call_llm_json(retry_prompt, "最终答案合并（引用格式修正）")
                    if self.logger:
                        self.logger.debug("=== 最终答案合并 LLM 调用 - 修正后 Response ===")
                        self.logger.debug(json.dumps(parsed, ensure_ascii=False))
                    answer_text = (parsed.get("answer") or "").strip()
                    if not answer_text:
                        raise ValueError("LLM修正后返回空答案")
                    synthesis_llm_calls.append(merge_llm_meta)
                    if self.logger:
                        self.logger.info(f"最终答案合并（修正后）完成，答案长度={len(answer_text)}")
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"最终答案合并失败，回退到简单拼接: {exc}")
                answer_text = "\n\n---\n\n".join(vr["answer"] for vr in answer_videos)
                # 此时 citations 已由 Step 1 算出
        synthesis_llm_metadata = {
            "description": "per_video_batch_synthesis",
            "per_video_batch_size": per_video_batch_size,
            "video_count": len(video_results),
            "total_calls": len(synthesis_llm_calls),
            "calls": synthesis_llm_calls,
        }

        # ── 过滤 citations，只保留最终答案中通过 [#N] 实际引用的条目 ──
        # 避免 LLM 在合并多个视频分组答案时未使用全部引用，导致列表额外显示
        citations_before = len(citations)
        citations = self._filter_citations_by_answer(answer_text, citations)
        # 过滤后重新从 #1 编号，并同步更新答案中的引用标记
        answer_text, citations = self._renumber_citations(answer_text, citations)
        if self.logger:
            self.logger.info(f"引用过滤: 保留 {len(citations)}/{citations_before} 个 citations，已重编号")

        created_at = datetime.now().isoformat(timespec="seconds")
        result = {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "retrieval": stats,
            "analysis_summary": analysis_summary,
            "useful_segment_count": len(useful_segments),
            "video_results": video_results,
            "created_at": created_at,
        }
        # 按处理环节顺序构建存档数据
        archive_data = {
            # 0. 元信息
            "question": question,
            "created_at": created_at,
            # 1. 向量查询改写 → 向量检索 → BM25查询改写 → BM25检索 → 合并 → 上下文扩展
            "retrieval": stats,
            # 2. 扩展后的候选片段列表（每条标明了基段/扩展段以及检索分数）
            "retrieval_segments": retrieval_segments,
            # 3. 分析摘要（跳过逻辑下的说明）
            "analysis_summary": analysis_summary,
            # 4. 分析结果（全部标记为有用）
            "analysis": analysis,
            # 5. 有用片段列表
            "useful_segments": [
                {
                    "segment_id": seg.segment_id,
                    "source_label": seg.source_label,
                    "video_title": seg.video_title,
                    "anchor_name": seg.anchor_name,
                    "video_offset": seg.hhmmss,
                    "absolute_time": seg.absolute_time,
                    "text": seg.text,
                }
                for seg in useful_segments
            ],
            # 6. 按直播分组合成结果
            "video_results": video_results,
            # 7. 所有LLM调用的元数据
            "llm_calls": {
                "analysis_batches": analysis_llm_calls,
                "synthesis": synthesis_llm_metadata,
            },
            # 8. 最终答案
            "answer": answer_text,
            "citations": citations,
            "retrieved_count": len(candidates),
            "useful_segment_count": len(useful_segments),
        }
        archive_path = self._archive(archive_data)
        result["archive_path"] = str(archive_path)
        return result

    @staticmethod
    def _validate_answer_citations(answer: str) -> list[str]:
        """校验答案中的引用标记格式是否符合严格格式。

        只允许格式：[#N]（如 [#1]、[#5]），一个中括号内只能有一个引用编号。
        不允许 [#N-#M]（区间写法）、[#N, #M]（逗号）等非标准格式。

        返回：不符合格式的引用字符串列表，为空表示全部合法。
        """
        if not answer:
            return []
        import re
        all_brackets = list(re.finditer(r"\[[^\]]*\]", answer))
        allowed = re.compile(r"^\[\#\d+\]$")
        invalid: list[str] = []
        for m in all_brackets:
            text = m.group()
            if "#" in text and not allowed.match(text):
                invalid.append(text)
        return invalid

    @staticmethod
    def _filter_citations_by_answer(answer: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """过滤 citations 列表，只保留答案中通过 [#N] 或 [#N-#M] 实际引用的条目。"""
        if not answer or not citations:
            return [] if answer else citations

        import re
        referenced: set[int] = set()
        for m in re.finditer(r"\[#(\d+)(?:-#(\d+))?\]", answer):
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            referenced.update(range(start, end + 1))

        if not referenced:
            return citations  # 未检测到引用标记，全部保留（兜底）

        def _parse_citation_id(cid: str) -> int:
            try:
                return int(cid.lstrip("#"))
            except (ValueError, AttributeError):
                return -1

        return [c for c in citations if _parse_citation_id(c.get("citation_id", "")) in referenced]

    @staticmethod
    def _renumber_citations(
        answer: str, citations: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """过滤后重新从 #1 编号 citations，并更新 answer 中的引用标记。"""
        if not citations:
            return answer, citations

        import re

        # 构建旧编号 → 新编号的映射
        old_ids = []
        for c in citations:
            try:
                old_ids.append(int(c.get("citation_id", "").lstrip("#")))
            except (ValueError, AttributeError):
                old_ids.append(-1)

        sorted_old_ids = sorted(old_ids)
        id_mapping: dict[int, int] = {}
        for new_idx, old_id in enumerate(sorted_old_ids, start=1):
            id_mapping[old_id] = new_idx

        # 更新 answer 中的引用标记（从大到小替换以避免冲突）
        # 最终输出可以压缩连续区间，如 [#1][#2][#3] → [#1-#3]
        def replace_ref(m: re.Match) -> str:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else start
            new_nums = sorted(id_mapping.get(n, n) for n in range(start, end + 1))
            # 压缩连续区间
            ranges = []
            cur_start = cur_end = None
            for n in new_nums:
                if cur_start is None:
                    cur_start = cur_end = n
                elif n == cur_end + 1:
                    cur_end = n
                else:
                    ranges.append((cur_start, cur_end))
                    cur_start = cur_end = n
            if cur_start is not None:
                ranges.append((cur_start, cur_end))
            parts = []
            for rs, re_ in ranges:
                if rs == re_:
                    parts.append(f"[#{rs}]")
                else:
                    parts.append(f"[#{rs}-#{re_}]")
            return "".join(parts)

        answer = re.sub(r"\[#(\d+)(?:-#(\d+))?\]", replace_ref, answer)

        # 更新每个 citation 的 citation_id
        for c in citations:
            try:
                old_id = int(c.get("citation_id", "").lstrip("#"))
                c["citation_id"] = f"#{id_mapping.get(old_id, old_id)}"
            except (ValueError, AttributeError):
                pass

        # 按新编号排序
        def sort_key(c):
            try:
                return int(c.get("citation_id", "").lstrip("#"), 10)
            except (ValueError, AttributeError):
                return 0
        citations.sort(key=sort_key)

        return answer, citations

    def _archive(self, result: dict[str, Any]) -> Path:
        archive_dir = self.kb_dir / "qa_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        archive_path = archive_dir / name
        archive_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return archive_path

    def _ensure_client(self) -> None:
        if self.client is not None:
            return
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "缺少 openai 依赖，请先执行: python -m pip install -r requirements_kb_qa.txt"
            ) from exc

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base) if self.api_base else OpenAI(api_key=self.api_key)
        else:
            self.client = OpenAI(base_url=self.api_base) if self.api_base else OpenAI()
