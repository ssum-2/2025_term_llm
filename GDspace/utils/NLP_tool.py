# NLP_tool.py
from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain
from typing import List, Sequence

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # ★ missing import fixed
from wtpsplit import SaT
import inspect

# optional – only if GPU ONNX available
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover – CPU‑only env
    ort = None
import warnings
warnings.filterwarnings("ignore")

# ────────────────── Lazy loader ──────────────────
@lru_cache(maxsize=None)
def _get_summarizer(lang: str = "kr"):
    """
    lang 별 (ko | en) 토크나이저와 모델을 최초 1회만 로드해 캐시
    """
    if "kr" in lang.lower():
        tok_name, model_name = "psyche/KoT5-summarization", "psyche/KoT5-summarization"
    elif "en" in lang.lower():
        tok_name, model_name = "google/long-t5-tglobal-base", "google/long-t5-tglobal-base"
    else:
        raise ValueError("lang should be 'ko' or 'en'")

    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    return tokenizer, model
def get_summary(ori_text: str, lang: str = "kr", max_pct_of_ori: int = 50, min_pct_of_ori: int = 10) -> str:
    tokenizer, model = _get_summarizer(lang)

    inputs = tokenizer(ori_text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)

    # 토큰 길이가 아니라 “문자 길이 비율” 기준이면 len(ori_text) 사용
    summary_ids = model.generate(
                                 **inputs,
                                 max_length=int(len(ori_text) * max_pct_of_ori // 100),
                                 min_length=int(len(ori_text) * min_pct_of_ori // 100),
                                 num_beams=5,
                                 no_repeat_ngram_size=3,
                                 early_stopping=True,
                                 )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def _filter_supported(func, kwargs: dict) -> dict:
    """
    전달하려는 kwargs 중 func 시그니처에 존재하는 것만 남김.
    SaT 버전에 따라 지원 파라미터가 달라져도 TypeError가 나지 않도록 한다.
    """
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}

@lru_cache(maxsize=1)
def _load_sat(
    *,
    checkpoint: str = "sat-12l-sm",
    ort_providers: Sequence[str] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    lora_path: str | None = None,
    fp16: bool = True,  # reserved — may be used in newer wtpsplit versions
) -> SaT:  # noqa: WPS231
    """Return cached SaT instance; GPU ONNX if available."""

    # PyTorch path fallback when onnxruntime not present
    if ort is None:
        return SaT(checkpoint, lora_path=lora_path)

    avail = set(ort.get_available_providers())
    providers = [p for p in ort_providers if p in avail] or ["CPUExecutionProvider"]

    # Build kwargs dynamically to avoid breaking older versions
    common_kw = dict(ort_providers=providers)
    if lora_path:
        common_kw["lora_path"] = lora_path

    # attempt fp16 if the current version supports it
    if fp16:
        try:
            return SaT(checkpoint, **common_kw, use_fp16_weights=True)
        except TypeError:
            pass  # silently fall back

    return SaT(checkpoint, **common_kw)

def _normalize_spacing(txt: str) -> str:
    # 한글<->영어/숫자 붙어 있는 위치에 공백 넣기
    txt = re.sub(r"([가-힣])(?=[A-Za-z0-9])", r"\1 ", txt)
    txt = re.sub(r"([A-Za-z0-9])(?=[가-힣])", r"\1 ", txt)
    # 중복 공백 정리
    return re.sub(r"\s+", " ", txt).strip()


def split_sentences(
    text: str,
    *,
    language: str = "auto",
    domain: str | None = None,
    max_chars: int = 4096,
    threshold: float = 0.025,
    stride: int = 128,
    block_size: int = 256,
    lookahead: int = 64,
    weighting: str = "hat",
    split_on_input_newlines: bool = False,
) -> List[str]:
    if not text:
        return []

    text = _normalize_spacing(text)

    # newline‑safe chunking
    chunks, buf, size = [], [], 0
    for para in text.splitlines(keepends=True):
        if size + len(para) > max_chars and buf:
            chunks.append("".join(buf)); buf, size = [], 0
        buf.append(para); size += len(para)
    if buf:
        chunks.append("".join(buf))

    sat = _load_sat()

    raw_kw = dict(
        threshold=threshold,
        stride=stride,
        block_size=block_size,
        lookahead=lookahead,
        weighting=weighting,
        split_on_input_newlines=split_on_input_newlines,
        language=None if language == "auto" else language,
        style_or_domain=domain,
    )
    kw = _filter_supported(sat.split, {k: v for k, v in raw_kw.items() if v is not None})

    sents = chain.from_iterable(sat.split(chunks, **kw))
    return [s.strip() for s in sents if s.strip()]

if __name__ == "__main__":
    messy_example = (
        "안녕하세요오늘도good morning☀️오늘은2025-05-22목요일입니다BTW"
        "이번주GDP YoY+3.2%(예상2.9%)…줄이기스킬필요😅“Don't tell me he said‘괜찮아’라고?"
        "Mr.Smith—a.k.a.‘Dr.No’—arrived at 7:30a.m.(UTC+9);조식은김치볶음밥🍚"
        "그러나英경쟁사들은‘AI-first’전략을採用中, e.g.DeepMind,B.A.I.,τ-Labs등등."
        "\n\n"
        "①資本의移動속도>政의조정속도→규제Lag발생!②金利↓→‘위기?’or‘기회?’(答없음)"
        "2024Q4보고서PDF: https://example.com/report.pdf (p.12참조)"
        "지리산(1,915m)눈꽃은>Everest base-camp영상美✨ #mountains #설경"
        "\n"
        "끝으로日本語도混ぜてみますね。次の行には中文也有：今天天气不错但PM2.5=135μg/m³😭"
        "最後にスペース없이한영중일모두混在끝!"
    )
    exp = split_sentences(messy_example)
    for s in exp:
        print("•", s)