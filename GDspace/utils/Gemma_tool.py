# Gemma_tool.py

__default_MODEL__      = "google/gemma-3-4b-it"   # 멀티모달 겸용 기본 모델
__default_LOAD_TYPE__  = "bf16"

import os
import time
import gc
import sys
import torch
import re
from threading import Thread
from tqdm.auto import tqdm
from PIL import Image
from GD_utils.AI_tool.NLP_tool import split_sentences

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    AutoConfig,
    AutoProcessor,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

try:
    import bitsandbytes as bnb
    BNB_OK = True
except ImportError:
    BNB_OK = False
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# 27B 모델은 VRAM 이슈로 heavy 로 간주
HEAVY_MODELS = {
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
}

# ───────────────────────────── 전역 ────────────────────────────────
MODEL_ID = None
LOAD_TYPE = None
tok = None
processor = None  # None for 1B 텍스트 모델
model = None

USE_GPU = torch.cuda.is_available()
DEVICE_MAP = "auto" if USE_GPU else {"": "cpu"}

# ───────────────────────────── Utils ───────────────────────────────

def _bnb_quant_cfg(ltype: str, heavy: bool):
    if ltype == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if heavy else torch.bfloat16,
        )
    if ltype == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_compute_dtype=torch.bfloat16)
    return None

def free_gpu(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def _is_multimodal_capable(mid: str) -> bool:
    return "-1b-" not in mid  # 1B만 텍스트 전용

def _is_quantized(mid: str) -> bool:
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        return getattr(cfg, "quantization_config", None) is not None
    except Exception:
        return False

# ───────────────────────────── Loader ──────────────────────────────

def load_gemma_model(mid: str, ltype: str):
    global tok, processor, model, MODEL_ID, LOAD_TYPE

    # ── ① 모델이 이미 떠 있고, id 또는 로드 방식이 달라질 때만 언로드
    if model is not None and (mid != MODEL_ID or ltype != LOAD_TYPE):
        _unload_current_model()          # ← 핵심 한 줄

    print(f"\n[Load] {mid} | {ltype}")
    kwargs: dict = dict(device_map=DEVICE_MAP, trust_remote_code=True)

    if USE_GPU and BNB_OK:
        qcfg = _bnb_quant_cfg(ltype, mid in HEAVY_MODELS)
        if qcfg:
            kwargs["quantization_config"] = qcfg
        elif ltype in {"fp16", "bf16"}:
            kwargs["torch_dtype"] = torch.float16 if ltype == "fp16" else torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.float32
        kwargs.pop("quantization_config", None)

    if _is_quantized(mid):
        kwargs.pop("quantization_config", None)

    # 토크나이저
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    tok.pad_token_id = tok.pad_token_id or tok.eos_token_id

    if _is_multimodal_capable(mid):
        model = Gemma3ForConditionalGeneration.from_pretrained(mid, **kwargs).eval()
        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True, use_fast=True)
    else:
        model = Gemma3ForCausalLM.from_pretrained(mid, **kwargs).eval()
        processor = None

    MODEL_ID, LOAD_TYPE = mid, ltype

# ───────────────────────────── Generation ───────────────────────────

@torch.inference_mode()
def _generate_stream(tok, model, inputs, max_new_tokens):
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
    thr = Thread(target=model.generate, kwargs={**inputs, "max_new_tokens": max_new_tokens, "streamer": streamer, "temperature": 1.0, "top_k":64,"top_p": 0.95,"min_p": 0.01, "repetition_penalty": 1.0, "do_sample": True}, daemon=True)

    thr.start()
    out = ""
    with tqdm(total=max_new_tokens, desc="Streaming", file=sys.stderr) as pbar:
        for piece in streamer:
            out += piece
            pbar.update(1)
    return out

@torch.inference_mode()
def _gen_text(prompt: str, max_new_tokens: int, stream: bool):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    if stream:
        return _generate_stream(tok, model, inputs, max_new_tokens).split(prompt, 1)[-1].lstrip()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=1, top_p=0.95,min_p = 0.01,top_k=64, repetition_penalty=1.0, do_sample=True)
    txt = tok.decode(out[0], skip_special_tokens=True)
    return txt.split(prompt, 1)[-1].strip()

@torch.inference_mode()
def _gen_vision(image_path: str, prompt: str, max_new_tokens: int):
    if processor is None:
        raise RuntimeError("현재 로드된 모델은 이미지 입력을 지원하지 않습니다.")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path}, {"type": "text", "text": prompt}
    ]}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text_prompt], images=[img], padding=True, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded=processor.decode(new_ids, skip_special_tokens=True)
    if isinstance(decoded, list):
        decoded = " ".join(decoded)
    return decoded

# ───────────────────────────── Public API ───────────────────────────
def get_Gemma_response(prompt: str, *, image_path: str | None = None, model_id: str | None = None, load_type: str | None = None, max_new_tokens: int = 2048, stream: bool = False, verbose: bool = False):
    if model is None or tok is None or (model_id and model_id != MODEL_ID):
        load_gemma_model(model_id or __default_MODEL__, load_type or __default_LOAD_TYPE__)

    stt_time = time.time()
    if image_path:
        VLM_ans = _gen_vision(image_path, prompt, max_new_tokens)
        if verbose:
            print(f'VLM Response ############################### \n {VLM_ans}\n###############################')
            print(f'VLM consume {time.time()-stt_time:.2f}s')
        return VLM_ans

    LLM_ans = _gen_text(prompt, max_new_tokens, stream)
    if verbose:
        print(f'LLM Response ############################### \n{LLM_ans}\n###############################')
        print(f'LLM consume {time.time() - stt_time:.2f}s')
    return LLM_ans

# ───────────────────────────── Summarizer ───────────────────────────
def summarize_long_text(text: str, *, language: str = "auto", model_id: str | None = None,
                        load_type: str | None = None, max_chunk_sents: int = 5, overlap_sents: int = 2,
                        max_new_tokens: int = 512):
    sents = split_sentences(text, language=language)
    if not sents:
        return ""
    chunks, idx = [], 0
    while True:
        nxt = idx + max_chunk_sents
        chunk = "\n".join(sents[idx:nxt])
        if not chunk:
            break
        chunks.append(chunk)
        if nxt >= len(sents):
            break
        idx = max(0, nxt - overlap_sents)

    partial = []
    for c in chunks:
        p = f"아래 글을 요약해줘.\n\n글:\n{c}\n요약:"
        partial.append(get_response(p, model_id=model_id, load_type=load_type, max_new_tokens=max_new_tokens))

    merged = "\n\n".join(partial)
    final_prompt = f"다음은 부분 요약 모음입니다. 이를 종합하여 한글로 5문장 이내로 핵심만 다시 요약해줘.\n\n{merged}\n\n최종 요약:"
    return get_response(final_prompt, model_id=model_id, load_type=load_type, max_new_tokens=max_new_tokens)

def _unload_current_model():
    """현재 메모리에 올라와 있는 모든 객체·캐시를 정리한다."""
    global model, tok, processor, MODEL_ID, LOAD_TYPE

    if model is not None:
        # (선택) CPU 로 옮겨 두면 GPU → PCIe 복사를 거치지 않고 바로 free
        model.to("cpu")

    # util 함수 활용
    print("[Unload] 모델 언로드")
    log_gpu_mem("Before free_gpu:")
    free_gpu(model, tok, processor)
    log_gpu_mem("After free_gpu:")

    # 파이썬 레벨에서도 참조 끊기
    model = tok = processor = None
    MODEL_ID = LOAD_TYPE = None
def log_gpu_mem(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"{prefix} allocated={allocated:.2f}MB, reserved={reserved:.2f}MB")

_ROLE_LINE = re.compile(r"^(?:user|system|assistant|model|원문)\s*$", re.I | re.M)
def clean_gemma_output(raw: str) -> str:
    """
    Gemma-3 출력에서 프롬프트 복사본·역할 태그·여분 개행을 제거하고
    '모델 최종 답변'만 남긴다.
    """
    if not raw:
        return ""

    text = raw.lstrip()

    # 1) 마지막 'model' 헤더 이후만 남기기
    m = re.search(r"(?s)\bmodel\s*\n", text)
    if m:
        text = text[m.end():]

    # 2) 역할(ROLE) 한글/영문 헤더 라인 삭제
    text = _ROLE_LINE.sub("", text)

    # 3) 중복 개행 정리: 3줄 이상 → 2줄
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ───────────────────────────── 테스트 ───────────────────────────────
if __name__ == "__main__":
    # todo: 수정 필요
    ko_example_to_be_summarized = "지난 주말 스위스 제네바에서 첫 공식 무역 협상을 마치고 미·중 양국은 90일간 상호 부과한 고율 관세를 대폭 인하하기로 합의했다고 밝힌 가운데, 미·중간 무역 갈등 완화 기대감이 지속된 점이 투자심리를 개선시키는 모습을 보인 가운데, 외국인, 기관 동반 순매도 등에 코스닥지수는 1% 넘게 하락하며 지수 상승을 이끌었으며 외국인은 8거래일 연속 순매수, 기관은 5거래일만에 순매수로 전환했다. 한편, 무디스는 신용등급 강등의 이유로 미국 정부의 만성적인 부채 증가와 이자 부담을 지적했으며, 월가 애널리스트들은 무역 협상 등을 주목하며 관망 심리가 악화할 것이라는 우려가 완화되고 있다. 미국 경제의 불확실성이 완화될 것이라는 기대감이 지속되어 투자심리가 개선될 것으로 보인다.  국내증시 코피스 지수는 0.21% 상승한 2626.87에 마감했다.   미국증시 5월 16일 뉴욕증시는 옵션 만기일을 맞이한 가운데, 단기 상승 모멘트가 부재한 가운데, 외국인과 기관이 동반 매수했다. 미·미·중 무역 갈등 완화에 대한 기대감 및 미·무역 갈등 완화 기대감 등을 우려하고 있다. 미국과 중국이 90일간 서로 부과한 관세를 낮추기로 합의했다고 밝혔으며, 미국과 중국의 관세 인하에 대한 우려가 완화될 것으로 예상된다. 미국, 미국, 중국, 중국 등 세계 경제가 악화할 수 있다는 우려에 대한 우려에 대해 우려로 상승할 것으로 예상되며, 미국 증시가 상승세로 전환될 것으로 전망된다. 한편 미·중국과의 무역갈등 완화를 기대하며 투자심리에 영향을 미칠 것으로 보고 있다. 미-중 무역갈등이 완화될 것이라고 전망했다. 미국-미중 무역 갈등이 완화될 것이란 기대감에 투자 심리가 완화되고 있는 가운데, 미국과 중국 양국의 무역갈등에 대한 우려로 인해 투자심리의 불확실성이 악화될 것이라는 우려로 투자심리는 지속되고 있어 투자심리로 전환될 수 있다는 우려가 커지고 있다. 한편 미국, 미중 무역분쟁 완화 기대감에 따라 투자심리도 개선될 것이라고 밝혔다. 미국 무역분쟁에 대한 불확실성이 커지고 있는 상황에서 투자 심리를 완화시킬 수 있을 것으로 보고 투자 심리에 대한 우려가 해소될 수 있을 것이라고 말했다. 한편 중국, 미국과의 무역 갈등이 해소될 것이라는 기대감에도 불구하고 미중 양국이 무역분쟁을 완화할 것이라는 기대감에 대한 시장의 불확실성이 높아지고 있다. 갭이 지속되고 있는 상황이다. 셧다운이 완화될 수 있는 상황이 지속되고 있다고 밝혔다. 랠리를 회복할 수 있을 것이라는 기대감으로 투자심리 회복이 지속될 수 있다는 것이 투자심리 개선으로 이어질 수 있을 것이다. 틸리스크에 대한 불안감이 해소될 것으로 보여진다. 챕터 랠리가 지속되는 가운데, 셧 다운을 완화시킬 것이라는 기대감이 커지고 있어 투자심리 완화를 위해 미국과 중국과의 무역전쟁이 완화되는지 랠리 랠리에서 벗어나는  에 대한 로 의 와 를 에서 , 라는 가 ()  및 적 을 라고 해석했다. 미국 이  반등하고 있다로 전환될지 도 고  상승 과  등  투자심리  증가는 증시 로 해석된다. 미국로 인해 미중  미·를 통해 미국로 인한 주 나 중의 불확실성 확대 제 완화  완화를 반영  완화 대가 완화될 것이다.가 될 수 있는 지를 완화로 작용할 수 있다.라는 미·대로 이어질 수 있다는 관세 세가 지속되고 있다가 지속될 수 있는 투자심리에 대해 투자로 작용해 거래소로 투자 시장로가 기대된다.라고 판단된다. 한편 한국경제가 기대감 감에 영향을 미치고 있다."
    en_example_to_be_summarized = "Commercial shipment is expected to grow 4.3% year over year to 138 million and witness a CAGR of 0.8% between 2025 and 2029 to hit 142.6 million. HPQ & AAPL’s Earnings Estimate Revision Goes South The Zacks Consensus Estimate for HPQ’s fiscal 2026 earnings is pegged at $3.39 per share, down 1.7% over the past 30 days, indicating a 0.3% increase over fiscal 2025’s reported figure. Quote The consensus mark for a HPQ earnings estimate has declined 0.8% to $7.12 per share over the last 30 days suggesting 5.48% growth over the fiscal 2024, suggesting a 1.7% increase in the first quarter of calendar 2025, per Canalys. Commercial shipments are expected to hit 12.8 million units in 2026, a jump of 1.7% from 2024 and a year-over-year shipment growth of 2.1% in 2025. The availability of Apple Intelligence globally with macOS Sequoia 15.4 updates in new languages, including French, German, Italian, Portuguese (Brazil), Spanish, Japanese, Korean, and Chinese (simplified), as well as localized English for Singapore and India, bodes well for Mac’s prospects. The Case for HP Stock Growing interest in Generative AI-enabled PCs might give a fresh boost to HP’s PC sales. However, HP faces meaningful downside risk if the U.S.-China tariff war escalates. The company relies heavily on China for manufacturing and assembling many of its PCs, printers, and related components. Higher import tariffs on Chinese-made goods would raise HP's production costs, forcing the company to either absorb the margin pressure or pass on costs to consumers, both negative outcomes. The growing demand for artificial intelligence (AI) powered PCs and Microsoft Windows 10 end-of-service in October 2025 are key catalysts. Gartner expects AI PCs (PC with an embedded neural processing unit) global shipments to hit 114 million units between 2026 and 2028. Commercial shipment grew 4.3% from 2025 to 2026 to hit 14.2 million units. Meanwhile, consumer PC shipments were expected to remain flat between 2022 and 2023. Commercial PC shipment growth was expected to be 1.7% in 2022 to hit 422.6 million in 2024. Meanwhile in 2023, the shipment growth grew 1.7% to hit 71.7 million. Meanwhile the consumer PC shipment was predicted to grow 1.9% in 2021 to hit 132.6 million units, an increase of 1.1% from the previous year. The increasing demand for AI-powered PCs is a key catalyst for the company’s growth."
    print(len(ko_example_to_be_summarized))
    print(len(en_example_to_be_summarized))

    # IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/chart/6a7b9f07_c_4_32.png"
    # IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/figure/3d5faac9_f_6_55.png"
    IMAGE_PATH = "C:/GD_GIT/GD_Crawling/Crawling_FnGuide/images/table/f98085a1_t_3_28.png"
    PROMPT = "이 이미지에 대해 자세히 설명해줘. 절대로 추측하거나 추론하지말것, 주어진 팩트 기반으로만 설명할 것"
    # ans_VLM = get_VLM_response(IMAGE_PATH, PROMPT)
    # print(ans_VLM)

    MODELS = [
        # 4B  ─ 텍스트 + 멀티모달
        "google/gemma-3-4b-it",
        # "google/gemma-3-4b-pt",

        # # 9B  ─ 텍스트 + 멀티모달
        # "google/gemma-3-12b-it",
        # "google/gemma-3-12b-pt",

        # # 27B ─ heavy (4bit/8bit 로드만 시도)
        # "google/gemma-3-27b-it",
        # "google/gemma-3-27b-pt",
    ]
    # ‣ bf16   : 1B/4B/9B 에 권장(속도·정밀도 균형)
    # ‣ 4bit   : 최대 메모리 절약, 27B 포함 전 모델 구동 가능
    # ‣ 8bit   : 27B 모델도 안정적으로 구동, 다만 속도 ↓
    # ‣ fp16   : 1B·4B 정도에서만 사용 권장
    LOAD_TYPES = [
        "bf16", #"4bit", "8bit", "fp16"
    ]

    for model_id in MODELS[::-1]:
        for load_type in LOAD_TYPES:
            prompt = (
                        "<start_of_turn>user\n"
                        "당신은 **전문 금융·투자 요약가**입니다.\n"
                        "[규칙]\n"
                        "- 시스템·사용자 지침 및 원문 텍스트를 **그대로 재현하지 마십시오**.\n"
                        "- **정보를 생성·추론하지 말고, 주어진 사실만 토대로 요약하십시오**.\n"
                        "- 반드시 **요약 문장만** 작성하십시오.\n"
                        "- `<think>` 등 내부 사고 과정을 출력하지 마십시오.\n\n"
                        "### 원문\n"
                        f"{ko_example_to_be_summarized}\n"
                        "<end_of_turn>\n"
                        "<start_of_turn>model\n"
                    )
            # stt_time = time.time()
            # # ans_LLM = get_LLM_response(prompt=prompt, model_id=model_id, load_type=load_type, stream=True)
            # ans_LLM = get_LLM_response(prompt=PROMPT, image_path=IMAGE_PATH, model_id=model_id, load_type=load_type, stream=True)
            # print(f'{model_id} | {load_type} | stream=True | {time.time()-stt_time:.2f}s 소요')
            # print(len(ans_LLM))
            # print(f'{ans_LLM}')

            stt_time = time.time()
            ans_LLM = get_LLM_response(prompt=prompt, model_id=model_id, load_type=load_type, stream=False)
            # ans_LLM = get_LLM_response(prompt=PROMPT, image_path=IMAGE_PATH, model_id=model_id, load_type=load_type, stream=False)
            print(f'{model_id} | {load_type} | stream=False| {time.time()-stt_time:.2f}s 소요')
            print(len(ans_LLM))
            print(f'{ans_LLM}')
            # print(f'{clean_gemma_output(ans_LLM)}')

            # print(f'{sum_LLM}')
            # print(clean_qwen_output(ans_LLM))