from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io import save_json
from .models import Scene, SceneAnnotation

FUNCTIONS = {
    "A": "初始情境",
    "B": "禁令",
    "C": "违反禁令",
    "D": "敌方侦察",
    "E": "获得或传递信息",
    "F": "骗局、陷阱、悬念或暗示",
    "G": "受骗或配合欺骗",
    "H": "敌人伤害、罪行或野心",
    "I": "主角匮乏",
    "J": "获知灾祸或任务",
    "K": "主角回应或反制",
    "L": "主角出发",
    "M": "赠予者或金手指出现",
    "N": "主角回应赠予者",
    "O": "获得物品、能力或升级",
    "P": "空间转移",
    "Q": "直接冲突",
    "R": "被识别或留下标记",
    "S": "胜利或失败",
    "T": "匮乏或危机消除",
    "U": "返回或被追逐",
    "V": "从危险中获救",
    "W": "主角未被认出或被低估",
    "Fr": "世界或能力系统设定",
    "X": "不合理要求或刁难",
    "Fa": "外表、化名或身份伪装",
    "Z": "任务完成或问题解决",
    "Re": "主角被认可",
    "De": "敌人、骗局或身份暴露",
    "Y": "困难任务",
    "Em": "情绪变化",
    "Fi": "超越不合理要求",
    "Lo": "失忆",
    "Ch": "角色、关系或权力变化",
}

_KEYWORD_FUNCTIONS = [
    ("Lo", ("失忆", "记忆")),
    ("Q", ("冲突", "对抗", "交锋", "打斗", "反击", "争吵", "羞辱")),
    ("De", ("揭穿", "暴露", "真相", "识破", "身份")),
    ("O", ("升级", "能力", "法宝", "获得", "解锁")),
    ("M", ("救场", "金手指", "援助", "高手", "帮手")),
    ("Y", ("任务", "难题", "危机", "困境")),
    ("Z", ("解决", "完成", "成功", "夺回", "化解")),
    ("S", ("获胜", "胜利", "失败", "完胜", "击退", "反杀")),
    ("F", ("陷阱", "阴谋", "悬念", "暗示", "伏笔", "伪善", "欺骗")),
    ("E", ("发现", "得知", "揭晓", "信息", "线索")),
    ("P", ("抵达", "前往", "进入", "离开", "回京")),
    ("Em", ("情绪", "愤怒", "恐惧", "感动", "伤心", "爱意")),
    ("Ch", ("关系", "信任", "依赖", "权力", "成长", "转变")),
    ("K", ("决定", "应对", "反制", "行动", "回击")),
]


def _shorten(text: str, length: int = 180) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean[:length]


def _heuristic_functions(text: str) -> list[str]:
    found = [code for code, words in _KEYWORD_FUNCTIONS if any(word in text for word in words)]
    if not found:
        found = ["A", "K"]
    if found[0] != "A":
        found.insert(0, "A")
    return found[:8]


def heuristic_annotation(scene: Scene) -> SceneAnnotation:
    metadata = scene.metadata
    core_plot = metadata.get("core_plot", scene.text)
    highlights = metadata.get("highlights", "")
    progression = metadata.get("main_storyline_progression", "")
    relationship = metadata.get("relationship_changes", "")
    growth = metadata.get("character_growth", "")
    hook = metadata.get("ending_hook", "")
    sentences = [part.strip() for part in re.split(r"[。！？\n]+", core_plot) if part.strip()]
    goal = progression or (sentences[0] if sentences else scene.heading)
    obstacle = next(
        (sentence for sentence in sentences if re.search(r"却|但|危机|阻碍|敌|羞辱|失败|不料", sentence)),
        sentences[1] if len(sentences) > 1 else "未显式提取",
    )
    action = highlights or (sentences[len(sentences) // 2] if sentences else core_plot)
    turn = next(
        (sentence for sentence in sentences if re.search(r"突然|转折|揭晓|发现|没想到|就在|却", sentence)),
        sentences[-2] if len(sentences) > 1 else "未显式提取",
    )
    outcome = progression or (sentences[-1] if sentences else core_plot)
    state_change = "；".join(value for value in (growth, relationship) if value) or outcome
    combined = "\n".join([core_plot, highlights, progression, hook])
    return SceneAnnotation(
        scene_id=scene.scene_id,
        episode_id=scene.episode_id,
        scene_number=scene.scene_number,
        goal=_shorten(goal),
        obstacle=_shorten(obstacle),
        action=_shorten(action),
        turn=_shorten(turn),
        outcome=_shorten(outcome),
        hook=_shorten(hook or outcome),
        state_change=_shorten(state_change),
        narrative_functions=_heuristic_functions(combined),
        annotation_method="heuristic-provisional",
        source_text=scene.text,
    )


def _extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    return json.loads(candidate)


def _llm_annotation(scene: Scene) -> SceneAnnotation:
    url = os.environ.get("DRAMA_EVAL_API_URL", "").strip()
    api_key = os.environ.get("DRAMA_EVAL_API_KEY", "").strip()
    model = os.environ.get("DRAMA_EVAL_MODEL", "").strip()
    if not url or not model:
        raise RuntimeError("LLM 标注需要 DRAMA_EVAL_API_URL 和 DRAMA_EVAL_MODEL")
    taxonomy = "；".join(f"{code}={name}" for code, name in FUNCTIONS.items())
    prompt = f"""你是短剧叙事结构标注员。只输出JSON对象，不评价文笔。\n叙事功能表：{taxonomy}\n按剧情发生顺序给出1至8个narrative_functions代码。goal/obstacle/action/turn/outcome/hook/state_change均须是去除人名地名后仍能表达叙事功能的短语；没有钩子写“无”。\nJSON字段严格为：goal, obstacle, action, turn, outcome, hook, state_change, narrative_functions。\n场景：\n{scene.heading}\n{scene.text[:12000]}"""
    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"LLM 标注请求失败：HTTP {error.code}") from error
    result = _extract_json(payload["choices"][0]["message"]["content"])
    functions = [code for code in result["narrative_functions"] if code in FUNCTIONS]
    return SceneAnnotation(
        scene_id=scene.scene_id,
        episode_id=scene.episode_id,
        scene_number=scene.scene_number,
        goal=_shorten(str(result["goal"])),
        obstacle=_shorten(str(result["obstacle"])),
        action=_shorten(str(result["action"])),
        turn=_shorten(str(result["turn"])),
        outcome=_shorten(str(result["outcome"])),
        hook=_shorten(str(result["hook"])),
        state_change=_shorten(str(result["state_change"])),
        narrative_functions=functions or ["A"],
        annotation_method=f"llm:{model}",
        source_text=scene.text,
    )


def _annotation_from_dict(item: dict[str, Any]) -> SceneAnnotation:
    allowed = {field.name for field in SceneAnnotation.__dataclass_fields__.values()}
    return SceneAnnotation(**{key: value for key, value in item.items() if key in allowed})


def load_or_create_annotations(
    scenes: list[Scene], cache_path: str | Path, annotator: str = "heuristic"
) -> list[SceneAnnotation]:
    cache = Path(cache_path)
    source_hash = hashlib.sha256(
        "\n".join(f"{scene.scene_id}:{scene.text}" for scene in scenes).encode("utf-8")
    ).hexdigest()
    cached: dict[str, Any] = {}
    if cache.exists():
        with cache.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("source_hash") == source_hash and cached.get("annotator") == annotator:
            return [_annotation_from_dict(item) for item in cached["annotations"]]
    annotation_function = _llm_annotation if annotator == "llm" else heuristic_annotation
    annotations = [annotation_function(scene) for scene in scenes]
    save_json(
        cache,
        {
            "schema_version": 1,
            "source_hash": source_hash,
            "annotator": annotator,
            "function_taxonomy": FUNCTIONS,
            "annotations": [annotation.to_dict() for annotation in annotations],
        },
    )
    return annotations
