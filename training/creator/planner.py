"""Bounded action planning, without models, network access or tensor dependencies.

The rule fallback deliberately handles a small Chinese/English command grammar.
It produces input controls, not assurances about generated video or world state.
An optional external proposal crosses the same schema and preservation boundary;
its claimed ``preserved`` value is never used as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .timeline_patch import (compile_edits, edits_from_intervals, frame_seconds,
                             intervals_from_rows, seconds_to_frame)


TOTAL_FRAMES = 240
FPS = 16
KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
MOVEMENT_KEYS = frozenset(KEYS[:4])
CAMERA_KEYS = frozenset(KEYS[4:])
OPPOSING = (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L"))
RESULT_FIELDS = frozenset(("status", "explanation", "action_segments", "goals",
                           "edit_scope", "preserved", "planner_kind", "edit_patch"))
PROPOSAL_FIELDS = RESULT_FIELDS | {"edits"}
LABELS = dict(W="向前移动", A="向左移动", S="向后移动", D="向右移动",
              I="镜头抬头", J="镜头向左", K="镜头低头", L="镜头向右")


def _keys(value) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(type(key) is str for key in value):
        raise ValueError("keys must be a list of canonical key strings")
    if len(set(value)) != len(value) or not set(value).issubset(KEYS):
        raise ValueError("unknown or duplicate keys")
    if any(a in value and b in value for a, b in OPPOSING):
        raise ValueError("opposing simultaneous keys are not supported")
    return tuple(key for key in KEYS if key in value)


def expand_segments(segments: list[dict]) -> list[tuple[str, ...]]:
    """Validate and expand exactly 240 frames into canonical key tuples."""
    if not isinstance(segments, list) or not 1 <= len(segments) <= TOTAL_FRAMES:
        raise ValueError("timeline needs 1..240 segments")
    rows = []
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"frames", "keys"}:
            raise ValueError("invalid action segment fields")
        frames = segment["frames"]
        if type(frames) is not int or not 1 <= frames <= TOTAL_FRAMES:
            raise ValueError("frames must be an integer in 1..240")
        if not isinstance(segment["keys"], list):
            raise ValueError("segment keys must be a list")
        keys = _keys(segment["keys"])
        if len(rows) + frames > TOTAL_FRAMES:
            raise ValueError("timeline exceeds 240 frames")
        rows.extend([keys] * frames)
    if len(rows) != TOTAL_FRAMES:
        raise ValueError("timeline must cover exactly 240 frames")
    return rows


def compress_segments(rows: list[tuple[str, ...]]) -> list[dict]:
    """Validate 240 key rows and combine adjacent identical rows losslessly."""
    if not isinstance(rows, (list, tuple)) or len(rows) != TOTAL_FRAMES:
        raise ValueError("timeline must cover exactly 240 frames")
    segments = []
    for row in rows:
        keys = list(_keys(row))
        if segments and segments[-1]["keys"] == keys:
            segments[-1]["frames"] += 1
        else:
            segments.append({"frames": 1, "keys": keys})
    return segments


def _compiled_control_goals(rows, edit_patch):
    """Describe executable tracks, not model-authored prose or observed success."""
    goals = []
    edited_keys = {edit["key"] for edit in edit_patch["edits"]}
    for key in KEYS:
        intervals = intervals_from_rows(rows, key)
        if not intervals and key not in edited_keys:
            continue
        spans = [f"[{frame_seconds(start)},{frame_seconds(end)})" for start, end in intervals]
        detail = "、".join(spans) + " 秒" if spans else "无按键区间（该键已取消）"
        prefix = f"控制输入目标（不代表画面已完成）：{key} {LABELS[key]}输入 "
        if len(prefix + detail) > 200:
            # Keep the existing 200-character goal contract. Dense protected
            # tracks remain fully specified in action_segments, never dropped.
            detail = "、".join(spans[:6]) + f" 秒；另有 {len(spans) - 6} 个区间，完整边界见动作时间线"
        goals.append(prefix + detail)
    return goals


def _result(status, explanation, rows=None, goals=None, scope="all", preserved=False,
            kind="rule_fallback", edit_patch=None):
    result = dict(status=status, explanation=explanation,
                action_segments=compress_segments(rows) if rows is not None else [],
                goals=goals or [], edit_scope=scope, preserved=preserved, planner_kind=kind)
    if edit_patch is not None:
        result["edit_patch"] = edit_patch
        result["goals"] = _compiled_control_goals(rows, edit_patch)
    return result


# Camera patterns run before movement patterns to avoid treating camera-left as A.
ACTION_PATTERNS = (
    ("I", r"(?:镜头|视角)?(?:向上看|抬头|仰视)|(?:look|tilt|camera|view)\s+up|raise\s+(?:the\s+)?camera"),
    ("K", r"(?:镜头|视角)?(?:向下看|低头|俯视)|(?:look|tilt|camera|view)\s+down|lower\s+(?:the\s+)?camera"),
    ("J", r"(?:镜头|视角)(?:向|往|朝)?左(?:转|看|移动)?|向左看|(?:look|pan|camera|view)\s+left"),
    ("L", r"(?:镜头|视角)(?:向|往|朝)?右(?:转|看|移动)?|向右看|(?:look|pan|camera|view)\s+right"),
    ("W", r"(?:向|往)?前(?:移动|走|进)|(?:move|walk|go)(?:\s+straight)?\s+forward|forward"),
    ("S", r"(?:向|往)?后(?:移动|走|退)|后退|(?:move|walk|go)\s+backward(?:s)?|backward(?:s)?"),
    ("A", r"(?:向|往)?左(?:移动|走|移)|(?:move|walk|strafe)\s+left"),
    ("D", r"(?:向|往)?右(?:移动|走|移)|(?:move|walk|strafe)\s+right"),
)
UNSUPPORTED = re.compile(
    r"钥匙|开门|关门|拿起|捡起|拾取|物品|npc|对话|交谈|攻击|跳跃|跳起|角色|人物|回头|转身|"
    r"精确|坐标|导航|到达|走到|新场景|换场景|切换场景|生成场景|任意场景|"
    r"\b(?:keycard|key|door|npc|interact|pick\s+up|grab|collect|jump|attack|character|avatar|"
    r"turn\s+around|navigate|navigation|coordinates?|destination|reach|new\s+scene|change\s+(?:the\s+)?scene)\b"
)
PRESERVE_MOVEMENT = re.compile(
    r"(?:保留|保持|不改|不改变)(?:原来的?|原有的?)?(?:前进|向前移动|移动|行走)|"
    r"(?:keep|preserve|retain)(?:\s+the)?\s+(?:forward\s+)?(?:movement|walking|forward)|"
    r"(?:movement|walking)\s+unchanged"
)
PRESERVE_CAMERA = re.compile(
    r"(?:保留|保持|不改|不改变)(?:原来的?|原有的?)?(?:镜头|视角)|"
    r"(?:keep|preserve|retain)(?:\s+the)?\s+(?:camera|view)|(?:camera|view)\s+unchanged"
)
ONLY_CAMERA = re.compile(
    r"(?:只|仅)(?:修改|调整|改变|改)?(?:镜头|视角)|"
    r"(?:edit|change|modify)\s+(?:only\s+)?(?:the\s+)?(?:camera|view)(?:\s+only)?|"
    r"(?:camera|view)\s+only"
)
ONLY_MOVEMENT = re.compile(
    r"(?:只|仅)(?:修改|调整|改变|改)?(?:移动|行走)|"
    r"(?:edit|change|modify)\s+(?:only\s+)?(?:the\s+)?(?:movement|walking)(?:\s+only)?|"
    r"(?:movement|walking)\s+only"
)
SHORTEN = re.compile(r"(?:缩短|减少)(?:一半)?|减半|短一点|少一点|shorten|reduce|shorter|less")
LENGTHEN = re.compile(r"(?:延长|增加)(?:一半)?|长一点|多一点|lengthen|increase|longer|more")
REMOVE = re.compile(r"不要|取消|去掉|不再|别|remove|without|do\s+not|don't|stop")
HALF_LATE = re.compile(r"后半段?|下半段?|最后一半|second\s+half|latter\s+half")
HALF_EARLY = re.compile(r"前半段?|上半段?|first\s+half")
SEQUENCE = re.compile(r"先|然后|再|接着|之后|first|then|afterwards")
CONTINUOUS = re.compile(r"全程|一直|持续|throughout|continuously|the\s+whole\s+time")
FILLER = re.compile(
    r"请|帮我|一下|一点|仅|只|要|把|将|让|的|时间|时长|动作|镜头|视角|移动|行走|"
    r"保持|保留|不变|原来|原有|改为|改成|修改|调整|同时|并且|并|和|再|然后|接着|之后|先|"
    r"前半段?|上半段?|后半段?|下半段?|最后一半|全程|一直|持续|"
    r"\b(?:please|only|just|the|a|an|and|while|with|at|in|for|of|to|duration|time|"
    r"movement|walking|camera|view|keep|preserve|retain|unchanged|original|change|edit|"
    r"first|then|afterwards|second|latter|half|throughout|continuously|whole|make|it|this)\b|"
    r"[\s,，。.!！;；:：、]+"
)


@dataclass
class _Intent:
    keys: list[str]
    scope: str
    operation: str
    text: str
    actions: list[tuple[int, int, str]]
    intervals: dict | None = None
    preserve_other: bool = False


_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_OTHER_UNCHANGED = re.compile(r"(?:其余|其他)(?:部分|动作|输入)?(?:都)?(?:保持)?(?:不变|不要动|不改动|不改|不动)")


def _timed_intervals(text, actions):
    """Recognize a bounded explicit-time grammar independently of model output.

    All clauses must be consumed. In particular, unfamiliar modifiers are not
    dropped merely because an action and a number appear somewhere in them.
    """
    marked, cursor = [], 0
    for start, end, key in actions:
        marked.extend((text[cursor:start], f"@{key}@"))
        cursor = end
    marked.append(text[cursor:])
    marked = re.sub(r"\s+", "", "".join(marked))
    clauses = re.split(r"[,，。;；！!]|(?=然后|随后|接着|之后)", marked)
    intervals, sequence_end = {}, 0
    saw_action = False
    for clause in clauses:
        clause = re.sub(r"^(?:请|帮我|只|仅|把|将)+", "", clause)
        if clause in ("", "不变", "保持不变"):
            continue
        subsequent = bool(re.match(r"(?:然后|随后|接着|之后|再)", clause))
        clause = re.sub(r"^(?:然后|随后|接着|之后|再|先)", "", clause)
        match = re.fullmatch(rf"(?:在)?(?:第)?({_NUMBER})(?:到|至|[-—–~～])({_NUMBER})秒(?:期间)?@([WASDIJKL])@", clause)
        if match:
            start, end, key = seconds_to_frame(match[1]), seconds_to_frame(match[2]), match[3]
        else:
            match = re.fullmatch(rf"前({_NUMBER})秒@([WASDIJKL])@", clause)
            if match:
                start, end, key = 0, seconds_to_frame(match[1]), match[2]
                if subsequent and saw_action:
                    raise ValueError("前 N 秒与随后执行相冲突")
            else:
                match = re.fullmatch(rf"@([WASDIJKL])@({_NUMBER})秒", clause)
                if match:
                    if saw_action and not subsequent:
                        raise ValueError("多个定长动作请用然后/随后明确先后")
                    start, key = sequence_end if saw_action else 0, match[1]
                    end = start + seconds_to_frame(match[2])
                else:
                    match = re.fullmatch(r"(?:全程|一直|持续)@([WASDIJKL])@", clause)
                    if not match:
                        raise ValueError("时间指令未完整识别；请用第 8–10 秒抬头，或前 5 秒前进，然后抬头 3 秒")
                    start, end, key = 0, TOTAL_FRAMES, match[1]
                    if subsequent:
                        raise ValueError("全程动作不能同时要求随后才开始")
        if end <= start or end > TOTAL_FRAMES:
            raise ValueError("动作时段须为 0–15 秒内的非空区间")
        if subsequent and saw_action and start < sequence_end:
            raise ValueError("明确区间与随后执行的顺序相冲突")
        intervals.setdefault(key, []).append((start, end))
        sequence_end, saw_action = end, True
    if not intervals:
        raise ValueError("未找到带时间的动作")
    for values in intervals.values():
        values.sort()
        if any(right[0] < left[1] for left, right in zip(values, values[1:])):
            raise ValueError("同一个动作的区间重叠")
    return intervals


def _parse(text, previous_rows):
    if type(text) is not str or not text.strip() or len(text) > 2000:
        return None, _result("clarify", "请输入 1–2000 字的移动或镜头指令。")
    text = text.strip().lower()
    if UNSUPPORTED.search(text):
        return None, _result("unsupported", "当前只支持 WASD 移动和 IJKL 镜头输入；不能保证物品交互、角色动作、精确导航或创建新场景。")
    keep_move = bool(PRESERVE_MOVEMENT.search(text) or ONLY_CAMERA.search(text))
    keep_camera = bool(PRESERVE_CAMERA.search(text) or ONLY_MOVEMENT.search(text))
    preserve_other = bool(_OTHER_UNCHANGED.search(text))
    cleaned = _OTHER_UNCHANGED.sub(" ", text)
    for pattern in (PRESERVE_MOVEMENT, PRESERVE_CAMERA, ONLY_CAMERA, ONLY_MOVEMENT):
        cleaned = pattern.sub(" ", cleaned)
    found, occupied = [], set()
    for key, pattern in ACTION_PATTERNS:
        for match in re.finditer(pattern, cleaned):
            positions = set(range(*match.span()))
            if not positions & occupied:
                found.append((match.start(), match.end(), key))
                occupied |= positions
    found.sort()
    keys = list(dict.fromkeys(key for _, _, key in found))
    operations = [name for name, pattern in (("shorten", SHORTEN), ("lengthen", LENGTHEN), ("remove", REMOVE))
                  if pattern.search(cleaned)]
    if len(operations) > 1:
        return None, _result("clarify", "一次局部修改请只选择缩短、延长或取消中的一种。")
    operation = operations[0] if operations else "set"
    # A duration-only edit is safe only if exactly one control is active.
    if not keys and operation != "set" and previous_rows is not None:
        active = set().union(*(set(row) for row in previous_rows))
        if keep_move:
            active &= CAMERA_KEYS
        if keep_camera:
            active &= MOVEMENT_KEYS
        if len(active) == 1:
            keys = [next(iter(active))]
    movement, camera = bool(set(keys) & MOVEMENT_KEYS), bool(set(keys) & CAMERA_KEYS)
    scope = "all"
    if keep_move and keep_camera:
        return None, _result("clarify", "移动和镜头都要求保留；请明确要修改哪一类输入。")
    if keep_move:
        scope = "camera"
        if movement:
            return None, _result("clarify", "保留移动与修改移动相冲突，请明确编辑范围。", scope=scope)
    elif keep_camera:
        scope = "movement"
        if camera:
            return None, _result("clarify", "保留镜头与修改镜头相冲突，请明确编辑范围。", scope=scope)
    elif previous_rows is not None and movement != camera:
        scope = "movement" if movement else "camera"
    if (keep_move or keep_camera or operation != "set") and previous_rows is None:
        return None, _result("clarify", "局部修改需要先有一条有效的原始时间线。", scope=scope)
    if not keys:
        return None, _result("clarify", "请明确前进、后退、左右移动，或抬头、低头、镜头向左/向右；单独说转向或减少时间可能有歧义。", scope=scope)
    if re.search(r"\d", cleaned):
        if operation != "set":
            return None, _result("clarify", "精确秒数请用替换区间表达，例如第 8–10 秒抬头；不要混合缩短/延长/取消。", scope=scope)
        try:
            intervals = _timed_intervals(cleaned, found)
        except ValueError as error:
            return None, _result("clarify", str(error), scope=scope)
        return _Intent(keys, scope, operation, cleaned, found, intervals, preserve_other), None
    residual = "".join(" " if index in occupied else char for index, char in enumerate(cleaned))
    for pattern in (SHORTEN, LENGTHEN, REMOVE, FILLER):
        residual = pattern.sub("", residual)
    if residual.strip():
        return None, _result("clarify", "规则回退无法完整理解这条指令；请用明确的方向、前/后半段，或缩短/延长/取消某个动作。", scope=scope)
    if operation != "set" and len(keys) != 1:
        return None, _result("clarify", "缩短、延长或取消时，请指定一个动作。", scope=scope)
    if len(found) != len(keys) and operation == "set":
        return None, _result("clarify", "重复动作的起止时刻不明确，请简化为一个动作或两个先后动作。", scope=scope)
    return _Intent(keys, scope, operation, cleaned, found, preserve_other=preserve_other), None


def _preserved(rows, original, scope):
    if original is None or scope == "all":
        return False
    protected = MOVEMENT_KEYS if scope == "camera" else CAMERA_KEYS
    return all(set(a) & protected == set(b) & protected for a, b in zip(rows, original))


def _legacy_set_intervals(intent):
    """Attach half/full-span modifiers to the action they describe, not globally."""
    keys = intent.keys
    if len(keys) > 2:
        raise ValueError("规则回退一次最多安排两个动作；请拆分指令。")
    prefixes, end = [], 0
    for start, stop, _ in intent.actions:
        prefixes.append(intent.text[end:start])
        end = stop
    tail = intent.text[end:]
    if len(keys) == 1:
        prefixes[0] += tail
    elif any(pattern.search(tail) for pattern in (HALF_EARLY, HALF_LATE, CONTINUOUS)):
        raise ValueError("请把前半段/后半段/全程写在对应动作之前。")
    early = [bool(HALF_EARLY.search(prefix)) for prefix in prefixes]
    late = [bool(HALF_LATE.search(prefix)) for prefix in prefixes]
    continuous = [bool(CONTINUOUS.search(prefix)) for prefix in prefixes]
    if any(sum(markers) > 1 for markers in zip(early, late, continuous)):
        raise ValueError("同一个动作的前半段、后半段和全程要求相冲突。")
    sequential = len(keys) > 1 and (bool(SEQUENCE.search(intent.text)) or any(early) or any(late))
    if len(keys) == 2 and (early[0] or late[0]) and not (early[1] or late[1]):
        raise ValueError("时间段对应的动作不明确，请分别写明两个动作的时段。")
    if sequential and len(keys) == 2 and continuous[1]:
        raise ValueError("后执行的动作不能同时从开头全程执行，请明确时段。")
    intervals = {}
    for index, key in enumerate(keys):
        start, stop = (0, 120) if sequential and index == 0 else (120, 240) if sequential else (0, 240)
        if early[index]:
            start, stop = 0, 120
        elif late[index]:
            start, stop = 120, 240
        elif continuous[index]:
            start, stop = 0, 240
        intervals[key] = [(start, stop)]
    if len(keys) == 2 and not continuous[0] and re.search(r"然后|随后|接着|之后|再|then|afterwards", prefixes[1]):
        if intervals[keys[1]][0][0] < intervals[keys[0]][0][1]:
            raise ValueError("前后半段与指定的先后顺序相冲突。")
    return intervals


def _fallback(intent, original):
    keys, scope, operation = intent.keys, intent.scope, intent.operation
    if intent.intervals is not None or (operation == "set" and intent.preserve_other):
        try:
            intervals = intent.intervals if intent.intervals is not None else _legacy_set_intervals(intent)
            canonical, patch = compile_edits(edits_from_intervals(intervals), original)
        except ValueError as error:
            return _result("clarify", "区间编辑被拒绝：" + str(error), scope=scope)
        return _result("ready", "规则回退（未调用语言模型）：按 16 FPS 精确编译半开时间区间；仅替换指定键的时间轨，其余键逐帧保留。只保证控制输入，不保证生成画面局部不变。",
                       canonical, [LABELS[key] for key in keys], scope,
                       _preserved(canonical, original, scope), edit_patch=patch)
    protected = MOVEMENT_KEYS if scope == "camera" else CAMERA_KEYS
    rows = [set(row) for row in original] if original is not None else [set() for _ in range(TOTAL_FRAMES)]
    if operation != "set":
        key = keys[0]
        indices = [i for i, row in enumerate(rows) if key in row]
        if not indices:
            return _result("clarify", f"原时间线没有{LABELS[key]}，无法执行这项局部修改。", scope=scope)
        if operation == "remove":
            for row in rows:
                row.discard(key)
            detail = f"取消{LABELS[key]}。"
        elif indices != list(range(indices[0], indices[-1] + 1)):
            return _result("clarify", "该动作有多个分散区间，请先指定要调整哪一段。", scope=scope)
        elif ('减半' in intent.text or '一半' in intent.text) and len(indices) % 2:
            return _result("clarify", "原时长为奇数帧，无法精确按一半分割；请使用普通缩短或延长并核对计划。", scope=scope)
        elif operation == "shorten":
            if len(indices) == 1:
                return _result("clarify", "该动作只有 1 帧；如需去掉，请明确取消。", scope=scope)
            duration = max(1, len(indices) // 2)
            for index in indices[duration:]:
                rows[index].discard(key)
            detail = f"将{LABELS[key]}从 {len(indices)} 帧缩短为 {duration} 帧，保留原开始时刻。"
        else:
            # Extend to the right first; if at the end, extend to the left.
            extra = max(1, len(indices) // 2)
            stop = min(TOTAL_FRAMES, indices[-1] + 1 + extra)
            start = max(0, indices[0] - (extra - (stop - indices[-1] - 1)))
            if '一半' in intent.text and stop - start != len(indices) + extra:
                return _result("clarify", "15 秒时间线不足以增加一半时长，请减少幅度或先缩短原动作。", scope=scope)
            if stop - start == len(indices):
                return _result("clarify", "该动作已覆盖全部 240 帧，无法继续延长。", scope=scope)
            for index in range(start, stop):
                rows[index].add(key)
            detail = f"将{LABELS[key]}从 {len(indices)} 帧延长为 {stop - start} 帧，优先向后延伸。"
    else:
        for row in rows:
            row.intersection_update(protected if scope != "all" else set())
        try:
            intervals = _legacy_set_intervals(intent)
        except ValueError as error:
            return _result("clarify", str(error), scope=scope)
        for key, ((start, stop),) in intervals.items():
            for row in rows[start:stop]:
                row.add(key)
        detail = "按明确方向安排输入；先后动作各占 120 帧，未指定时段的单个动作占 240 帧。"
        if CONTINUOUS.search(intent.text):
            detail = "全程修饰的动作覆盖 240 帧；其他动作按各自的时段编排。"
    try:
        canonical = [_keys(tuple(row)) for row in rows]
    except ValueError:
        return _result("clarify", "修改会产生相反方向的同时按键，请明确要替换的动作或时段。", scope=scope)
    return _result("ready", "规则回退（未调用语言模型）：" + detail + "只编排控制输入，不保证生成画面中的任务完成。",
                   canonical, [LABELS[key] for key in keys], scope,
                   _preserved(canonical, original, scope))


def _validate_proposal(proposal, intent, original):
    kind, scope = "external_proposal_validated", intent.scope
    try:
        if not isinstance(proposal, dict) or not set(proposal).issubset(PROPOSAL_FIELDS):
            raise ValueError("unknown proposal fields")
        if "edit_scope" not in proposal or not ({"action_segments", "edits"} & set(proposal)):
            raise ValueError("proposal requires edits or action_segments, and edit_scope")
        if "edits" in proposal and proposal.get("action_segments") not in (None, []):
            raise ValueError("proposal cannot supply both edits and executable action_segments")
        if proposal["edit_scope"] != scope:
            raise ValueError("proposal edit_scope differs from the requested scope")
        if proposal.get("status", "ready") not in ("ready", "clarify", "unsupported"):
            raise ValueError("invalid proposal status")
        for field in ("explanation", "planner_kind"):
            if field in proposal and (type(proposal[field]) is not str or len(proposal[field]) > 2000):
                raise ValueError(f"invalid {field}")
        goals = proposal.get("goals", [LABELS[key] for key in intent.keys])
        if not isinstance(goals, list) or len(goals) > 16 or any(type(x) is not str or not x or len(x) > 200 for x in goals):
            raise ValueError("invalid goals")
        if "preserved" in proposal and type(proposal["preserved"]) is not bool:
            raise ValueError("invalid preserved flag")
        status = proposal.get("status", "ready")
        if status != "ready":
            if proposal.get("action_segments", []) != [] or proposal.get("edits", []) != []:
                raise ValueError("non-ready proposal must not contain executable actions")
            return _result(status, proposal.get("explanation", "外部提案需要进一步澄清。"), scope=scope, kind=kind)
        patch = None
        if "edits" in proposal:
            rows, patch = compile_edits(proposal["edits"], original)
            if {edit["key"] for edit in patch["edits"]} != set(intent.keys):
                raise ValueError("patch keys differ from requested directions")
        else:
            rows = expand_segments(proposal["action_segments"])
        if intent.operation == "set":
            requested = set(intent.keys)
            protected = MOVEMENT_KEYS if scope == "camera" else CAMERA_KEYS if scope == "movement" else set()
            if patch is not None or intent.intervals is not None or intent.preserve_other:
                protected = set(KEYS) - requested
            retained = set().union(*(set(row) & protected for row in original)) if original else set()
            actual = set().union(*(set(row) for row in rows))
            if not requested.issubset(actual) or actual - requested - retained:
                raise ValueError("proposal drops requested directions or introduces unrelated controls")
            has_timing = (intent.intervals is not None or intent.preserve_other or SEQUENCE.search(intent.text)
                          or HALF_EARLY.search(intent.text) or HALF_LATE.search(intent.text)
                          or CONTINUOUS.search(intent.text))
            if has_timing:
                expected = _fallback(intent, original)
                if expected["status"] != "ready":
                    raise ValueError("requested time semantics need clarification")
                expected_rows = expand_segments(expected["action_segments"])
                if any(set(a) & requested != set(b) & requested for a, b in zip(rows, expected_rows)):
                    raise ValueError("proposal does not match requested temporal order or intervals")
                if intent.intervals is not None or intent.preserve_other:
                    if rows != expected_rows:
                        raise ValueError("interval edit changes an unrelated control track")
                    patch = expected["edit_patch"]
        preserved = _preserved(rows, original, scope)
        if scope != "all" and not preserved:
            raise ValueError("proposal changes controls outside the requested edit_scope")
        if intent.operation != "set":
            key = intent.keys[0]
            before = sum(key in row for row in original)
            after = sum(key in row for row in rows)
            valid = ((intent.operation == "remove" and after == 0 and before > 0)
                     or (intent.operation == "shorten" and 0 < after < before)
                     or (intent.operation == "lengthen" and before < after))
            if not valid:
                raise ValueError("proposal does not perform the requested duration edit")
            if '减半' in intent.text or '一半' in intent.text:
                if before % 2:
                    raise ValueError("explicit half-duration edit needs an even original frame count")
                expected = max(1, before // 2) if intent.operation == 'shorten' else before + max(1, before // 2)
                if after != expected:
                    raise ValueError("proposal does not match the explicitly requested duration ratio")
            if any(set(a) - {key} != set(b) - {key} for a, b in zip(rows, original)):
                raise ValueError("duration edit changes unrelated controls")
        return _result("ready", "外部提案已通过结构和编辑范围验证；这不验证生成画面或任务完成。",
                       rows, goals, scope, preserved, kind, edit_patch=patch)
    except (ValueError, TypeError) as error:
        return _result("clarify", "外部提案被拒绝：" + str(error), scope=scope, kind=kind)


def plan_request(text: str, previous: dict | None = None, proposal: dict | None = None) -> dict:
    """Plan 240 control frames or return a non-executable clarification.

    ``previous`` is a successful result or ``{'action_segments': [...]}``.
    ``proposal`` requires ``edit_scope`` and either legacy ``action_segments``
    or typed ``edits``. A replace_intervals edit rewrites only its key's track.
    Seconds are exact 16-FPS frame boundaries; all other controls are preserved.
    Optional ``edit_patch`` output metadata is recomputed, never trusted.
    Local edits recompute preservation per frame, never from a caller's flag.
    """
    original = None
    if previous is not None:
        try:
            if not isinstance(previous, dict) or not set(previous).issubset(RESULT_FIELDS):
                raise ValueError("unknown previous-plan fields")
            if previous.get("status", "ready") != "ready":
                raise ValueError("previous plan is not ready")
            original = expand_segments(previous.get("action_segments"))
        except (ValueError, TypeError) as error:
            return _result("clarify", "原时间线无效：" + str(error))
    intent, error = _parse(text, original)
    if error is not None:
        return error
    if proposal is not None:
        return _validate_proposal(proposal, intent, original)
    return _fallback(intent, original)
