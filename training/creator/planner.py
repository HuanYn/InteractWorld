"""Bounded action planning, without models, network access or tensor dependencies.

The rule fallback deliberately handles a small Chinese/English command grammar.
It produces input controls, not assurances about generated video or world state.
An optional external proposal crosses the same schema and preservation boundary;
its claimed ``preserved`` value is never used as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


TOTAL_FRAMES = 240
FPS = 16
KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")
MOVEMENT_KEYS = frozenset(KEYS[:4])
CAMERA_KEYS = frozenset(KEYS[4:])
OPPOSING = (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L"))
RESULT_FIELDS = frozenset(("status", "explanation", "action_segments", "goals",
                           "edit_scope", "preserved", "planner_kind"))
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


def _result(status, explanation, rows=None, goals=None, scope="all", preserved=False,
            kind="rule_fallback"):
    return dict(status=status, explanation=explanation,
                action_segments=compress_segments(rows) if rows is not None else [],
                goals=goals or [], edit_scope=scope, preserved=preserved, planner_kind=kind)


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
SHORTEN = re.compile(r"缩短|减少|短一点|少一点|shorten|reduce|shorter|less")
LENGTHEN = re.compile(r"延长|增加|长一点|多一点|lengthen|increase|longer|more")
REMOVE = re.compile(r"不要|取消|去掉|不再|别|remove|without|do\s+not|don't|stop")
HALF_LATE = re.compile(r"后半段?|下半段?|最后一半|second\s+half|latter\s+half")
HALF_EARLY = re.compile(r"前半段?|上半段?|first\s+half")
SEQUENCE = re.compile(r"先|然后|再|接着|之后|first|then|afterwards")
CONTINUOUS = re.compile(r"全程|一直|持续|throughout|continuously|the\s+whole\s+time")
FILLER = re.compile(
    r"请|帮我|一下|一点|仅|只|要|把|将|的|时间|时长|动作|镜头|视角|移动|行走|"
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


def _parse(text, previous_rows):
    if type(text) is not str or not text.strip() or len(text) > 2000:
        return None, _result("clarify", "请输入 1–2000 字的移动或镜头指令。")
    text = text.strip().lower()
    if UNSUPPORTED.search(text):
        return None, _result("unsupported", "当前只支持 WASD 移动和 IJKL 镜头输入；不能保证物品交互、角色动作、精确导航或创建新场景。")
    keep_move = bool(PRESERVE_MOVEMENT.search(text) or ONLY_CAMERA.search(text))
    keep_camera = bool(PRESERVE_CAMERA.search(text) or ONLY_MOVEMENT.search(text))
    cleaned = text
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
    residual = "".join(" " if index in occupied else char for index, char in enumerate(cleaned))
    for pattern in (SHORTEN, LENGTHEN, REMOVE, FILLER):
        residual = pattern.sub("", residual)
    if residual.strip():
        return None, _result("clarify", "规则回退无法完整理解这条指令；请用明确的方向、前/后半段，或缩短/延长/取消某个动作。", scope=scope)
    if operation != "set" and len(keys) != 1:
        return None, _result("clarify", "缩短、延长或取消时，请指定一个动作。", scope=scope)
    if len(found) != len(keys) and operation == "set":
        return None, _result("clarify", "重复动作的起止时刻不明确，请简化为一个动作或两个先后动作。", scope=scope)
    return _Intent(keys, scope, operation, cleaned, found), None


def _preserved(rows, original, scope):
    if original is None or scope == "all":
        return False
    protected = MOVEMENT_KEYS if scope == "camera" else CAMERA_KEYS
    return all(set(a) & protected == set(b) & protected for a, b in zip(rows, original))


def _fallback(intent, original):
    keys, scope, operation = intent.keys, intent.scope, intent.operation
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
            if stop - start == len(indices):
                return _result("clarify", "该动作已覆盖全部 240 帧，无法继续延长。", scope=scope)
            for index in range(start, stop):
                rows[index].add(key)
            detail = f"将{LABELS[key]}从 {len(indices)} 帧延长为 {stop - start} 帧，优先向后延伸。"
    else:
        for row in rows:
            row.intersection_update(protected if scope != "all" else set())
        late, early = bool(HALF_LATE.search(intent.text)), bool(HALF_EARLY.search(intent.text))
        sequential = len(keys) > 1 and (bool(SEQUENCE.search(intent.text)) or late or early)
        if len(keys) > 2:
            return _result("clarify", "规则回退一次最多安排两个动作；请拆分指令。", scope=scope)
        if len(keys) == 1 and late and early:
            return _result("clarify", "同一个动作同时指定了前半段和后半段，请明确时段。", scope=scope)
        if len(keys) == 2 and (late or early) and not (
                (HALF_LATE.search(intent.text) and HALF_LATE.search(intent.text).start() > intent.actions[0][0])
                or (early and late)):
            return _result("clarify", "时间段对应的动作不明确，请使用“先前进，然后抬头”。", scope=scope)
        for index, key in enumerate(keys):
            start, stop = (0, TOTAL_FRAMES)
            if sequential:
                start, stop = ((0, 120) if index == 0 else (120, 240))
                if index == 0 and CONTINUOUS.search(intent.text):
                    start, stop = 0, TOTAL_FRAMES
            elif late:
                start = 120
            elif early:
                stop = 120
            for row in rows[start:stop]:
                row.add(key)
        detail = "按明确方向安排输入；先后动作各占 120 帧，未指定时段的单个动作占 240 帧。"
        if sequential and CONTINUOUS.search(intent.text):
            detail = "第一个动作覆盖 240 帧，第二个动作从第 121 帧开始。"
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
        if not isinstance(proposal, dict) or not set(proposal).issubset(RESULT_FIELDS):
            raise ValueError("unknown proposal fields")
        if not {"action_segments", "edit_scope"}.issubset(proposal):
            raise ValueError("proposal requires action_segments and edit_scope")
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
            if proposal["action_segments"] != []:
                raise ValueError("non-ready proposal must not contain executable actions")
            return _result(status, proposal.get("explanation", "外部提案需要进一步澄清。"), scope=scope, kind=kind)
        rows = expand_segments(proposal["action_segments"])
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
            if any(set(a) - {key} != set(b) - {key} for a, b in zip(rows, original)):
                raise ValueError("duration edit changes unrelated controls")
        return _result("ready", "外部提案已通过结构和编辑范围验证；这不验证生成画面或任务完成。",
                       rows, goals, scope, preserved, kind)
    except (ValueError, TypeError) as error:
        return _result("clarify", "外部提案被拒绝：" + str(error), scope=scope, kind=kind)


def plan_request(text: str, previous: dict | None = None, proposal: dict | None = None) -> dict:
    """Plan 240 control frames or return a non-executable clarification.

    ``previous`` is a successful result or ``{'action_segments': [...]}``.
    ``proposal`` requires ``action_segments`` and ``edit_scope``; optional fields
    are the other result fields. Unknown fields and unknown controls are errors.
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
