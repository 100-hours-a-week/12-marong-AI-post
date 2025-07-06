AXIS = ["ei_score", "sn_score", "tf_score", "jp_score"]

AXIS_LABEL_MAP = {
    "EI": "ei_score",
    "SN": "sn_score",
    "TF": "tf_score",
    "JP": "jp_score"
}

AXIS_REVERSE_MAP = {v: k for k, v in AXIS_LABEL_MAP.items()}


def get_change_type(predicted: int, current: int) -> str:
    if predicted > current:
        return "상승"
    elif predicted < current:
        return "하락"
    return "유지"


def format_axis_reason(axis: str, before: int, after: int, change: str, reason: str) -> str:
    label = axis.replace("_score", "").upper()
    return f"[{label} 축]\n- 이전 점수: {before}, 예측 점수: {after} ({change})\n- 이유: {reason}"