import unicodedata

# --- Bảng luật ---
RULES_3 = {
    "ngh": "e01f",
}

RULES_2 = {
    "ph": "e021",
    "th": "e022",
    "gi": "e013",
    "tr": "e01d",
    "ch": "e01d",
    "nh": "e020",
    "ng": "e01f",
    "kh": "e01e",
    "gh": "e015",
    "uô": "e00c",
    "uơ": "e00c",
    "uâ": "e00c",
    "ươ": "e00b",
    "yê": "e00a",
    "iê": "e00a",
}

RULES_1 = {
    "a": "e000",
    "ă": "e001",
    "â": "e002",
    "b": "e011",
    "c": "e012",
    "d": "e013",
    "đ": "e014",
    "e": "e003",
    "ê": "e004",
    "g": "e015",
    "h": "e016",
    "i": "e005",
    "k": "e012",
    "l": "e017",
    "m": "e018",
    "n": "e019",
    "o": "e006",
    "ô": "e007",
    "ơ": "e002",
    "p": "e011",
    "q": "e012",
    "r": "e013",
    "s": "e01c",
    "t": "e01a",
    "u": "e008",
    "ư": "e009",
    "v": "e01b",
    "x": "e01c",
    "y": "e005",
}

FINAL_LOCK_MAP = {
    "ng": "e015",
    "nh": "e015",
    "o": "e00f",
    "u": "e00f",
    "i": "e00e",
    "y": "e00e",
    "m": "e024",
    "n": "e025",
    "t": "e026",
}


ONSETS_3 = ["ngh"]

ONSETS_2 = [
    "ph", "th", "tr", "ch", "nh", "ng", "kh", "gh", "gi"
]

ONSETS_1 = list("bcdđghklmnpqrstvx")


# ============ BỎ DẤU THANH (GIỮ Â, Ă, Ê, Ơ, Ư, Đ) ============
def remove_tone_marks(text):
    TONE_MARKS = {
        "\u0300",  # huyền
        "\u0301",  # sắc
        "\u0303",  # ngã
        "\u0309",  # hỏi
        "\u0323",  # nặng
    }

    normalized = unicodedata.normalize("NFD", text)
    result = []

    for ch in normalized:
        if ch in TONE_MARKS:
            continue
        result.append(ch)

    return unicodedata.normalize("NFC", "".join(result))
# ============ THÊM O VÀO TIẾNG THIẾU PHỤ ÂM ĐẦU ============

SPECIAL_ONSET_INSERT = chr(int("e006", 16))

def split_onset_rime(word):
    for o in ONSETS_3:
        if word.startswith(o):
            return o, word[len(o):]

    for o in ONSETS_2:
        if word.startswith(o):
            return o, word[len(o):]

    if word and word[0] in ONSETS_1:
        return word[0], word[1:]

    return "", word  # không có phụ âm đầu

def insert_virtual_onset(word):
    if not word:
        return word

    # Nếu đã có phụ âm đầu thật thì thôi
    onset, rime = split_onset_rime(word)
    if onset:
        return word

    # Nếu bắt đầu bằng o hoặc u thì không thêm
    if rime.startswith(chr(int("e00f", 16))):
        return word

    return SPECIAL_ONSET_INSERT + word

def apply_virtual_onset_rule(text):
    words = text.split()
    words = [insert_virtual_onset(w) for w in words]
    return " ".join(words)

# ============ CHUYỂN MỘT SỐ NGUYÊN ÂM ĐẶC BIỆT LÊN ĐẦU ============

PROTECTED_CLUSTERS = ["iê", "yê", "ia", "uô", "uơ"]
VOWELS_TO_MOVE = ["ươ", "ô", "ê", "ơ", "e", "â"]

def reorder_syllable(syllable):
    # Nếu chứa cụm nguyên âm bảo vệ → không đụng vào phần đó
    protected_positions = []

    for cluster in PROTECTED_CLUSTERS:
        start = syllable.find(cluster)
        if start != -1:
            protected_positions.extend(range(start, start + len(cluster)))

    # Tìm nguyên âm cần đảo nhưng không nằm trong vùng bảo vệ
    for v in VOWELS_TO_MOVE:
        idx = syllable.find(v)
        if idx != -1 and idx not in protected_positions:
            syllable = syllable[:idx] + syllable[idx+len(v):]
            return v + syllable

    return syllable

def apply_reorder_rule(text):
    syllables = text.split()
    syllables = [reorder_syllable(s) for s in syllables]
    return " ".join(syllables)


# ============ ÁP DỤNG KHOÁ ĐUÔI ============

def apply_final_locking(word):
    if not word:
        return word

    # Ưu tiên cụm 2 ký tự ở cuối
    for tail in ("ng", "nh"):
        if word.endswith(tail):
            return word[:-len(tail)] + chr(int(FINAL_LOCK_MAP[tail], 16))

    # Xét ký tự cuối
    last_char = word[-1]
    if last_char in FINAL_LOCK_MAP:
        return word[:-1] + chr(int(FINAL_LOCK_MAP[last_char], 16))

    return word

def apply_final_locking_rule(text):
    words = text.split()
    words = [apply_final_locking(w) for w in words]
    return " ".join(words)


# ================= HÀM CHUYỂN TỰ =================

def encode_custom(text):
    text = text.lower()
    text = remove_tone_marks(text)

    text = apply_final_locking_rule(text)
    text = apply_virtual_onset_rule(text)
    text = apply_reorder_rule(text)


    i = 0
    output = ""

    while i < len(text):
        # ưu tiên cụm 3 ký tự
        if i + 3 <= len(text) and text[i:i+3] in RULES_3:
            code = RULES_3[text[i:i+3]]
            output += chr(int(code, 16))
            i += 3
            continue

        # cụm 2 ký tự
        if i + 2 <= len(text) and text[i:i+2] in RULES_2:
            code = RULES_2[text[i:i+2]]
            output += chr(int(code, 16))
            i += 2
            continue

        # 1 ký tự
        ch = text[i]
        if ch in RULES_1:
            code = RULES_1[ch]
            output += chr(int(code, 16))
        else:
            output += ch  # giữ nguyên ký tự lạ (dấu cách, số, ...)
        i += 1

    return output
