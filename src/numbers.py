"""Turn spoken numbers in STT transcripts into digits (English + romanised Hindi).

Gnani STT writes what was said: "eight nine seven eight two eight three seven",
"four thousand five hundred thirty three", "saat hazaar nau sau chhattees teen hazaar ...".
Order IDs, phone digits, pincodes and dates must reach the agent as digits, so we convert:

    "order number hai eight nine seven eight two eight three seven" -> "order number hai 89782837"
    "last four digits four thousand five hundred thirty three"     -> "last 4 digits 4533"
    "seventeenth october twenty twenty six"                        -> "17 october 2026"

A run of number words is parsed into groups; groups that cannot combine (e.g. "seven eight",
"chhattees teen hazaar") are written next to each other, which is how people read IDs aloud.
Ambiguous Hindi words ("do", "ek", "teen", "das" ...) are only converted inside a run of
two or more number words, so ordinary sentences are left alone.
"""
from __future__ import annotations

import re

EN_UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
EN_ORD = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
    "thirtieth": 30,
}
_HI = """ek 1|do 2|teen 3 tin 3|char 4 chaar 4|paanch 5 panch 5 paach 5|chhe 6 chhah 6 chhah 6 che 6|saat 7 sat 7|
aath 8 ath 8|nau 9 nao 9|das 10|gyarah 11 gyaarah 11|barah 12 baarah 12|terah 13|chaudah 14|pandrah 15 pandra 15|
solah 16|satrah 17 satra 17|atharah 18 athaarah 18|unnis 19 unees 19 unnees 19|bees 20 bis 20|ikkis 21 ikkees 21|
bais 22 baees 22 baais 22|teis 23 teyees 23|chaubis 24 chaubees 24|pachchis 25 pachees 25 pachis 25|
chhabbis 26 chhabbees 26|sattais 27 sattaees 27|atthais 28 atthaees 28|untis 29 unatees 29|tees 30 tis 30|
iktis 31 ikattees 31|battis 32 battees 32|taintis 33 tetees 33|chauntis 34 chautees 34|paintis 35 paintees 35|
chhattis 36 chhattees 36|saintis 37 saintees 37|adtis 38 artees 38 adtees 38|untalis 39 unchaalees 39|
chalis 40 chaalees 40 chaalis 40|iktalis 41 iktaalees 41|bayalis 42 bayaalees 42|taintalis 43 taintaalees 43|
chavalis 44 chavaalees 44|paintalis 45 paintaalees 45|chhiyalis 46 chhiyaalees 46|saintalis 47 saintaalees 47|
adtalis 48 adtaalees 48|unchas 49 unanchaas 49|pachas 50 pachaas 50|ikyavan 51|bavan 52 baavan 52|tirpan 53|
chauvan 54|pachpan 55|chhappan 56|sattavan 57|atthavan 58|unsath 59|saath 60|iksath 61|basath 62|tirsath 63|
chausath 64|painsath 65|chhiyasath 66|sadsath 67|adsath 68|unhattar 69|sattar 70|ikhattar 71|bahattar 72|
tihattar 73|chauhattar 74|pachhattar 75|chhihattar 76|satattar 77|athattar 78|unasi 79|assi 80|ikyasi 81|
bayasi 82|tirasi 83|chaurasi 84|pachasi 85|chhiyasi 86|satasi 87|athasi 88|nawasi 89|nabbe 90|ikyanve 91|
banve 92|tiranve 93|chauranve 94|pachanve 95|chhiyanve 96|sattanve 97|athanve 98|ninyanve 99"""
HI_UNITS = {}
for part in _HI.replace("\n", "").split("|"):
    toks = part.split()
    for w, v in zip(toks[::2], toks[1::2]):
        HI_UNITS[w] = int(v)
# Hindi words that are also common words: only convert inside a run of number words
HI_AMBIGUOUS = {"do", "ek", "teen", "tin", "char", "das", "bees", "bis", "tees", "tis", "sat", "saat", "saath",
                "nao", "ath", "che", "bais"}
MULT = {"hundred": 100, "sau": 100, "thousand": 1000, "hazaar": 1000, "hazar": 1000, "lakh": 100000,
        "lac": 100000}
UNITS = {**EN_UNITS, **HI_UNITS}


def _value(tok):
    if tok in UNITS:
        return UNITS[tok], False
    if tok in EN_ORD:
        return EN_ORD[tok], True
    return None, False


def _parse_run(tokens: list[str]) -> str:
    groups, total, cur = [], 0, 0

    def flush():
        nonlocal total, cur
        if total or cur:
            groups.append(total + cur)
        total = cur = 0

    for t in tokens:
        if t in MULT:
            m = MULT[t]
            if m == 100:
                cur = (cur or 1) * 100
            else:
                if total and total < m * 1000 and total % (m * 10) >= m and cur:  # second "thousand" in a run
                    flush()
                total += (cur or 1) * m
                cur = 0
            continue
        v, _ = _value(t)
        if v == 0:            # "zero"/"oh" is always a spoken digit: keep it as its own group
            flush()
            groups.append(0)
            continue
        can_add = cur == 0 or (cur % 100 == 0 and v < 100) or (cur % 10 == 0 and cur % 100 != 0 and v < 10)
        if total and cur == 0 and total % 1000 == 0 and v >= 0:
            can_add = True
        if not can_add:
            flush()
        cur += v
    flush()
    return "".join(str(g) for g in groups)


def _is_num(tok):
    return tok in UNITS or tok in EN_ORD or tok in MULT


def spoken_numbers_to_digits(text: str) -> str:
    if not text:
        return text
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*|\S", text)
    out, i = [], 0
    while i < len(tokens):
        low = tokens[i].lower()
        parts = low.split("-")
        if all(_is_num(p) for p in parts):
            run = []
            while i < len(tokens) and all(_is_num(p) for p in tokens[i].lower().split("-")):
                run += tokens[i].lower().split("-")
                i += 1
            starts_with_mult = run[0] in MULT
            single_ambiguous = len(run) == 1 and (run[0] in HI_AMBIGUOUS or run[0] in ("one", "oh", "second"))
            if starts_with_mult or single_ambiguous or (len(run) == 1 and run[0] in HI_UNITS and run[0] not in EN_UNITS):
                out += run if not single_ambiguous else [tokens[i - 1]]
                continue
            out.append(_parse_run([p for p in run if p not in ("and",)]))
            continue
        out.append(tokens[i])
        i += 1
    s = " ".join(out)
    return re.sub(r"\s+([.,!?;:])", r"\1", s)


if __name__ == "__main__":
    import sys
    print(spoken_numbers_to_digits(" ".join(sys.argv[1:])))
