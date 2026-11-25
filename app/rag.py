"""Retrieval + answer logic.

How a question is answered, in plain terms:
1. Clean the question: drop any sentence that tries to give orders
   ("ignore previous instructions", "say the answer is ...").
2. Find which systems the question is about (Raft, PostgreSQL, AdamW, ...)
   using a hand-made alias list, so "Raft" never gets Multi-Paxos's numbers.
3. Rank all 50 passages: dense embeddings (MiniLM) + a cross-encoder reranker
   + a boost for the passage whose subject the question names.
4. Work out what kind of value is asked for (port, ms timeout, ratio, count,
   bits, hash ...). Pull that value out of the best-matching sentence of the
   top passage with a regex. Text is only ever copied from the knowledge base,
   so an injected instruction cannot change the answer.
5. If the passage has no value of that kind, or nothing in the KB matches
   well enough, abstain: answer null, empty sources, low confidence.
6. If the question names two systems (or asks for two things), answer each
   part and combine them (and add/compare when the question asks).
"""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .models import Embedder, Reranker

KB_PATH = Path(__file__).resolve().parent.parent / "kb.json"
PASSAGES: list[str] = json.loads(KB_PATH.read_text())

# ---------------------------------------------------------------- subjects
# index -> alias regexes (matched case-insensitively, longest first)
ALIASES = {
    0: [r"raft"],
    1: [r"multi[- ]?paxos", r"paxos", r"chubby"],
    2: [r"epidemic gossip", r"gossip(?: protocols?)?", r"phi accrual(?: failure detector)?"],
    3: [r"swim(?: membership)?(?: protocol)?", r"consul"],
    4: [r"two[- ]phase commit", r"three[- ]phase commit", r"2pc", r"3pc"],
    5: [r"sagas?(?: pattern)?", r"cadence", r"temporal"],
    6: [r"consistent hash(?:ing)?(?: rings?)?", r"amazon dynamo", r"dynamo"],
    7: [r"rendezvous hashing", r"rendezvous", r"highest random weight(?: hashing)?", r"hrw"],
    8: [r"vector clocks?", r"version vectors?", r"riak(?: kv)?"],
    9: [r"hybrid logical clocks?", r"hlcs?"],
    10: [r"stochastic gradient descent", r"sgd"],
    11: [r"adamw", r"adam", r"decoupled weight decay"],
    12: [r"transformer encoders?", r"encoders?", r"encoder[- ]only"],
    13: [r"(?:autoregressive )?(?:transformer )?decoders?", r"decoder[- ]only", r"autoregressive"],
    14: [r"(?:binary )?logistic regression"],
    15: [r"support vector machines?", r"svms?"],
    16: [r"multi[- ]?layer perceptrons?", r"perceptrons?", r"mlps?", r"feedforward classifier"],
    17: [r"residual (?:convolutional )?networks?", r"resnets?", r"convnext", r"convolutional (?:neural )?networks?", r"cnns?"],
    18: [r"batch[- ]?norm(?:alization)?"],
    19: [r"layer[- ]?norm(?:alization)?"],
    20: [r"postgre(?:s|sql)", r"pgbouncer"],
    21: [r"mysql(?: innodb)?", r"innodb"],
    22: [r"b\+ ?trees?", r"b-trees?", r"btrees?"],
    23: [r"log[- ]structured merge(?:[- ]trees?)?", r"lsm(?:[- ]trees?)?", r"rocksdb"],
    24: [r"mongo(?:db)?", r"wiredtiger"],
    25: [r"(?:apache )?cassandra"],
    26: [r"redis"],
    27: [r"memcached?"],
    28: [r"write[- ]ahead log(?:ging|s)?", r"wal", r"aries"],
    29: [r"shadow pag(?:ing|es?)"],
    30: [r"transmission control protocol", r"tcp"],
    31: [r"quic"],
    32: [r"domain name system", r"dns"],
    33: [r"multicast dns", r"mdns", r"bonjour", r"avahi"],
    34: [r"http/?2(?:\.0)?", r"hpack"],
    35: [r"http/?3", r"qpack"],
    36: [r"tls ?(?:v|version )?1\.3", r"transport layer security", r"tls"],
    37: [r"tls ?(?:v|version )?1\.2"],
    38: [r"border gateway protocol", r"bgp"],
    39: [r"open shortest path first", r"ospf"],
    40: [r"public[- ]key cryptography", r"asymmetric (?:key )?cryptography", r"rsa"],
    41: [r"elliptic[- ]curve(?: cryptography)?", r"ecc", r"ecdsa", r"ed25519"],
    42: [r"(?:secure )?credential storage", r"password (?:hashing|storage)", r"argon2", r"salting"],
    43: [r"pbkdf2"],
    44: [r"web application firewalls?", r"wafs?"],
    45: [r"api gateways?", r"token buckets?(?: rate limiting)?", r"kong"],
    46: [r"advanced encryption standard", r"aes(?:[- ]gcm)?"],
    47: [r"chacha20(?:-poly1305)?", r"chacha", r"poly1305"],
    48: [r"oauth(?: 2\.0)?(?: pkce)?", r"pkce", r"proof key for code exchange"],
    49: [r"json web tokens?", r"jwts?"],
}
NAMES = {
    0: "Raft", 1: "Multi-Paxos", 2: "Gossip (phi accrual)", 3: "SWIM", 4: "Two-phase commit",
    5: "Saga", 6: "Consistent hashing", 7: "Rendezvous hashing", 8: "Vector clocks",
    9: "Hybrid logical clocks", 10: "SGD", 11: "AdamW", 12: "Transformer encoder",
    13: "Transformer decoder", 14: "Logistic regression", 15: "SVM", 16: "MLP", 17: "ResNet",
    18: "Batch normalization", 19: "Layer normalization", 20: "PostgreSQL", 21: "MySQL InnoDB",
    22: "B+ Tree", 23: "LSM tree", 24: "MongoDB", 25: "Cassandra", 26: "Redis", 27: "Memcached",
    28: "Write-ahead logging", 29: "Shadow paging", 30: "TCP", 31: "QUIC", 32: "DNS", 33: "mDNS",
    34: "HTTP/2", 35: "HTTP/3", 36: "TLS 1.3", 37: "TLS 1.2", 38: "BGP", 39: "OSPF",
    40: "Public key cryptography (RSA)", 41: "Elliptic curve cryptography", 42: "Password storage",
    43: "PBKDF2", 44: "Web application firewall", 45: "API gateway", 46: "AES", 47: "ChaCha20-Poly1305",
    48: "OAuth PKCE", 49: "JWT",
}
_ALIAS_LIST = sorted(((a, i) for i, al in ALIASES.items() for a in al), key=lambda x: -len(x[0]))
_ALIAS_RE = [(re.compile(r"(?<![\w/+-])" + a + r"(?![\w+-])", re.I), i) for a, i in _ALIAS_LIST]


def find_subjects(text: str) -> list[tuple[int, int]]:
    """Return [(position, passage_index)] for subject mentions, longest alias wins."""
    taken = [False] * len(text)
    hits = []
    for rx, idx in _ALIAS_RE:
        for m in rx.finditer(text):
            if any(taken[m.start():m.end()]):
                continue
            for k in range(m.start(), m.end()):
                taken[k] = True
            hits.append((m.start(), idx))
    hits.sort()
    return hits


# ---------------------------------------------------------------- facts
FACT_RX = {
    "PORT": re.compile(r"\bport (\d{2,5})\b", re.I),
    "MS": re.compile(r"\b(\d+)\s?ms\b"),
    "BITS": re.compile(r"\b(\d+)(-bit| bits?)\b"),
    "HASH": re.compile(r"\b(SHA-256|SHA-3|BLAKE2)\b"),
    "CODE": re.compile(r"\bHTTP (\d{3})\b"),
    "DEC": re.compile(r"(?<![\d.])(0\.\d+)(?!\d|\.\d)"),
    "COUNT": re.compile(r"(?<![\w./-])(\d+)(?![\w./-]|\.\d|\s?ms\b| bits?\b)"),
}


def sentence_facts(sent: str) -> list[tuple[str, str, float | None]]:
    """[(type, display_value, numeric)] found in one sentence."""
    out, used = [], []
    for t in ["PORT", "MS", "BITS", "HASH", "CODE", "DEC", "COUNT"]:
        for m in FACT_RX[t].finditer(sent):
            if any(m.start() < e and s < m.end() for s, e in used):
                continue
            used.append((m.start(), m.end()))
            g = m.group(1)
            if t == "PORT":
                out.append((t, f"port {g}", float(g)))
            elif t == "MS":
                out.append((t, f"{g}ms", float(g)))
            elif t == "BITS":
                out.append((t, f"{g} bits", float(g)))
            elif t == "HASH":
                out.append((t, g, None))
            else:
                out.append((t, g, float(g)))
    return out


def split_sentences(p: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if s.strip()]


SENTS = [split_sentences(p) for p in PASSAGES]
FACTS = [[sentence_facts(s) for s in ss] for ss in SENTS]

# ---------------------------------------------------------------- question typing
TYPE_RX = {
    "PORT": r"\bports?\b|listen(?:s|ing)? on|binds? to",
    "CODE": r"status code|http code|error code|response code",
    "HASH": r"hash(?:ing)? (?:function|algorithm|primitive)s?|digest|\bsha\b|(?:which|what) hash\b|primitive",
    "BITS": r"\bbits?\b|key (?:size|length)s?|\bkey sizes?\b|how (?:long|large|big) (?:is|are) (?:the )?(?:\w+ )?keys?|security level|curve size|group size|\b(?:keys?|tags?|timestamps?)\b.*\b(?:size|length|long|large|big|wide)\b|\b(?:size|length|long|large|big|wide)\b.*\b(?:keys?|tags?|timestamps?)\b",
    "MS": r"time ?outs?|timer|heartbeats?|intervals?|latenc(?:y|ies)|duration|delays?|leases?|lease terms?|windows?|\bwait\b|deadline|\bms\b|millisec|how long|how quickly|how fast|clock skew|\bskew\b|budget|\blimit\b|\bperiod\b|\bevery\b|within|\btime\b|drift|how soon|after how",
    "DEC": r"coefficients?|\bfactor\b|scalar|multiplier|step size|thresholds?|ratio|probabilit(?:y|ies)|\brate\b|learning rate|\bskew\b|saturation|utili[sz]ation|fraction|metric|percent|\blimit\b|capacity|cut-?off|slack|dropout|correlation|variance|value|level",
    "COUNT": r"how many|number of|quorum|\bnodes\b|replicas|members|acceptors|participants|retries|blocks|layers|\bsize\b|sizes|samples|records|entries|tokens|bytes|chunks?|batch|count|instances|vectors|length|groups?|how much|\bminimum\b|stack|depth|deep",
}
TYPE_RX = {k: re.compile(v, re.I) for k, v in TYPE_RX.items()}

UNSUPPORTED = re.compile(
    r"\bwho\b|invent|created by|developed by|designed by|\bauthor(?:s|ed)?\b|what year|which year|when was|when did|"
    r"written in|programming language|licen[cs]e|\bprice\b|\bcost\b|founded|\bGB\b|\bMB\b|gigabyte|megabyte|"
    r"cpu cores|how much memory|ram (?:usage|size)|default (?:username|password)|replication factor|"
    r"maximum (?:number of )?connections|\b(?:max|maximum)\b.*\bmemory\b|\bmemory (?:limit|size)|\bstock\b|\bweather\b|\bcapital of\b|recipe|\bmovie|\bsong",
    re.I,
)


def mask_subjects(text: str) -> str:
    for rx, _ in _ALIAS_RE:
        text = rx.sub(" SUBJ ", text)
    return text


def question_types(text: str) -> set[str]:
    text = mask_subjects(text)
    ts = {t for t, rx in TYPE_RX.items() if rx.search(text)}
    if "BITS" in ts and re.search(r"key (?:size|length)", text, re.I):
        ts.discard("COUNT")
    if "PORT" in ts:  # "which port" is unambiguous
        ts = {"PORT"}
    if "CODE" in ts:
        ts = {"CODE"}
    return ts


STOP = set("""what which how many much does do did is are was were the a an of in on for to by use uses used using
value default set configured have has with at and or it its this that be number big large long small kind type
when where whose whom from per as into over under about can could would should will there their they them
what's whats tell me give please exactly only one word answer question system get gets get much more less""".split())


def _stem(w: str) -> str:
    return w.rstrip("s")[:5]


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", mask_subjects(text).lower().replace("subj", " "))
    return {_stem(w) for w in words if w not in STOP and len(w) > 2 and not w.isdigit()}


def overlaps(qwords: set[str], sent: str) -> bool:
    sw = {_stem(w) for w in re.findall(r"[a-z0-9]+", sent.lower()) if len(w) > 2}
    return bool(qwords & sw)


KB_VOCAB = {w.lower() for p in PASSAGES for w in re.findall(r"[A-Za-z][A-Za-z0-9+/-]*", p)}
KB_VOCAB |= {w.lower() for p in PASSAGES for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", p)}


def unknown_entities(q: str) -> list[str]:
    """Proper-noun-looking words (not sentence-initial) that the KB never mentions, e.g. Kafka."""
    masked = mask_subjects(q)
    out = []
    for m in re.finditer(r"(?<![.?!:]\s)(?<!^)\b([A-Z][A-Za-z0-9]*[A-Za-z0-9])\b", masked):
        w = m.group(1)
        if w == "SUBJ":
            continue
        pre = masked[: m.start()].rstrip()
        if not pre or pre[-1] in ".?!:\"'":
            continue
        base = re.sub(r"'s$", "", w).lower()
        if base in KB_VOCAB or base in STOP or base.rstrip("s") in KB_VOCAB:
            continue
        out.append(w)
    return out


QUANT_RX = re.compile(r"how many|how much|how long|how big|how large|\b(?:maximum|minimum|max|min|default)\b|\blimit\b|\bsize\b|\bnumber\b|\bvalue\b", re.I)


# ---------------------------------------------------------------- injection
INJ_RX = re.compile(
    r"ignore|disregard|forget (?:all|everything|previous|prior|the)|system prompt|\bsystem\s*:|\[?inst\]|"
    r"you are now|pretend|act as|roleplay|override|jailbreak|developer mode|new instructions?|"
    r"instead,? (?:say|answer|respond|output|return|reply|tell)|"
    r"(?:respond|reply|answer|output|return|print|say)(?: only)? (?:with|that|exactly|the word|\")|"
    r"set (?:the |your )?confidence|confidence (?:to|=|of|:)|(?:the )?(?:correct |real |true )?answer (?:is|should be|must be)|"
    r"always (?:say|answer)|reveal|do not (?:use|consult|look)|don't (?:use|consult)|"
    r"without (?:checking|consulting|looking)|regardless of|assistant\s*:|user\s*:|<\s*/?\s*(?:system|admin|instruction)",
    re.I,
)


def sanitize(q: str) -> tuple[str, bool]:
    """Remove instruction-like fragments. Returns (clean_question, had_injection)."""
    q = re.sub(r"<!--.*?(?:-->|$)", " ", q, flags=re.S)
    pieces = re.split(r"(?<=[.!?;])\s+|\n+|[()\[\]{}<>]|\s[-–—]{1,2}\s|\"|“|”", q)
    kept, inj = [], False
    for p in pieces:
        p = (p or "").strip()
        if not p:
            continue
        m = INJ_RX.search(p)
        if m:
            inj = True
            head = p[: m.start()].strip(" ,:;-")
            if len(head.split()) >= 3:
                kept.append(head)
            continue
        kept.append(p)
    return " ".join(kept).strip(), inj


PREMISE_RX = re.compile(
    r"^\s*(?:since|given(?: that)?|assuming(?: that)?|suppose(?: that)?|if|because|as|considering(?: that)?|knowing(?: that)?)\b[^,?]*,\s*"
    r"|,?\s*(?:since|given(?: that)?|assuming(?: that)?|because)\b[^,?]*(?=[,?]|$)",
    re.I,
)


def strip_premise(q: str) -> str:
    """'Since Raft requires 100 nodes, what is the heartbeat?' -> 'what is the heartbeat?'
    A claim stated in the question is not evidence; only the KB is."""
    out = PREMISE_RX.sub(" ", q).strip()
    return out if len(out.split()) >= 3 else q


# ---------------------------------------------------------------- engine
def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


SUM_RX = re.compile(r"\btotal\b|\bsum\b|combined|\btogether\b|\badd(?:ed|ing)?\b|\bplus\b|altogether", re.I)
DIFF_RX = re.compile(r"difference|how (?:much|many) (?:more|longer|higher|larger|greater|less|fewer|shorter|lower|smaller)|\bminus\b|subtract|\bgap\b", re.I)
MAX_RX = re.compile(r"which\b.*\b(?:higher|larger|longer|greater|bigger|more|slower|stricter|largest|highest|longest)\b", re.I)
MIN_RX = re.compile(r"which\b.*\b(?:lower|smaller|shorter|less|fewer|faster|smallest|lowest|shortest)\b", re.I)
RATIO_RX = re.compile(r"\bratio\b.*\bbetween\b|how many times|times (?:larger|more|bigger|greater|longer)", re.I)
COMPARE_RX = re.compile(r"\bvs\.?\b|versus|compare|compared|comparison|\bboth\b|\bbetween\b|\beach\b|differ|" + SUM_RX.pattern + "|" + DIFF_RX.pattern, re.I)
SAME_RX = re.compile(r"\bsame\b|identical|\bequal\b|match", re.I)
CLAUSE_SPLIT = re.compile(r"\band\b|,|;|\bvs\.?\b|\bversus\b|compared (?:to|with)|\bwhile\b|\bwhereas\b|\bor\b|\bthan\b", re.I)


N_CE = int(__import__("os").environ.get("N_CE", "5"))  # passages sent to the cross-encoder
FAST = __import__("os").environ.get("FAST", "1") == "1"  # cosine for sentences / named subjects


class RAG:
    def __init__(self):
        self.emb = Embedder()
        self.ce = Reranker()
        self.P = self.emb.encode(PASSAGES)
        self.S = [self.emb.encode(ss) for ss in SENTS]
        # warm up
        self.ce.score("warm up", PASSAGES[:2])

    @lru_cache(maxsize=4096)
    def qvec(self, q: str) -> np.ndarray:
        return self.emb.encode([q])[0]

    # ---- ranking
    def rank(self, q: str, subj: list[int]):
        qv = self.qvec(q)
        cos = np.maximum(self.P @ qv, np.array([float((S @ qv).max()) for S in self.S]))
        if subj and FAST:
            # the question names its subject: skip the slow cross-encoder, use cosine
            cand = sorted(set(np.argsort(-cos)[:N_CE].tolist()) | set(subj))
            ce = [10.0 * float(cos[i]) - 3.0 for i in cand]
        else:
            cand = sorted(set(np.argsort(-cos)[:N_CE].tolist()) | set(subj))
            ce = self.ce.score(q, [PASSAGES[i] for i in cand])
        rows = []
        for i, c in zip(cand, ce):
            boost = 4.0 if i in subj else 0.0
            final = float(c) + 3.0 * float(cos[i]) + boost
            rows.append((final, i, float(c), float(cos[i])))
        for i in range(len(PASSAGES)):  # the rest: cosine only, always ranked below
            if i not in cand:
                rows.append((3.0 * float(cos[i]) - 30.0, i, -12.0, float(cos[i])))
        rows.sort(reverse=True)
        return rows

    def best_sentence(self, q: str, pidx: int, types: set[str] | None, qwords: set[str] | None = None):
        sents = SENTS[pidx]
        if FAST and types:  # typed values: regex does the work, cosine picks the sentence
            sc = 10.0 * (self.S[pidx] @ self.qvec(q)) - 3.0
        else:
            sc = self.ce.score(q, sents)
        order = np.argsort(-sc)
        for j in order:
            facts = FACTS[pidx][j]
            if types:
                fs = [f for f in facts if f[0] in types]
                strong = fs and all(f[0] in ("PORT", "HASH", "CODE", "BITS") for f in fs)
                if not strong and qwords is not None and not overlaps(qwords, sents[j]):
                    continue
                if fs:
                    # prefer type order as listed by specificity
                    pri = ["PORT", "CODE", "HASH", "BITS", "MS", "DEC", "COUNT"]
                    fs.sort(key=lambda f: pri.index(f[0]))
                    return j, float(sc[j]), fs[0]
            else:
                return j, float(sc[j]), None
        return None, float(sc.max()) if len(sc) else -10.0, None

    # ---- main
    def answer(self, question: str, top_k: int = 3) -> dict:
        top_k = max(1, min(int(top_k or 3), 50))
        q, inj = sanitize(question or "")
        if len(q) < 3 or not re.search(r"[A-Za-z]{2}", q):
            return abstain()
        if UNSUPPORTED.search(q) or unknown_entities(q):
            return abstain()

        hits = find_subjects(q)
        subj_order = []
        for _, i in hits:
            if i not in subj_order:
                subj_order.append(i)

        # Build tasks: (subject or None, types, clause text)
        ask = strip_premise(q)
        clauses = [c.strip() for c in CLAUSE_SPLIT.split(ask) if c and c.strip()]
        tasks = []
        global_types = question_types(ask)
        for c in clauses:
            cs = [i for _, i in find_subjects(c)]
            ct = question_types(c)
            if not cs and not ct:
                continue
            tasks.append([cs, ct, c])
        # fill gaps: inherit subjects / types from neighbours
        for k, t in enumerate(tasks):
            if not t[0]:
                prev = next((tasks[j][0] for j in range(k - 1, -1, -1) if tasks[j][0]), None)
                nxt = next((tasks[j][0] for j in range(k + 1, len(tasks)) if tasks[j][0]), None)
                t[0] = (prev or nxt or [])[-1:] if prev else (nxt or [])[:1]
            if not t[1]:
                t[1] = global_types
        multi_subject = len(subj_order) >= 2 and (COMPARE_RX.search(q) or len({tuple(t[0]) for t in tasks if t[0]}) >= 2)
        ranking = self.rank(q, subj_order)
        if not multi_subject:
            main = [r[1] for r in ranking if r[1] in subj_order][:1]
            # one subject (or none): tasks differ only by type
            seen, merged = set(), []
            for t in tasks:
                key = frozenset(t[1])
                if key and key not in seen:
                    seen.add(key)
                    merged.append((main, t[1], t[2]))
            if len(merged) > 1 and any(not (a[1] & b[1]) for a in merged for b in merged if a is not b):
                tasks = merged
            else:
                tasks = [(main, global_types, ask)]
        else:
            # one task per subject, keep clause types
            per = {}
            for t in tasks:
                for s in t[0]:
                    if s not in per:
                        per[s] = (t[1] or global_types, t[2])
            for s in subj_order:
                per.setdefault(s, (global_types, q))
            tasks = [([s], per[s][0], per[s][1]) for s in subj_order]


        # ---- single subject, single type (most questions)
        results = []
        for subj, types, clause in tasks:
            if subj:
                pidx = subj[0]
                row = next(r for r in ranking if r[1] == pidx)
            else:
                row = ranking[0]
                pidx = row[1]
            # gate: is this passage relevant at all?
            if not subj and ((row[2] < 0.0 and row[3] < 0.5) or row[3] < 0.30):
                return abstain()
            if subj and row[2] < -9.0:
                return abstain()
            sq = ask if len(tasks) == 1 else clause + " " + ask
            j, sscore, fact = self.best_sentence(sq, pidx, types if types else None, content_words(ask))
            if types and fact is None:
                results.append((pidx, None, None, sscore))
                continue
            if not types:
                # descriptive question: need a clearly relevant sentence
                qw = content_words(ask)
                sw = {_stem(w) for w in re.findall(r"[a-z0-9]+", SENTS[pidx][j].lower()) if len(w) > 2}
                ratio = len(qw & sw) / max(1, len(qw))
                if row[2] < (1.0 if subj else 3.0) or sscore < -4.0 or QUANT_RX.search(q) or (ratio < 0.4 and not multi_subject):
                    results.append((pidx, None, None, sscore))
                    continue
                results.append((pidx, SENTS[pidx][j], None, sscore))
            else:
                results.append((pidx, fact[1], fact, sscore))

        good = [r for r in results if r[1] is not None]
        if not good or (len(results) == 1 and results[0][1] is None):
            return abstain()
        if len(good) < len(results) and len(good) == 0:
            return abstain()

        # ---- compose answer
        if len(results) == 1:
            ans = good[0][1]
        else:
            parts = []
            for pidx, val, fact, _ in results:
                parts.append(f"{NAMES[pidx]}: {val if val is not None else 'not stated'}")
            # same-subject multi-type -> simple join
            if len({r[0] for r in results}) == 1:
                ans = "; ".join(str(r[1]) for r in results if r[1] is not None)
            else:
                ans = "; ".join(parts)
                nums = [(r[0], r[2][2]) for r in good if r[2] and r[2][2] is not None]
                if len(nums) == len(results) and len(nums) >= 2:
                    unit = "ms" if all(r[2][0] == "MS" for r in good) else ""
                    fmt = lambda x: (f"{x:g}{unit}")
                    vals = [n for _, n in nums]
                    if SAME_RX.search(q):
                        ans = ("Yes" if len(set(vals)) == 1 else "No") + f" ({ans})"
                    elif RATIO_RX.search(q) and vals[1]:
                        ans = f"{vals[0] / vals[1]:g} ({ans})"
                    elif DIFF_RX.search(q):
                        ans = f"{fmt(abs(vals[0] - vals[1]))} ({ans})"
                    elif SUM_RX.search(q):
                        ans = f"{fmt(sum(vals))} ({ans})"
                    elif MAX_RX.search(q):
                        w = max(nums, key=lambda x: x[1])[0]
                        ans = f"{NAMES[w]} ({ans})"
                    elif MIN_RX.search(q):
                        w = min(nums, key=lambda x: x[1])[0]
                        ans = f"{NAMES[w]} ({ans})"

        # ---- sources & scores
        used = []
        for r in results:
            if r[1] is not None and r[0] not in used:
                used.append(r[0])
        rest = [r[1] for r in ranking if r[1] not in used]
        order = (used + rest)[:top_k]
        smap = {r[1]: r for r in ranking}
        scores, last = [], 1.0
        for k, i in enumerate(order):
            if i in used:
                s = 0.75 + 0.24 * _sig(max(r[3] for r in results if r[0] == i))
            else:
                f = smap.get(i, (0, i, -10, 0))
                s = 0.6 * _sig(f[2])
            s = min(s, last - 1e-4 if k else s)
            s = round(max(0.0, min(1.0, s)), 4)
            scores.append(s)
            last = s
        best = min(r[3] for r in good)
        conf = 0.7 + 0.27 * _sig(best / 2)
        if len(good) < len(results):
            conf = 0.62
        return {"answer": str(ans), "sources": order, "scores": scores, "confidence": round(conf, 4)}


def abstain() -> dict:
    return {"answer": None, "sources": [], "scores": [], "confidence": 0.1}
