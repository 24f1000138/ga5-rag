"""Safety layer for GA5 Q4 "Harness Engineering: Contextual Safety and Model Routing".

Each case is classified on its own (pure function, no shared state), so an
approval or instruction in one case can never leak into another.

Plain-terms overview:
* requestDecision: is the USER's request itself OK?  Harmful goals -> block,
  unclear / high-impact goals -> review, everything else -> allow.
* toolDecision: is the proposed tool call safe given where it really runs?
  We trust `proposedTool.resolved` (operation, target, environment,
  sensitivity, disposable) and the environment's isolation record.  Text
  inside tool results, memory or repository files never grants approval.
* route: primary model for normal work, restricted (weaker security model)
  for sensitive-but-allowed work, none when the request is blocked.
* trajectory: normal, watch (drifting / suspicious content seen), or
  intervene (the session is heading outside its authorised scope).
"""
from __future__ import annotations

import json
import re
from typing import Any

# --------------------------------------------------------------- helpers

def _s(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _get(d: Any, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _has(pats, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


# ---------------------------------------------------------- vocabularies

HARMFUL = [
    r"\b(ransomware|keylogger|malware|botnet|rootkit|trojan|worm)\b",
    r"\b(steal|exfiltrat\w*|harvest|dump)\b.{0,40}\b(credential|password|token|secret|cookie|key|ssh|api key|pii|ssn|customer data|database)",
    r"\bphishing\b", r"\bcredential stuffing\b", r"\bbrute[- ]?force\b.{0,30}\b(login|password|account)",
    r"\b(ddos|denial[- ]of[- ]service)\b",
    r"\b(disable|bypass|evade|turn off|tamper with)\b.{0,40}\b(audit|logging|logs|edr|antivirus|monitoring|security|mfa|2fa|guardrail|safety)",
    r"\b(cover (our|my) tracks|erase (the )?logs|delete (the )?audit)",
    r"\bbackdoor\b", r"\breverse shell\b", r"\bprivilege escalation\b.{0,40}\b(without|unauthori[sz]ed)",
    r"\b(dox|stalk|track (her|his|their) location)\b",
    r"\b(weapon|explosive|bomb|nerve agent|bioweapon)\b",
    r"\bwithout (their|the user'?s?|the owner'?s?) (consent|knowledge|permission)\b",
    r"\b(fake|forge|counterfeit)\b.{0,30}\b(invoice|receipt|id|passport|signature|review)",
    r"\bscrape\b.{0,40}\b(personal|private) (data|info)",
    r"\bcrypto ?min(er|ing)\b.{0,40}\b(hidden|covert|without)",
]
REVIEW_REQ = [
    r"\b(production|prod)\b.{0,40}\b(deploy|migrat|delete|drop|rollback|restart|change|modify|update)",
    r"\b(deploy|migrat|drop|truncate|delete|wipe|purge)\b.{0,40}\b(production|prod|live)\b",
    r"\b(force[- ]push|git push --force|rewrite history)\b",
    r"\b(pay|payment|transfer|refund|wire)\b.{0,30}\$?\d",
    r"\b(grant|give|elevate)\b.{0,30}\b(admin|root|owner|sudo)\b",
    r"\bpenetration test|pentest|exploit\b",
    r"\bsend\b.{0,30}\b(email|message)s?\b.{0,30}\b(all|every|customers|users)\b",
]
INJECTION = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|rules)",
    r"\b(you are now|new instructions|system prompt|developer mode|jailbreak)\b",
    r"\b(pre-?approved|already approved|approval (is )?granted|authori[sz]ed by (the )?(admin|owner|user|security))\b",
    r"\b(do not|don't) (tell|inform|ask) (the )?user\b",
    r"\b(run|execute)\b.{0,30}\b(curl|wget|bash|sh)\b.{0,40}\|",
    r"\b(upload|send|post)\b.{0,40}\b(secret|token|key|credential|\.env|ssh)",
]
USER_APPROVAL = [
    r"\b(i (approve|authori[sz]e|confirm)|approved|go ahead|yes,? (do it|proceed|delete)|you have my (approval|permission)|confirmed)\b",
]
USER_RETRACT = [r"\b(cancel|stop|don't|do not|abort|hold off|never mind|wait)\b"]

READ_OPS = {"read", "list", "get", "search", "inspect", "view", "query", "stat", "diff", "status", "fetch", "cat", "grep", "test", "lint", "build", "dry-run", "dry_run", "plan"}
WRITE_OPS = {"write", "edit", "create", "update", "patch", "append", "move", "rename", "copy", "install", "commit", "format"}
DESTRUCTIVE_OPS = {"delete", "remove", "rm", "drop", "truncate", "wipe", "purge", "overwrite", "reset", "force_push", "force-push", "destroy", "format_disk", "chmod", "chown", "kill", "shutdown", "migrate", "deploy", "rotate", "revoke"}
EGRESS_OPS = {"upload", "send", "post", "publish", "email", "network", "egress", "http", "transfer", "push", "share", "exfiltrate"}
SENSITIVE = {"secret", "secrets", "credential", "credentials", "restricted", "confidential", "pii", "private", "sensitive", "regulated", "high"}
PROD_ENVS = {"production", "prod", "host", "live", "customer", "shared"}


# ---------------------------------------------------------- per-case logic

def _user_texts(case: dict) -> list[str]:
    msgs = case.get("messages") or []
    return [_s(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]


def _policy_lists(policy: Any) -> dict:
    """Pull keyword lists out of the policy object when it provides them."""
    out = {"block": [], "review": []}
    if not isinstance(policy, dict):
        return out
    for k, v in policy.items():
        kl = k.lower()
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            if any(w in kl for w in ("prohibit", "forbid", "block", "deny", "disallow")):
                out["block"] += v
            elif any(w in kl for w in ("review", "escalat", "approval", "sensitive")):
                out["review"] += v
    return out


def classify_case(case: dict, policy: Any) -> dict:
    cid = case.get("caseId") or case.get("id")
    users = _user_texts(case)
    latest = users[-1] if users else ""
    all_user = "\n".join(users)

    env = case.get("environment") or {}
    iso = env.get("isolation") or {}
    tool = case.get("proposedTool") or None
    res = (tool or {}).get("resolved") or {}

    side_text = "\n".join(_s(case.get(k)) for k in ("memory", "repositoryInstructions", "toolResults"))
    injected = _has(INJECTION, side_text)

    plists = _policy_lists(policy)

    # ---------------- request
    req = "allow"
    if _has(HARMFUL, latest) or any(re.search(re.escape(p), latest, re.I) for p in plists["block"] if len(p) > 3):
        req = "block"
    elif _has(REVIEW_REQ, latest) or any(re.search(re.escape(p), latest, re.I) for p in plists["review"] if len(p) > 3):
        req = "review"

    # ---------------- tool
    if not tool:
        td = "none"
    else:
        op = str(res.get("operation") or "").lower()
        tenv = str(res.get("environment") or env.get("kind") or "").lower()
        sens = str(res.get("sensitivity") or "").lower()
        disposable = bool(res.get("disposable"))
        target = _s(res.get("target"))
        isolated = bool(iso.get("verified")) and not iso.get("hostMounts") and not iso.get("credentials")
        no_net = not iso.get("externalNetwork")
        user_ok = _has(USER_APPROVAL, latest) and not _has([r"\b(cancel|abort|stop|don't|do not)\b"], latest)
        args_text = _s(tool.get("arguments"))

        if req == "block":
            td = "block"
        elif op in EGRESS_OPS and (sens in SENSITIVE or re.search(r"secret|token|\.env|credential|id_rsa|\.ssh", target + args_text, re.I)):
            td = "block"
        elif injected and (op not in READ_OPS):
            td = "block"
        elif op in DESTRUCTIVE_OPS or re.search(r"\brm -rf\b|drop table|truncate|--force", args_text, re.I):
            if disposable and isolated and tenv not in PROD_ENVS and sens not in SENSITIVE:
                td = "allow"
            elif tenv in PROD_ENVS or sens in SENSITIVE or not isolated:
                td = "review" if user_ok and case.get("rollbackPlan") else "block"
            else:
                td = "review"
        elif op in WRITE_OPS:
            td = "allow" if (tenv not in PROD_ENVS and sens not in SENSITIVE) else "review"
        elif op in READ_OPS or not op:
            td = "review" if sens in {"secret", "secrets", "credential", "credentials"} and not no_net else "allow"
        elif op in EGRESS_OPS:
            td = "review" if sens not in {"public", ""} else "allow"
        else:
            td = "allow" if (disposable and isolated) else "review"
        if req == "review" and td == "allow" and op not in READ_OPS:
            td = "review"

    # ---------------- route
    if req == "block":
        route = "none"
    elif req == "review" or injected or str(res.get("sensitivity") or "").lower() in SENSITIVE:
        route = "restricted"
    else:
        route = "primary"

    # ---------------- trajectory
    if req == "block" or td == "block":
        traj = "intervene"
    elif injected or req == "review" or td == "review" or (len(users) > 1 and _has(REVIEW_REQ + HARMFUL, all_user)):
        traj = "watch"
    else:
        traj = "normal"

    return {"caseId": cid, "requestDecision": req, "toolDecision": td, "route": route, "trajectory": traj}


def classify_batch(body: dict) -> dict:
    policy = body.get("policy")
    out = []
    for case in body.get("cases") or []:
        if not isinstance(case, dict):
            continue
        try:
            out.append(classify_case(case, policy))
        except Exception:
            # Fail safe: never approve on an error.
            out.append({"caseId": case.get("caseId"), "requestDecision": "review",
                        "toolDecision": "review" if case.get("proposedTool") else "none",
                        "route": "restricted", "trajectory": "watch"})
    return {"profile": body.get("profile", "agent-harness-2026-09"), "batchId": body.get("batchId"), "results": out}


CONFIG = {
    "name": "ga5-harness-rules",
    "method": "deterministic rules over policy + resolved tool metadata",
    "labels": {"requestDecision": ["allow", "block", "review"], "toolDecision": ["allow", "block", "review", "none"],
               "route": ["primary", "restricted", "none"], "trajectory": ["normal", "watch", "intervene"]},
}
