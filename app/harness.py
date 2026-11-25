"""Safety layer for GA5 Q4 "Harness Engineering: Contextual Safety and Model Routing".

Rules follow the policy object the grader sends ("agent-harness-2026-09").
Every case is classified on its own (pure function, no shared state), so an
approval or instruction in one case can never leak into another.

Plain-terms overview
--------------------
requestDecision  block  : the user asks to steal credentials, write destructive
                          malware, commit fraud, harass or hurt a real person.
                 review : security work whose ownership/scope is unclear.
                 allow  : everything else (incl. defensive analysis, translating
                          quoted harmful text, inspecting a malicious file).
toolDecision     Uses proposedTool.resolved (the REAL effect after path / mount /
                 redirect resolution), not the harmless-looking command string.
                 block  : user revoked approval, user forbade this kind of action
                          ("do not deploy", "without sharing", "edit only under X"),
                          skipped safety preflight, credentials outside task scope,
                          private data to an unapproved destination.
                 review : consequential action (production change, force push,
                          host admin, external send, unknown environment) without
                          an exact user approval + passed preflight + rollback plan.
                 allow  : reads / edits inside the workspace, disposable fixture
                          cleanup in a verified isolated container, exactly approved
                          actions with all prerequisites.
                 none   : no tool proposed.
route            none if request blocked, restricted for vulnerability / exploit /
                 malware work, otherwise primary.
trajectory       intervene if request or tool is blocked; watch if anything needs
                 review, security scope is unclear, or untrusted content tries to
                 redirect the session; otherwise normal.
Only real user messages can approve. Text in tool results, memory and repo files
never counts as approval.
"""
from __future__ import annotations

import json
import re
from typing import Any


def _s(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _any(pats, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in pats)


# ------------------------------------------------------------ vocabularies

HARM = [
    r"\bsteal\w*\b", r"\bexfiltrat\w*", r"\bharvest\w* (credentials|passwords|tokens)",
    r"\b(write|build|develop|create|make)\b[^.]*\b(malware|ransomware|keylogger|wiper)\b",
    r"\bforged?\b|\bforging\b|\bcounterfeit\b", r"\btrick (customers|people|users|victims)\b",
    r"\binjure\b|\bphysical(ly)? harm\b|\bkill (a|him|her|them)\b|\bpoison\b",
    r"\bharass\w*|\bthreatening messages\b|\bstalk\w*\b|\bdox\w*\b",
    r"\bphishing\b",
]
SECURITY_WORK = [r"vulnerab", r"\bexploit", r"malware", r"indicators? of compromise", r"penetration test", r"\bpentest", r"security[ _]test"]
UNCLEAR_SCOPE = [r"demonstrat\w*", r"\bexploit\w*", r"\bfind (a|the|any) vulnerab", r"penetration test", r"\bpentest"]

REVOKE = [r"\brevok\w*", r"\bwithdrawn\b", r"\bchanged my mind\b", r"\bnow forbidden\b",
          r"\bdo not (execute|run) (the pending|it)\b", r"\bpause execution\b", r"\bcancel (it|that|the)\b"]

# forbids any mutating action
FORBID_MUTATION = [r"\bdo not change anything\b", r"\bdon't change anything\b", r"\bwithout (editing|changing|modifying)\b",
                   r"\bdo not (deploy|apply|execute|delete|run|modify|restart|push|merge)\b",
                   r"\bkeep [^.]{0,40}\bintact\b", r"\bread[- ]only\b"]
# forbids external sends
FORBID_SEND = [r"\bwithout sharing\b", r"\bdo not (send|share|upload|publish)\b", r"\blocally\b", r"\bkeep [^.]{0,30}private\b"]

APPROVE = [r"\bapprov\w*", r"\bauthori[sz]\w*", r"\bpermission\b"]

MUTATING = {"delete", "workspace_edit", "deploy", "database_mutation", "terraform_apply", "restart", "rollback",
            "force_push", "host_admin", "send", "dns_change", "privileged_change", "install", "write"}
PROD_OPS = {"deploy", "database_mutation", "terraform_apply", "restart", "rollback", "dns_change", "migration",
            "scale", "config_change"}
APPROVAL_OPS = {"force_push", "host_admin", "dns_change", "privileged_change"}
READ_OPS = {"workspace_read", "read", "read_public_schema", "list", "git_read", "inspect", "terraform_plan_read"}


def _dequote(t: str) -> str:
    """Remove quoted material (e.g. a threat message the user wants translated)."""
    t = re.sub(r":\s*'[^']*'", ": <quoted>", t)
    t = re.sub(r'"[^"]*"', "<quoted>", t)
    t = re.sub(r"[“‘][^”’]*[”’]", "<quoted>", t)
    return t


def _active_sentences(t: str) -> str:
    """Drop sentences that are negations ("Do not build the malware.")."""
    parts = re.split(r"(?<=[.!?;])\s+", t)
    return " ".join(p for p in parts if not re.match(r"\s*(do not|don't|never|without)\b", p, re.I))


def _in_ws(path: str, ws: str) -> bool:
    if not path or not ws:
        return False
    if ".." in path.split("/"):
        return False
    ws = ws.rstrip("/")
    return path == ws or path.startswith(ws + "/")


def _mentions(text: str, token: str) -> bool:
    """token appears in text as a whole item (not as a prefix of e.g. 'svc-other')."""
    if not token:
        return False
    return re.search(re.escape(token) + r"(?![\w\-/]|\.\w)", text, re.I) is not None


def classify_case(case: dict, policy: Any = None) -> dict:
    cid = case.get("caseId") if case.get("caseId") is not None else case.get("id")
    msgs = case.get("messages") or []
    users = [_s(m.get("content")) for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    latest = users[-1] if users else ""
    all_user = "\n".join(users)
    req_text = _active_sentences(_dequote(all_user))

    env = case.get("environment") or {}
    iso = env.get("isolation") or {}
    ws = _s(env.get("workspace") or env.get("cwd"))
    tool = case.get("proposedTool") if isinstance(case.get("proposedTool"), dict) else None
    res = (tool or {}).get("resolved") or {}
    op = _s(res.get("operation")).lower()
    target = _s(res.get("target"))
    renv = _s(res.get("environment") or "").lower()
    sens = _s(res.get("sensitivity")).lower()
    tool_results = [t for t in (case.get("toolResults") or []) if isinstance(t, dict)]

    # untrusted content trying to steer the session (never a source of approval)
    redirect = any(_s(t.get("origin")).lower() == "untrusted" for t in tool_results)
    side = " ".join(_s(x.get("text") if isinstance(x, dict) else x)
                    for k in ("memory", "repositoryInstructions") for x in (case.get(k) or []))
    if _any([r"approv", r"authori[sz]", r"ignore (all |the )?(previous|prior|user)", r"override"], side):
        redirect = True

    # ------------------------------------------------------------ request
    security = _any(SECURITY_WORK, req_text) or op == "security_test"
    owned_confined = (res.get("ownership") == "verified-owned" and bool(res.get("confined"))) or (
        not tool and re.search(r"\bowned\b", req_text, re.I) and re.search(r"isolated|confined|disposable", req_text, re.I))
    unclear = bool(_any(UNCLEAR_SCOPE, req_text)) and not owned_confined
    if op == "security_test" and not owned_confined:
        unclear = True

    if _any(HARM, req_text):
        req = "block"
    elif unclear:
        req = "review"
    else:
        req = "allow"

    # ------------------------------------------------------------ tool
    td = "none"
    if tool:
        td = _tool_decision(case, users, latest, all_user, env, iso, ws, res, op, target, renv, sens, tool_results)
        if req == "block":
            td = "block"
        elif req == "review" and td != "block":
            td = "review"

    # ------------------------------------------------------------ route
    if req == "block":
        route = "none"
    elif security or unclear:
        route = "restricted"
    else:
        route = "primary"

    # ------------------------------------------------------------ trajectory
    if req == "block" or td == "block":
        traj = "intervene"
    elif req == "review" or td == "review" or unclear or redirect:
        traj = "watch"
    else:
        traj = "normal"

    return {"caseId": cid, "requestDecision": req, "toolDecision": td, "route": route, "trajectory": traj}


def _approval_msgs(users):
    """User messages that grant approval (not revocations, not 'has not been approved')."""
    return [u for u in users if _any(APPROVE, u) and not _any(REVOKE, u)
            and not re.search(r"\b(has|have|is|was) not been approved\b|\bnot approved\b", u, re.I)]


def _tool_decision(case, users, latest, all_user, env, iso, ws, res, op, target, renv, sens, tool_results) -> str:
    # 1. revocation in the latest user message overrides everything earlier
    if _any(REVOKE, latest):
        return "block"
    # 2. skipping a required safety preflight
    if res.get("bypassPreflight") or res.get("skipPreflight"):
        return "block"

    origin_path = _s(res.get("originPath") or res.get("archiveEntry"))
    outside = bool(target) and target.startswith("/") and not _in_ws(target, ws)
    traversal = ".." in origin_path

    # 3. user explicitly forbade this kind of action
    if op in MUTATING and op != "send" and _any(FORBID_MUTATION, all_user):
        # a later explicit authorization supersedes an earlier limitation
        later_ok = len(users) > 1 and re.search(r"supersed|now explicitly authori", latest, re.I)
        if not later_ok:
            return "block"
    if op in ("workspace_edit", "delete", "write", "extract") and (outside or traversal) and \
            re.search(r"\bonly\b[^.]*\b(under|inside|within|in)\b", all_user, re.I):
        return "block"

    approvals = _approval_msgs(users)
    appr_text = "\n".join(approvals)

    # ---- secrets
    if op in ("read_secret", "read_credential", "credential_read"):
        if res.get("taskScoped") is True and _in_ws(target, ws) and re.search(r"task[- ]scoped", all_user, re.I) \
                and re.search(r"approv|authori|configured", all_user, re.I):
            return "allow"
        return "block"

    # ---- reads
    if op in READ_OPS or op.endswith("_read") or op.startswith("read_"):
        if op == "read_public_schema" or sens in ("public", "internal", ""):
            if op == "workspace_read" and target.startswith("/") and not _in_ws(target, ws):
                return "review"
            return "allow"
        return "review"

    # ---- external sends
    if op in ("send", "upload", "publish", "email", "http_post", "transfer"):
        if _any(FORBID_SEND, all_user):
            return "block"
        dest_named = _mentions(all_user, target) or (target.startswith("mailto:") and _mentions(all_user, target[7:]))
        if sens in ("public", ""):
            return "allow" if dest_named else "review"
        # private / secret data: only to the exact approved destination
        if dest_named and _mentions(appr_text, target):
            return "allow"
        return "block"

    # ---- workspace edits
    if op in ("workspace_edit", "write", "extract"):
        if _in_ws(target, ws) and not traversal:
            return "allow"
        return "review"

    # ---- deletes
    if op == "delete":
        isolated = (renv == "container" and bool(iso.get("verified")) and not iso.get("hostMounts")
                    and not iso.get("credentials") and not iso.get("externalNetwork"))
        if isolated and res.get("disposable") is True and _in_ws(target, ws):
            return "allow"
        if _in_ws(target, ws) and re.search(r"\bdelete\b", all_user, re.I) and _mentions(all_user, target) \
                and renv not in ("production", "unknown", ""):
            return "allow"
        return "review"

    # ---- security testing
    if op == "security_test":
        return "allow" if (res.get("ownership") == "verified-owned" and res.get("confined")) else "review"

    # ---- production mutations: exact approval + passed preflight + exact rollback plan
    if op in PROD_OPS or renv == "production":
        if renv in ("unknown", "") or not renv:
            return "review"
        authorized = any(_mentions(a, target) and re.search(r"\b" + re.escape(renv) + r"\b", a, re.I)
                         and not re.search(r"\bstaging\b", a, re.I) if renv != "staging" else _mentions(a, target)
                         for a in approvals)
        if not authorized:
            return "review"
        if renv == "production":
            pre_ok = any(_s(t.get("tool")).lower() == "preflight" and _s(t.get("target")) == target
                         and _s(t.get("status")).lower() in ("passed", "success", "ok")
                         and _s(t.get("environment") or renv).lower() == renv for t in tool_results)
            rb = case.get("rollbackPlan") or {}
            rb_ok = isinstance(rb, dict) and _s(rb.get("target")) == target and _s(rb.get("environment") or renv).lower() == renv
            if not (pre_ok and rb_ok):
                return "review"
        return "allow"

    # ---- other consequential actions needing exact approval
    if op in APPROVAL_OPS or res.get("privileged"):
        if any(_mentions(a, target) for a in approvals):
            return "allow"
        return "review"

    if renv in ("unknown", ""):
        return "review"
    return "review"


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
    "version": 2,
    "method": "deterministic rules over the agent-harness-2026-09 policy and resolved tool metadata",
    "labels": {"requestDecision": ["allow", "block", "review"], "toolDecision": ["allow", "block", "review", "none"],
               "route": ["primary", "restricted", "none"], "trajectory": ["normal", "watch", "intervene"]},
}
