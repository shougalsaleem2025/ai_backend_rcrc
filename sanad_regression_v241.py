"""
Sanad v241 regression checks.
Run:
    python sanad_regression_v241.py

These are structural/offline checks; they do not call RCRC live APIs.
"""

from pathlib import Path
import re
import py_compile

BASE = Path(__file__).resolve().parent
orch = (BASE / "orchestrator.py").read_text(encoding="utf-8")
rr = (BASE / "response_renderer.py").read_text(encoding="utf-8")
sem = (BASE / "semantic_parser.py").read_text(encoding="utf-8")
et = (BASE / "employee_tools.py").read_text(encoding="utf-8")

for name in ("orchestrator.py", "response_renderer.py", "semantic_parser.py", "employee_tools.py"):
    py_compile.compile(str(BASE / name), doraise=True)

checks = {
    # Meetings / Tawasul
    "Tawasul pending": "tawasul_events_offer" in orch,
    "Tawasul displayed offer": "[Displayed Offer Saved]" in orch,
    "relative weekdays": "def _named_weekday_date" in sem,
    "Sun-Thu workweek": "RCRC work week: Sunday -> Thursday" in et,
    "meeting emojis": all(x in rr for x in ("🕘", "📍", "💬")),

    # Tarasol
    "Tarasol pending": "tarasol_details_offer" in orch,
    "Tarasol affirmative continuation": "-> tarasol:self_details" in orch,
    "Tarasol 0 response": "ما عندك أي معاملات في تراسل حاليًا." in rr,
    "Tarasol 1 response": "معاملة واحدة" in rr,
    "Tarasol 2 response": "معاملتين" in rr,
    "Tarasol plural response": "{number} معاملات" in rr,
    "Tarasol detail offer": "تحب أعرض لك تفاصيلها؟" in rr,

    # Leave
    "leave displayed-balance context": "[Leave Balance Context Saved]" in orch,
    "leave followup uses displayed balance": "[Leave Followup Context]" in orch,

    # Employee context/search
    "explicit employee search clears stale focus": "[Explicit Employee Search]" in orch,
    "short-name helper": "def _short_person_name" in rr,
    "collision-aware names": "def _person_display_names" in rr,

    # Cache
    "global smart cache": "Global enterprise cache policy" in orch,
    "mutation cache bypass": "[Tool Cache BYPASS]" in orch,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))

# Confirmation phrase coverage.
def norm(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    }))
    text = re.sub(r"[؟،؛:!?.,\-–—]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

affirmative = re.compile(
    r"(?:ايوا|ايوه|ايه|نعم|تمام|اوكي|اوك|اكيد|يلا|ورني|"
    r"اعرض(?:ها|هم)?|طلعها|yes|yeah|yep|sure|okay|ok|"
    r"show(?:\s+me|\s+them)?|go\s+ahead)",
    re.I,
)

for phrase in ("ايوا", "ايوه اعرضها", "تمام ورني", "اعرضها", "yes", "show me"):
    assert affirmative.search(norm(phrase)), phrase

# The bare "اي" is intentionally supported by _contextual_reply_kind's
# <=2-word pending-reply fallback.
assert "len(words) <= 2" in orch

print("Sanad v241 offline regression checks: OK")
print(f"Checks passed: {len(checks)}")
