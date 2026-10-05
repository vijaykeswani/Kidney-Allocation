"""
End-of-study questionnaire, defined as data so the template stays generic and
wording lives in one place.

Wording choice to confirm: the brief offered three phrasings for the
"decision process" language ([your own / you want to use / you think you
should use]); this uses "you want to use" / "you want to make" throughout.
Change DP_* below to switch it everywhere at once.
"""

DP_PROCESS = "you want to use"          # "the process [you want to use] to make decisions"
DP_PROCESS_NOUN = "the decision process you want to use"
DP_CHOICE = "you want to make"          # "confidence in what choice [you want to make]"

LIKERT7_SIMILAR = ("Not at all similar", "Completely similar")
LIKERT7_TRUST = ("Not at all", "Completely")
IMPACT5 = ("Not at all", "A very big impact")

CHATBOT_FREQ = [
    ("never", "Never"),
    ("rarely", "Less than once a month"),
    ("monthly", "A few times a month"),
    ("weekly", "A few times a week"),
    ("daily", "About once a day"),
    ("many_daily", "Several times a day"),
]

GENDERS = [
    ("woman", "Woman"),
    ("man", "Man"),
    ("nonbinary", "Non-binary"),
    ("self_describe", "Prefer to self-describe"),
    ("no_answer", "Prefer not to say"),
]

EXPERIENCES = [
    ("seen_predictions", "Seeing how the AI model would predict your choices", ("treatment", "control")),
    ("interact", "Interacting with the visualization of the AI model to better understand how it predicts your choices", ("treatment",)),
    ("own_scenarios", "Creating your own scenarios to explore what choices the AI model would predict you would make", ("treatment",)),
    ("modify", f"Modifying the visualization of the AI model to make it better reflect {DP_PROCESS_NOUN}", ("treatment",)),
]

IMPACT_QUESTIONS = [
    ("impact_understand_model", "To what degree did each of these experiences impact how well you understood the model?"),
    ("impact_trust_final", "To what degree did each of these experiences impact how well you trust the model to help you make decisions, when you are the final decision-maker?"),
    ("impact_trust_behalf", "To what degree did each of these experiences impact how well you trust the model to make decisions on your behalf when you cannot be consulted?"),
    ("impact_understand_process", f"To what degree did each of these experiences impact your understanding of the process {DP_PROCESS} to make decisions in these contexts?"),
    ("impact_confidence", f"To what degree did each of these experiences impact your confidence in what choice {DP_CHOICE} in these contexts?"),
]


def build_survey(arm):
    """Ordered list of sections; each has 'title', optional 'intro', and 'items'.
    Item kinds: likert (scale 1..n with end labels), choice, number, matrix."""
    exps = [(k, label) for k, label, arms in EXPERIENCES if arm in arms]
    sections = [
        {"title": "How you decide", "items": [
            {"id": "similarity", "kind": "likert", "n": 7, "anchors": LIKERT7_SIMILAR,
             "text": "How similar do you think the way you actually do make decisions about which patient should get a kidney is to the way you wish you would make decisions about which patient should get a kidney?"},
        ]},
        {"title": "The AI model you saw",
         "intro": "Consider the AI model we showed you earlier that predicts who you would choose to get a kidney when one is available.",
         "items": [
            {"id": "trust_stressed", "kind": "likert", "n": 7, "anchors": LIKERT7_TRUST,
             "text": "How much would you trust this AI model to help you decide who should get a kidney in situations when you are stressed or don't have a lot of time?"},
            {"id": "trust_behalf_override", "kind": "likert", "n": 7, "anchors": LIKERT7_TRUST,
             "text": "How much would you trust this AI model to make decisions about who should get a kidney on your behalf, if you are able to override the decisions?"},
        ]},
        {"title": "AI chatbots",
         "intro": "General-purpose AI chatbots are systems that you can interact with conversationally through text or voice for a variety of tasks, such as ChatGPT, Claude, or Gemini.",
         "items": [
            {"id": "chatbot_freq", "kind": "choice", "options": CHATBOT_FREQ,
             "text": "How often do you use general-purpose AI chatbots?"},
            {"id": "chatbot_trust", "kind": "likert", "n": 7, "anchors": LIKERT7_TRUST,
             "text": "How much would you trust your favorite general-purpose AI chatbot to make decisions like this on your behalf?"},
        ]},
        {"title": "Reflecting on this study",
         "intro": "Take a moment to reflect on the experiences you had in this study.",
         "items": [
            {"id": qid, "kind": "matrix", "n": 5, "anchors": IMPACT5, "rows": exps, "text": qtext}
            for qid, qtext in IMPACT_QUESTIONS
        ]},
        {"title": "About you", "items": [
            {"id": "age", "kind": "number", "min": 18, "max": 120, "text": "What is your age?"},
            {"id": "gender", "kind": "choice", "options": GENDERS, "text": "What is your gender?",
             "other": {"when": "self_describe", "id": "gender_self_describe", "placeholder": "Please describe"}},
        ]},
    ]
    return sections


def collect_answers(sections, form):
    out = {}
    for sec in sections:
        for it in sec["items"]:
            if it["kind"] == "matrix":
                out[it["id"]] = {k: form.get(f"{it['id']}__{k}") for k, _ in it["rows"]}
            else:
                out[it["id"]] = form.get(it["id"])
                if it.get("other"):
                    out[it["other"]["id"]] = (form.get(it["other"]["id"]) or "").strip()[:200]
    return out