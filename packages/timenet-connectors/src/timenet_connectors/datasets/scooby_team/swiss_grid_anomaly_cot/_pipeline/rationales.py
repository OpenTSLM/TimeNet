"""Turn computed facts into rationale text, and check that the text invents no number.

The text comes from fixed templates. A seeded random choice among equivalent phrasings, keyed by
the window id, varies the wording without changing any fact, so a rebuild gives the same text.
An LLM may later rephrase these rationales; ``check_numbers`` is the gate any rewrite must pass.
"""

import random
import re


_NUMBER = re.compile(r"(?<![\w.\-])-?\d[\d,]*(?:\.\d+)?")
"""A number not glued to a word, so "COVID-19" holds none and "z = -3.5" holds -3.5."""
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")  # fmt: skip


def mw(x: float) -> str:
    return f"{x:,.0f}"


def pct(x: float) -> str:
    return f"{abs(x) * 100:.1f}"


def one(x: float) -> str:
    return f"{x:.1f}"


def day_name(date) -> str:
    return f"{WEEKDAYS[date.day_of_week]} {date.day} {MONTHS[date.month - 1]} {date.year}"


def numbers_in(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER.findall(text)}


def check_numbers(text: str, facts_text: str) -> set[str]:
    """Numbers of ``text`` that do not appear in ``facts_text``. Empty means the text is grounded."""
    allowed = numbers_in(facts_text)
    # Strip a leading minus: the templates write "12.3% below", the facts store the signed value.
    allowed |= {a.lstrip("-") for a in allowed}
    return {n for n in numbers_in(text) if n.lstrip("-") not in allowed}


class Phrasing:
    def __init__(self, key: str):
        # Seeded for a reproducible choice of wording, not for security.
        self.rng = random.Random(key)  # ruff: ignore[suspicious-non-cryptographic-random-usage]

    def pick(self, *options: str) -> str:
        return self.rng.choice(options)


CAUSE_CLAUSE = {
    "public_holiday": "{day} was {holiday}, a public holiday for {share}% of the population",
    "bridge_day": "{day} was a bridge day between a public holiday and the weekend",
    "christmas_newyear": "{day} falls in the Christmas and New Year holiday period",
    "summer_holidays": "{day} falls in the core summer holiday weeks, when the school holidays of "
    "nearly all cantons overlap",
    "covid_lockdown": "{day} falls in the COVID-19 lockdown of spring 2020",
    "cold_snap": "{day} was unusually cold",
    "heatwave": "{day} was unusually hot",
    "unexplained": "no calendar or weather rule explains {day}",
}
CAUSE_SUMMARY = {
    "public_holiday": "a public holiday",
    "bridge_day": "a bridge day",
    "christmas_newyear": "the Christmas and New Year holiday period",
    "summer_holidays": "the summer holidays",
    "covid_lockdown": "the COVID-19 lockdown",
    "cold_snap": "a cold snap",
    "heatwave": "a heatwave",
    "unexplained": "no known cause",
}


def anomaly_rationale(facts: dict, key: str) -> str:
    """Rationale of the cause classification and the localization task of one window."""
    p = Phrasing(key + "/anomaly")
    w = facts["weather"]
    parts = [
        p.pick(
            "The expected consumption follows from the weekday profile, the temperature and the solar radiation.",
            "I compare each day with the consumption expected from its weekday, temperature and solar radiation.",
            "Each day is compared with what its weekday, weather and season predict.",
        ),
        p.pick(
            "The week averaged {t} °C, {a} °C {dir} the seasonal normal, which {verb} expected consumption by about {e} MW.",
            "Temperatures averaged {t} °C, {a} °C {dir} normal for the season; this {verb} the expected load by roughly {e} MW.",
        ).format(
            t=w["temp_mean_c"],
            a=w["temp_anomaly_abs_c"],
            dir=w["direction"],
            verb="raises" if w["effect_sign"] > 0 else "lowers",
            e=w["weather_effect_abs_mw"],
        ),
    ]
    days = facts["anomalous_days"]
    if not days:
        lg = facts["largest_deviation"]
        parts.append(
            p.pick(
                "No day leaves the normal range: the largest deviation is {pct}% {dir} expectation on {day} (z = {z}).",
                "Every day stays within the normal range; the largest gap is {pct}% {dir} expectation, on {day} (z = {z}).",
            ).format(pct=lg["deviation_pct"], dir=lg["direction"], day=lg["day"], z=lg["z"])
        )
        parts.append("No anomaly. Answer: none")
        return " ".join(parts)

    for d in days:
        parts.append(
            p.pick(
                "On {day}, consumption averaged {load} MW, {pct}% {dir} the expected {exp} MW (z = {z}).",
                "{day}: {load} MW against an expected {exp} MW, {pct}% {dir} (z = {z}).",
            ).format(**d)
        )
        clause = CAUSE_CLAUSE[d["cause"]].format(day=d["day"], holiday=d.get("holiday", ""), share=d.get("share", ""))
        parts.append(clause[0].upper() + clause[1:] + ".")
    main = facts["main_anomaly"]
    parts.append(
        p.pick(
            "The largest deviation, on {day}, is explained by {why}. Answer: {label}",
            "The strongest anomaly ({day}) is due to {why}. Answer: {label}",
        ).format(day=main["day"], why=CAUSE_SUMMARY[main["cause"]], label=main["cause"])
    )
    return " ".join(parts)


def localization_rationale(facts: dict, key: str) -> str:
    p = Phrasing(key + "/localization")
    days = facts["anomalous_days"]
    if not days:
        return (
            p.pick(
                "Every day of the window stays within the normal range of the weather- and calendar-adjusted expectation.",
                "No day of this window deviates enough from its expected consumption to count as anomalous.",
            )
            + " Answer: none"
        )
    listed = "; ".join(f"{d['day']} ({d['pct']}% {d['dir']} expectation, z = {d['z']})" for d in days)
    text = p.pick("Anomalous days: {l}.", "The days that deviate from the expectation are: {l}.").format(l=listed)
    return text + " Answer: " + ", ".join(d["date"] for d in days)


def exchange_rationale(facts: dict, key: str) -> str:
    """Rationale and answer of the import/export question."""
    p = Phrasing(key + "/exchange")
    x = facts["exchange"]
    borders = sorted(x["borders"], key=lambda b: b["net_import_mw_signed"], reverse=True)
    imp = [b for b in borders if b["net_import_mw_signed"] > 0]
    exp = [b for b in borders if b["net_import_mw_signed"] < 0]
    parts = [
        p.pick(
            "Over the window, Switzerland was a net {role}: net {noun} averaged {net} MW, with net imports in {share}% of the hours.",
            "Switzerland was a net {role} this window, with average net {noun} of {net} MW; {share}% of the hours were net importing.",
        ).format(role=x["role"], noun=x["net_flow_noun"], net=x["net_flow_abs_mw"], share=x["import_hours_pct"]),
        p.pick(
            "Domestic production averaged {prod} MW against a total consumption of {cons} MW, {gap_txt}.",
            "Production in the Swiss control block averaged {prod} MW while total consumption averaged {cons} MW, {gap_txt}.",
        ).format(
            prod=x["production_mw"],
            cons=x["consumption_mw"],
            gap_txt=("a shortfall of {g} MW" if x["production_short"] else "a surplus of {g} MW").format(g=x["gap_mw"]),
        ),
    ]
    if imp:
        parts.append("Net imports came from " + ", ".join(f"{b['name']} ({b['net_import_mw']} MW)" for b in imp) + ".")
    if exp:
        parts.append(
            "Net exports went to " + ", ".join(f"{b['name']} ({b['net_export_mw']} MW)" for b in reversed(exp)) + "."
        )
    parts.append(f"Answer: net {x['role']}")
    return " ".join(parts)
