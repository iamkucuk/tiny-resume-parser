"""CV record schema: the fixed JSON template the teacher labels and the student emits.

Base: JSON Resume (https://jsonresume.org/schema). Additions come from the field survey of
this dataset (OCR of all 2,451 pages, 2026-10-09; see docs/PLAN.md "Decisions 2026-10-09"):
personal attributes, job type, per-entry technologies, project type, education honors,
skill type, language test scores, and `extra` for anything no other key holds.
`corrections` lists every spelling or grammar fix the labeler made to ordinary words; names,
abbreviations, emails, URLs and codes are never fixed (user decision 2026-10-09).

Missing-value policy: every key is always present, so every record has the same template.
A missing single value is `null`, a missing list is `[]`. Never "N/A" or "" — those could
be real text on a CV.

Dates are ISO 8601 as precise as the CV allows: "2022", "2022-07", "2022-07-15".
An ongoing entry has endDate "present".

The previous flat-17-field / DocTags schema was removed (in git history before 2026-10-15).
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, field_validator

Str = str | None

_DATE_RE = re.compile(r"\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?|present")


def _check_date(v: str | None) -> str | None:
    if v is not None and not _DATE_RE.fullmatch(v):
        raise ValueError(f"date must be YYYY, YYYY-MM, YYYY-MM-DD or 'present', got {v!r}")
    return v


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2026": "..."})


def clean_text(s: str) -> str:
    """Remove differences that carry no information (pilot 2026-10-09: 42 of 160 teacher
    "errors" were only these): curly quotes, the ellipsis character, runs of whitespace,
    space before punctuation, trailing separators."""
    s = re.sub(r"\s+", " ", s.translate(_QUOTES)).strip()
    s = re.sub(r" ([,;:.])", r"\1", s)
    return re.sub(r"[\s,;:.]+$", "", s)


Date = Annotated[str | None, AfterValidator(_check_date)]


class _Strict(BaseModel):
    # Unknown keys are an error, so a teacher answer that drifts from the template fails loud.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @field_validator("*", mode="before")
    @classmethod
    def _clean(cls, v):
        # "" means missing; the template uses null for that. Blank list items are dropped.
        if isinstance(v, str):
            return clean_text(v) or None
        if isinstance(v, list):
            return [y for y in (clean_text(x) if isinstance(x, str) else x for x in v) if y not in ("", None)]
        return v


class Location(_Strict):
    address: Str = None
    postalCode: Str = None
    city: Str = None
    region: Str = None
    country: Str = None


class Profile(_Strict):
    network: Str = None  # "LinkedIn", "GitHub", ...
    username: Str = None
    url: Str = None


class Basics(_Strict):
    name: Str = None
    label: Str = None  # headline under the name, e.g. "Élève ingénieur en génie logiciel"
    emails: list[str] = []
    phones: list[str] = []  # as written
    url: Str = None  # personal website / portfolio
    summary: Str = None
    location: Location = Location()
    profiles: list[Profile] = []
    # Personal attributes (included by user decision, 2026-10-09)
    birthDate: Date = None
    placeOfBirth: Str = None
    age: Str = None  # only when written as an age; never computed from birthDate
    gender: Str = None
    maritalStatus: Str = None
    nationality: Str = None
    drivingLicense: Str = None  # as written, e.g. "Permis B"
    availability: Str = None
    hasPhoto: bool | None = None  # a photo of the person is on the CV


class Work(_Strict):
    name: Str = None  # company / organisation
    position: Str = None
    type: Literal["job", "internship", "freelance"] | None = None  # null unless the page states it
    location: Str = None
    url: Str = None
    startDate: Date = None
    endDate: Date = None
    summary: Str = None  # e.g. the internship subject ("Sujet")
    highlights: list[str] = []  # tasks / achievements, one per item
    technologies: list[str] = []


class Volunteer(_Strict):
    organization: Str = None
    position: Str = None
    url: Str = None
    startDate: Date = None
    endDate: Date = None
    summary: Str = None
    highlights: list[str] = []


class Education(_Strict):
    institution: Str = None
    area: Str = None  # field / speciality / section
    studyType: Str = None  # degree or programme, e.g. "Baccalauréat", "Cycle d'ingénieur"
    location: Str = None
    url: Str = None
    startDate: Date = None
    endDate: Date = None
    score: Str = None  # GPA / moyenne, as written
    honors: Str = None  # "mention", as written
    summary: Str = None  # description of the programme / thesis, as written
    courses: list[str] = []


class Award(_Strict):
    title: Str = None  # prizes, hackathon / competition results
    date: Date = None
    awarder: Str = None
    summary: Str = None


class Certificate(_Strict):
    name: Str = None
    date: Date = None
    issuer: Str = None
    url: Str = None
    summary: Str = None  # description / topics, as written


class Publication(_Strict):
    name: Str = None
    publisher: Str = None
    releaseDate: Date = None
    url: Str = None
    summary: Str = None


class Skill(_Strict):
    name: Str = None  # the category as written, e.g. "Langages de programmation"
    type: Literal["technical", "soft", "other"] | None = None
    level: Str = None  # as written; dots "4/5"; bars estimated "~3/5"
    keywords: list[str] = []  # the individual skills


class Language(_Strict):
    language: Str = None
    fluency: Str = None  # as written ("courant", "B2"); dots "4/5"; bars estimated "~3/5"
    certificate: Str = None  # test and score, e.g. "TOEIC 850"


class Interest(_Strict):
    name: Str = None
    keywords: list[str] = []


class Reference(_Strict):
    name: Str = None
    reference: Str = None  # role / contact details / text, as written


class Project(_Strict):
    name: Str = None
    type: Literal["academic", "final_year", "personal", "professional"] | None = None  # null unless stated
    entity: Str = None  # school or company the project was done for
    description: Str = None
    highlights: list[str] = []
    technologies: list[str] = []
    roles: list[str] = []
    url: Str = None
    startDate: Date = None
    endDate: Date = None


class Extra(_Strict):
    section: Str = None  # heading the information was under
    label: Str = None
    value: Str = None


class Correction(_Strict):
    # One spelling or grammar fix the labeler made (rule 1 of the teacher prompt). The list
    # holds the evidence; counts per kind are computed from it, not written by the model.
    original: Str = None  # the word(s) as written on the page
    corrected: Str = None  # what the record holds instead
    kind: Literal["spelling", "grammar"] | None = None


class CVRecord(_Strict):
    basics: Basics = Basics()
    work: list[Work] = []
    volunteer: list[Volunteer] = []
    education: list[Education] = []
    awards: list[Award] = []
    certificates: list[Certificate] = []
    publications: list[Publication] = []
    skills: list[Skill] = []
    languages: list[Language] = []
    interests: list[Interest] = []
    references: list[Reference] = []
    projects: list[Project] = []
    extra: list[Extra] = []
    corrections: list[Correction] = []  # user decision 2026-10-09: fix spelling/grammar, record each fix


def template() -> dict:
    """The empty record: every key present, nulls and empty lists."""
    return CVRecord().model_dump()


def entry_templates() -> dict[str, dict]:
    """One empty entry per list section, to show the teacher the shape of each entry."""
    return {
        name: field.annotation.__args__[0]().model_dump()
        for name, field in CVRecord.model_fields.items()
        if name != "basics"
    }


def to_record(data: dict) -> dict:
    """Validate a teacher/student answer and return it with the full template filled in.

    Raises pydantic.ValidationError on unknown keys or wrong types.
    """
    return CVRecord.model_validate(data).model_dump()


def _drop_empty(x):
    if isinstance(x, dict):
        kept = {k: _drop_empty(v) for k, v in x.items()}
        return {k: v for k, v in kept.items() if v not in (None, [], {})}
    if isinstance(x, list):
        return [_drop_empty(v) for v in x]
    return x


def to_target(record: dict) -> str:
    """The training target string: compact JSON in fixed key order, with null / [] / {} keys left out.

    ~26% fewer tokens than the full record (decision 2026-10-11); to_record() refills the left-out keys.
    """
    return json.dumps(_drop_empty(to_record(record)), ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    print(json.dumps(template(), indent=1))
    print(json.dumps(entry_templates(), indent=1, ensure_ascii=False))
