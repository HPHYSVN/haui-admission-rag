from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Category(str, Enum):
    ADMISSION = "01_admission"
    MINISTRY_REGULATIONS = "02_ministry_regulations"
    CUTOFF_SCORES = "03_cutoff_scores"
    CURRICULUM = "04_curriculum"
    TUITION_SCHOLARSHIP = "05_tuition_scholarship"
    CAREER = "06_career"
    CAREER_ORIENTATION = "07_career_orientation"
    ENROLLMENT_PROCEDURE = "08_enrollment_procedure"
    TRAINING_REGULATIONS = "09_training_regulations"
    DORMITORY_HOUSING = "10_dormitory_housing"
    STUDENT_LIFE = "11_student_life"
    STUDENT_SUPPORT = "12_student_support"
    TRAINING_SYSTEMS = "13_training_systems"
    ABOUT_HAUI = "14_about_haui"


class SourceType(str, Enum):
    OFFICIAL_WEBSITE = "official_website"
    OFFICIAL_DOCUMENT = "official_document"
    OFFICIAL_PORTAL = "official_portal"
    GOVERNMENT_DOCUMENT = "government_document"
    GOVERNMENT_PORTAL = "government_portal"


class Status(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


@dataclass
class SourceSeed:
    category: str
    url: str
    source_name: str
    source_type: str
    title_hint: str
    subcategory: str | None = None
    authority: str | None = None
    university: str | None = None
    document_number: str | None = None
    document_type: str | None = None
    local_path: str | None = None


@dataclass
class Document:
    # 10 Required fields
    id: str
    title: str
    content: str
    source_url: str
    source_name: str
    source_type: str
    category: str
    crawled_date: str
    language: str = "vi"
    status: str = "active"

    # Optional metadata fields
    subcategory: str | None = None
    year: int | None = None
    published_date: str | None = None
    effective_date: str | None = None
    expiry_date: str | None = None
    university: str | None = None
    authority: str | None = None
    document_number: str | None = None
    document_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Exclude optional fields that are None to keep documents clean and compliant
        clean_data: dict[str, Any] = {}
        for k, v in data.items():
            if v is not None:
                clean_data[k] = v
        return clean_data
