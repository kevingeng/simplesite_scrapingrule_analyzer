from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


@dataclass
class FetchResult:
    url: str
    ok: bool
    status_code: Optional[int] = None
    text: str = ""
    error: Optional[str] = None


@dataclass
class SiteAnalyzeResult:
    input_url: str
    normalized_url: Optional[str] = None
    site_type: Optional[str] = None
    homepage_navs: list[dict[str, str]] = field(default_factory=list)
    sampled_list_pages: list[dict[str, Any]] = field(default_factory=list)
    sampled_content_pages: list[dict[str, Any]] = field(default_factory=list)
    list_rules: list[dict[str, Any]] = field(default_factory=list)
    content_rules: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SiteTypeModel(BaseModel):
    site_type: Literal["新闻类", "机构类"] = Field(...)


class NavItemModel(BaseModel):
    nav_type: str = Field(...)
    nav_label: str = Field(...)
    nav_href: str = Field(...)


class NavListModel(BaseModel):
    nav_items: list[NavItemModel] = Field(default_factory=list)


class ListPageItemModel(BaseModel):
    title: str = Field(...)
    href: str = Field(...)


class ListPageModel(BaseModel):
    list_page_type: Literal["文章列表页", "子类别列表页", "不是列表页"] = Field(...)
    list_items: list[ListPageItemModel] = Field(default_factory=list)
