from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional,Union

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


# class NavItemModel(BaseModel):
#     nav_type: str = Field(...)
#     nav_label: str = Field(...)
#     nav_href: str = Field(...)


# class NavListModel(BaseModel):
#     nav_items: list[NavItemModel] = Field(default_factory=list)

class NewsSiteNavModel(BaseModel):
    nav_type:Literal["政治类","国际新闻类","当地新闻类","战争类","人权类","女权类","社会矛盾类","犯罪或法律类","灾难事件类"]=Field(...,description='此导航项的类别的中文值')
    nav_label:str =Field(...,description='导航文本')
    nav_href:str =Field(...,description='导航链接')
    
class OrgSiteNavModel(BaseModel):
    nav_type:Literal["政策","法律","新闻"]=Field(...,description='此导航项的类别的中文值')
    nav_label:str =Field(...,description='导航文本')
    nav_href:str =Field(...,description='导航链接')

class NewsSiteNavListModel(BaseModel):
    nav_items:list[NewsSiteNavModel]=Field(...,description='符合要求的导航项列表')
class OrgSiteNavListModel(BaseModel):
    nav_items:list[OrgSiteNavModel]=Field(...,description='符合要求的导航项列表')

# class NavListModel(BaseModel):
#     nav_items:  Union[list[NewsSiteNavModel],list[OrgSiteNavModel]]
#     def __init__(self,from_model:Union[list[NewsSiteNavModel],list[OrgSiteNavModel]]):
#         self.nav_items
        
    

class ListPageItemModel(BaseModel):
    title: str = Field(...)
    href: str = Field(...)


class ListPageModel(BaseModel):
    list_page_type: Literal["文章列表页", "子类别列表页", "不是列表页"] = Field(...)
    list_items: list[ListPageItemModel] = Field(default_factory=list)


class ArticleModel(BaseModel):
    title: str = Field(..., description="文章标题")
    date: str = Field(..., description="文章日期")
    content: str = Field(..., description="文章正文内容")


class ListItemsFitModel(BaseModel):
    fit_count: int = Field(..., description="抽样列表项中符合目标类型的数量")
    total_count: int = Field(..., description="抽样列表项总数")
    passed: bool = Field(..., description="该列表页是否通过类型验证")
    reason: str = Field(..., description="简短原因")
