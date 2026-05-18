from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from http_helper import headers, load_page, proxies_7890


RE_HTTP = re.compile(r"^https?://", re.IGNORECASE)


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
    site_type: Literal["新闻类", "机构类"] = Field(..., description="此站点的类型")


class NavItemModel(BaseModel):
    nav_type: str = Field(..., description="导航类型")
    nav_label: str = Field(..., description="导航文本")
    nav_href: str = Field(..., description="导航链接")


class NavListModel(BaseModel):
    nav_items: list[NavItemModel] = Field(default_factory=list)


class ListPageItemModel(BaseModel):
    title: str = Field(..., description="列表项标题")
    href: str = Field(..., description="正文页链接")


class ListPageModel(BaseModel):
    list_page_type: Literal["文章列表页", "子类别列表页", "不是列表页"] = Field(..., description="列表页类型")
    list_items: list[ListPageItemModel] = Field(default_factory=list)


class SiteRuleAnalyzer:
    """按你原先 notebook 方式改造：核心识别由 lmc.extract 驱动，DOM/LCA 用于规则落地。"""

    def __init__(
        self,
        *,
        lmc: Any,
        timeout: tuple[int, int] = (5, 20),
        list_page_limit: int = 3,
        content_per_list_limit: int = 3,
    ) -> None:
        if lmc is None:
            raise ValueError("lmc 不能为空：请传入你的 llm_client 实例")
        self.lmc = lmc
        self.timeout = timeout
        self.list_page_limit = list_page_limit
        self.content_per_list_limit = content_per_list_limit
        self.session = None

    def analyze_site(self, target: str) -> SiteAnalyzeResult:
        result = SiteAnalyzeResult(input_url=target)
        normalized = self._step_1_validate_target(target)
        if not normalized:
            result.errors.append("目标网址为空或无效")
            return result

        home = self._step_2_probe_protocol_and_fetch_home(normalized)
        if not home.ok:
            result.errors.append(f"首页抓取失败: {home.error or home.status_code}")
            return result

        result.normalized_url = home.url
        home_soup = BeautifulSoup(home.text, "html.parser")
        result.site_type = self._step_3_detect_site_type(home.url, home.text)
        result.homepage_navs = self._step_4_extract_navs_with_llm(home.url, home.text, result.site_type)

        list_pages = self._step_5_sample_list_pages(result.homepage_navs)
        for lp in list_pages:
            lfr = self._fetch(lp["url"])
            if not lfr.ok:
                continue
            lp_soup = BeautifulSoup(lfr.text, "html.parser")
            lp["raw_html"] = lfr.text
            lp["status_code"] = lfr.status_code

            list_items = self._step_5_extract_list_items_with_llm(lp["url"], lfr.text)
            lp["items"] = list_items

            list_rule = self._infer_list_dom_rule(lp_soup, list_items, lp["url"])
            if list_rule:
                result.list_rules.append(list_rule)
            result.sampled_list_pages.append(lp)

        for lp in result.sampled_list_pages:
            for it in lp.get("items", [])[: self.content_per_list_limit]:
                cfr = self._fetch(it.get("href", ""))
                if not cfr.ok:
                    continue
                csoup = BeautifulSoup(cfr.text, "html.parser")
                fields = self._step_6_extract_content_fields_with_llm(cfr.url, cfr.text)
                result.sampled_content_pages.append({"url": cfr.url, "fields": fields})
                content_rule = self._infer_content_dom_rule(csoup, fields)
                if content_rule:
                    result.content_rules.append(content_rule)

        return result

    def _step_1_validate_target(self, target: str) -> Optional[str]:
        t = (target or "").strip()
        if not t or t.upper() in {"#N/A", "N/A", "NA", "NULL"}:
            return None
        return t

    def _step_2_probe_protocol_and_fetch_home(self, target: str) -> FetchResult:
        candidates = [target] if RE_HTTP.match(target) else [f"https://{target}", f"http://{target}"]
        for u in candidates:
            r = self._fetch(u)
            if r.ok:
                return r
        return FetchResult(url=candidates[0], ok=False, error="all candidates failed")

    def _step_3_detect_site_type(self, target_url: str, homepage_html: str) -> str:
        prompt = "你的任务根据用户输入的网站和网址判断其类型。只返回'新闻类'或'机构类'。"
        inp = f"url={target_url}\nhtml_head={homepage_html[:8000]}"
        out = self.lmc.extract(prompt, inp, SiteTypeModel)
        if isinstance(out, SiteTypeModel):
            return out.site_type
        return "新闻类"

    def _step_4_extract_navs_with_llm(self, base_url: str, homepage_html: str, site_type: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(homepage_html, "html.parser")
        text = "\n".join(a.get_text(" ", strip=True) + " | " + (a.get("href") or "") for a in soup.select("a[href]")[:200])
        if site_type == "新闻类":
            instruction = (
                "从首页导航候选中提取与[政治类,国际新闻类,当地新闻类,战争类,法律犯罪类,灾难事件类]相关项，返回nav_items"
            )
        else:
            instruction = "从首页导航候选中提取与[政策,法律,新闻]相关项，返回nav_items"

        out = self.lmc.extract(instruction, text, NavListModel)
        navs: list[dict[str, str]] = []
        if isinstance(out, NavListModel):
            for n in out.nav_items:
                href = (n.nav_href or "").strip()
                if not href:
                    continue
                abs_url = urljoin(base_url, href)
                if urlparse(abs_url).netloc != urlparse(base_url).netloc:
                    continue
                navs.append({"label": n.nav_label.strip(), "url": abs_url, "type": n.nav_type.strip()})
        return navs

    def _step_5_sample_list_pages(self, navs: list[dict[str, str]]) -> list[dict[str, Any]]:
        return [{"url": n["url"], "nav_label": n.get("label", "")} for n in navs[: self.list_page_limit]]

    def _step_5_extract_list_items_with_llm(self, page_url: str, page_html: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(page_html, "html.parser")
        md_like = "\n".join(a.get_text(" ", strip=True) + " | " + (a.get("href") or "") for a in soup.select("a[href]")[:500])
        instruction = (
            "这是内容列表页候选。提取主体列表项（title,href）；若不是列表页返回list_page_type='不是列表页'且list_items空。"
        )
        out = self.lmc.extract(instruction, md_like, ListPageModel)
        if not isinstance(out, ListPageModel):
            return []
        if out.list_page_type == "不是列表页":
            return []
        items = []
        for it in out.list_items:
            href = urljoin(page_url, (it.href or "").strip())
            if not href:
                continue
            items.append({"title": it.title.strip(), "href": href})
        return items

    def _step_6_extract_content_fields_with_llm(self, page_url: str, page_html: str) -> dict[str, str]:
        # 保持你的原方式：正文字段也交给 LLM 先抽出，再回贴 DOM 推规则
        instruction = "从新闻正文页HTML中提取title/date/body(正文纯文本)三个字段，没有则空字符串。"
        schema = {
            "title": "str",
            "date": "str",
            "body": "str",
        }
        out = self.lmc.extract(instruction, f"url={page_url}\nhtml={page_html[:15000]}\nschema={schema}")
        if isinstance(out, dict):
            return {
                "title": str(out.get("title", "")),
                "date": str(out.get("date", "")),
                "body": str(out.get("body", "")),
            }
        return {"title": "", "date": "", "body": ""}

    def _infer_list_dom_rule(self, soup: BeautifulSoup, items: list[dict[str, str]], base_url: str) -> Optional[dict[str, Any]]:
        matched_nodes: list[Tag] = []
        href_map = {i["href"]: i for i in items}
        for a in soup.select("a[href]"):
            ah = urljoin(base_url, (a.get("href") or "").strip())
            if ah in href_map:
                matched_nodes.append(a)
                continue
            txt = a.get_text(" ", strip=True)
            if any(txt and txt in it["title"] for it in items[:30]):
                matched_nodes.append(a)
        if len(matched_nodes) < 2:
            return None
        lca = self._lowest_common_ancestor(matched_nodes)
        if not lca:
            return None
        return {
            "root_selector": self._selector_with_identity(lca),
            "item_branch_selectors": [self._css_path(n) for n in matched_nodes[:15]],
            "item_link_selector": "a[href]",
        }

    def _infer_content_dom_rule(self, soup: BeautifulSoup, fields: dict[str, str]) -> Optional[dict[str, Any]]:
        chunks = [c.strip() for c in fields.get("body", "").split("\n") if len(c.strip()) > 12]
        if not chunks:
            return None
        candidate = chunks[:3]
        matched: list[Tag] = []
        for p in soup.select("article p, main p, div p, p"):
            t = p.get_text(" ", strip=True)
            if any(c[:20] in t for c in candidate):
                matched.append(p)
        if len(matched) < 2:
            return None
        lca = self._lowest_common_ancestor(matched)
        if not lca:
            return None
        return {
            "content_root_selector": self._selector_with_identity(lca),
            "title_selector": self._stable_field_selector(soup, fields.get("title", ""), ["h1", "h2"]),
            "date_selector": self._stable_field_selector(soup, fields.get("date", ""), ["time", ".date", "[class*=date]"]),
            "paragraph_selector": "p",
        }

    def _stable_field_selector(self, soup: BeautifulSoup, value: str, fallbacks: list[str]) -> str:
        if value:
            for el in soup.find_all(True):
                if value[:20] and value[:20] in el.get_text(" ", strip=True):
                    return self._selector_with_identity(el)
        return fallbacks[0]

    def _lowest_common_ancestor(self, nodes: list[Tag]) -> Optional[Tag]:
        if not nodes:
            return None
        paths = [self._ancestor_chain(n) for n in nodes]
        common = set(paths[0])
        for p in paths[1:]:
            common &= set(p)
        for anc in paths[0]:
            if anc in common:
                return anc
        return None

    def _ancestor_chain(self, node: Tag) -> list[Tag]:
        chain: list[Tag] = []
        cur: Optional[Tag] = node
        while isinstance(cur, Tag):
            chain.append(cur)
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return chain

    def _selector_with_identity(self, node: Tag) -> str:
        if node.get("id"):
            return f"#{node['id']}"
        classes = [c for c in (node.get("class") or []) if isinstance(c, str)]
        if classes:
            return f"{node.name}." + ".".join(classes[:2])
        return self._css_path(node)

    def _css_path(self, node: Tag) -> str:
        parts = []
        cur: Optional[Tag] = node
        while isinstance(cur, Tag) and cur.name != "[document]":
            sibs = [s for s in cur.parent.find_all(cur.name, recursive=False)] if isinstance(cur.parent, Tag) else [cur]
            idx = sibs.index(cur) + 1 if cur in sibs else 1
            parts.append(f"{cur.name}:nth-of-type({idx})")
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return " > ".join(reversed(parts))

    def _fetch(self, url: str) -> FetchResult:
        ret: dict[str, Any] = {}
        ok = load_page(
            url,
            ret,
            session=self.session,
            headers=headers,
            proxies=proxies_7890,
            timeout=self.timeout,
        )
        code_raw = str(ret.get("code", "")).split(",")[-1].strip()
        try:
            status_code = int(code_raw) if code_raw else None
        except ValueError:
            status_code = None
        return FetchResult(
            url=url,
            ok=bool(ok),
            status_code=status_code,
            text=str(ret.get("content", "") or ""),
            error=None if ok else str(ret.get("code", "load_page_failed")),
        )


__all__ = ["SiteRuleAnalyzer", "SiteAnalyzeResult"]
