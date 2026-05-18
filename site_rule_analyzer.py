from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


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


class SiteRuleAnalyzer:
    """单站点（深度优先）规则分析器。

    典型流程：
    1) 校验目标 URL
    2) 协议探测并抓首页
    3) 站点类型识别（启发式）
    4) 首页导航识别
    5) 列表页抽样 + 列表 DOM 规则归纳
    6) 正文页抽样 + 正文 DOM 规则归纳
    """

    def __init__(
        self,
        *,
        timeout: tuple[int, int] = (5, 20),
        user_agent: str = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        list_page_limit: int = 3,
        content_per_list_limit: int = 4,
    ) -> None:
        self.timeout = timeout
        self.list_page_limit = list_page_limit
        self.content_per_list_limit = content_per_list_limit
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

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

        result.site_type = self._step_3_detect_site_type(home_soup)
        result.homepage_navs = self._step_4_extract_navs(home_soup, home.url)

        list_pages = self._step_5_sample_list_pages(result.homepage_navs)
        for lp in list_pages:
            lfr = self._fetch(lp["url"])
            if not lfr.ok:
                continue
            lp["status_code"] = lfr.status_code
            lp["raw_html"] = lfr.text
            lp_soup = BeautifulSoup(lfr.text, "html.parser")
            items = self._extract_list_items(lp_soup, lp["url"])
            lp["items"] = items
            if items:
                rule = self._infer_list_dom_rule(lp_soup, items)
                if rule:
                    result.list_rules.append(rule)
            result.sampled_list_pages.append(lp)

        for lp in result.sampled_list_pages:
            items = lp.get("items", [])[: self.content_per_list_limit]
            for it in items:
                cfr = self._fetch(it["href"])
                if not cfr.ok:
                    continue
                csoup = BeautifulSoup(cfr.text, "html.parser")
                extracted = self._extract_content_fields(csoup)
                page_obj = {
                    "url": it["href"],
                    "title_hint": it.get("title", ""),
                    "raw_html": cfr.text,
                    "fields": extracted,
                }
                result.sampled_content_pages.append(page_obj)
                rule = self._infer_content_dom_rule(csoup, extracted)
                if rule:
                    result.content_rules.append(rule)

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

    def _step_3_detect_site_type(self, soup: BeautifulSoup) -> str:
        text = soup.get_text(" ", strip=True).lower()[:10000]
        org_kw = ["about us", "policy", "regulation", "ministry", "department", "agency"]
        news_kw = ["breaking", "latest", "news", "world", "politics", "opinion"]
        org_hits = sum(k in text for k in org_kw)
        news_hits = sum(k in text for k in news_kw)
        return "机构类" if org_hits > news_hits else "新闻类"

    def _step_4_extract_navs(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        navs: list[dict[str, str]] = []
        seen = set()
        for a in soup.select("nav a[href], header a[href], a[href]"):
            label = a.get_text(" ", strip=True)
            href = (a.get("href") or "").strip()
            if not label or not href:
                continue
            abs_url = urljoin(base_url, href)
            if urlparse(abs_url).netloc != urlparse(base_url).netloc:
                continue
            key = (label, abs_url)
            if key in seen:
                continue
            seen.add(key)
            navs.append({"label": label, "url": abs_url})
            if len(navs) >= 40:
                break
        return navs

    def _step_5_sample_list_pages(self, navs: list[dict[str, str]]) -> list[dict[str, Any]]:
        # 简单启发式：优先看像频道/分类页的链接
        pats = ("news", "world", "politics", "latest", "category", "topics", "china", "international")
        ranked = sorted(navs, key=lambda x: any(p in x["url"].lower() for p in pats), reverse=True)
        return [{"url": x["url"], "nav_label": x["label"]} for x in ranked[: self.list_page_limit]]

    def _extract_list_items(self, soup: BeautifulSoup, base_url: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for a in soup.select("article a[href], main a[href], section a[href]"):
            t = a.get_text(" ", strip=True)
            href = (a.get("href") or "").strip()
            if len(t) < 8 or not href:
                continue
            abs_href = urljoin(base_url, href)
            items.append({"title": t, "href": abs_href})
            if len(items) >= 30:
                break
        dedup = {}
        for it in items:
            dedup[it["href"]] = it
        return list(dedup.values())

    # ----------- Step5核心：列表项回贴DOM并找最近祖先 -----------
    def _infer_list_dom_rule(self, soup: BeautifulSoup, items: list[dict[str, str]]) -> Optional[dict[str, Any]]:
        matched_nodes: list[Tag] = []
        href_map = {i["href"]: i for i in items}
        for a in soup.select("a[href]"):
            ah = urljoin("", (a.get("href") or "").strip())
            # 优先用链接匹配；其次用文本弱匹配
            if ah in href_map:
                matched_nodes.append(a)
                continue
            txt = a.get_text(" ", strip=True)
            if any(txt and txt in it["title"] for it in items[:20]):
                matched_nodes.append(a)
        if len(matched_nodes) < 2:
            return None

        lca = self._lowest_common_ancestor(matched_nodes)
        if not lca:
            return None

        branch_selectors = [self._css_path(n) for n in matched_nodes[:10]]
        root_selector = self._selector_with_identity(lca)

        return {
            "root_selector": root_selector,
            "item_branch_selectors": branch_selectors,
            "item_link_selector": "a[href]",
        }

    # ----------- Step6核心：正文片段回贴DOM并找正文根 -----------
    def _extract_content_fields(self, soup: BeautifulSoup) -> dict[str, str]:
        title = ""
        h1 = soup.select_one("h1")
        if h1:
            title = h1.get_text(" ", strip=True)
        date = ""
        dt = soup.select_one("time")
        if dt:
            date = dt.get_text(" ", strip=True)
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article p, main p, p") if len(p.get_text(strip=True)) > 30]
        body = "\n".join(paragraphs[:20])
        return {"title": title, "date": date, "body": body}

    def _infer_content_dom_rule(self, soup: BeautifulSoup, fields: dict[str, str]) -> Optional[dict[str, Any]]:
        body = fields.get("body", "")
        chunks = [c.strip() for c in body.split("\n") if len(c.strip()) > 20]
        if not chunks:
            return None
        sample_chunks = random.sample(chunks, k=min(3, len(chunks)))

        matched: list[Tag] = []
        for p in soup.select("article p, main p, p"):
            t = p.get_text(" ", strip=True)
            if any(c[:20] in t for c in sample_chunks):
                matched.append(p)
        if len(matched) < 2:
            return None

        lca = self._lowest_common_ancestor(matched)
        if not lca:
            return None

        content_root = self._selector_with_identity(lca)
        title_sel = self._stable_field_selector(soup, fields.get("title", ""), ["h1", "h2", "header h1"])
        date_sel = self._stable_field_selector(soup, fields.get("date", ""), ["time", ".date", "[class*=date]"])

        return {
            "content_root_selector": content_root,
            "title_selector": title_sel,
            "date_selector": date_sel,
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
        if not common:
            return None
        # 返回最深公共祖先
        for anc in paths[0]:
            if anc in common:
                return anc
        return None

    def _ancestor_chain(self, node: Tag) -> list[Tag]:
        chain = []
        cur: Optional[Tag] = node
        while isinstance(cur, Tag):
            chain.append(cur)
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return chain

    def _selector_with_identity(self, node: Tag) -> str:
        # 在根附近尝试 class/id 特征来稳定定位
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
        try:
            resp = self.session.get(url, timeout=self.timeout)
            return FetchResult(url=url, ok=resp.ok, status_code=resp.status_code, text=resp.text)
        except Exception as exc:  # noqa: BLE001
            return FetchResult(url=url, ok=False, error=str(exc))


__all__ = ["SiteRuleAnalyzer", "SiteAnalyzeResult"]
