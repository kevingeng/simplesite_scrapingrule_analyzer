from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import html2text
from bs4 import BeautifulSoup, Tag

from tools.http_helper import headers, load_page, proxies_7890
from .models import FetchResult, ListPageModel, NavListModel, SiteAnalyzeResult, SiteTypeModel
from .utils import load_json, save_json

RE_HTTP = re.compile(r"^https?://", re.IGNORECASE)
ht = html2text.HTML2Text()


class SiteRuleAnalyzer:
    def __init__(self, *, lmc: Any, work_dir: Path, log, timeout: tuple[int, int] = (5, 20)) -> None:
        self.lmc = lmc
        self.work_dir = work_dir
        self.state_path = work_dir / "state.json"
        self.log = log
        self.timeout = timeout
        self.session = None
        self.state: dict[str, Any] = load_json(self.state_path)

    def _persist(self) -> None:
        save_json(self.state_path, self.state)

    def _clear_from_step(self, force_restep: int) -> None:
        keys = ["step1", "step2", "step3", "step4", "step5", "step6", "result"]
        for i, k in enumerate(keys, start=1):
            if i >= force_restep and k in self.state:
                self.state.pop(k, None)

    def analyze_site(self, target: str, force_restep: int = 0) -> SiteAnalyzeResult:
        if force_restep > 0:
            self.log.info("force_restep=%s: clear state from this step", force_restep)
            self._clear_from_step(force_restep)
            self._persist()

        s1 = self.state.get("step1") or self._step_1_validate_target(target)
        self.state["step1"] = s1
        self._persist()

        s2 = self.state.get("step2") or self._step_2_fetch_home(s1["target"])
        self.state["step2"] = s2
        self._persist()

        s3 = self.state.get("step3") or self._step_3_site_type(s2["normalized_url"], s2["home_html"])
        self.state["step3"] = s3
        self._persist()

        s4 = self.state.get("step4") or self._step_4_nav(s2["normalized_url"], s2["home_html"], s3["site_type"])
        self.state["step4"] = s4
        self._persist()

        s5 = self.state.get("step5") or self._step_5_lists(s4["homepage_navs"])
        self.state["step5"] = s5
        self._persist()

        s6 = self.state.get("step6") or self._step_6_contents(s5["sampled_list_pages"])
        self.state["step6"] = s6
        self._persist()

        res = SiteAnalyzeResult(
            input_url=target,
            normalized_url=s2["normalized_url"],
            site_type=s3["site_type"],
            homepage_navs=s4["homepage_navs"],
            sampled_list_pages=s5["sampled_list_pages"],
            sampled_content_pages=s6["sampled_content_pages"],
            list_rules=s5["list_rules"],
            content_rules=s6["content_rules"],
            errors=[],
        )
        self.state["result"] = res.__dict__
        self._persist()
        return res

    def _make_page_md(self, base_url: str, raw_html: str) -> str:
        ht.baseurl = base_url
        ht.images_to_alt = True
        return ht.handle(raw_html or "")

    def _step_1_validate_target(self, target: str) -> dict[str, Any]:
        self.log.info("step1 validate target")
        t = (target or "").strip()
        if not t or t.upper() in {"#N/A", "N/A", "NA", "NULL"}:
            raise ValueError("目标网址为空或无效")
        return {"step_idx": 1, "target": t}

    def _step_2_fetch_home(self, target: str) -> dict[str, Any]:
        self.log.info("step2 fetch home")
        candidates = [target] if RE_HTTP.match(target) else [f"https://{target}", f"http://{target}"]
        for url in candidates:
            fr = self._fetch(url)
            if fr.ok:
                return {"step_idx": 2, "normalized_url": fr.url, "home_html": fr.text}
        raise RuntimeError("home fetch failed")

    def _step_3_site_type(self, url: str, home_html: str) -> dict[str, Any]:
        self.log.info("step3 detect site_type")
        out = self.lmc.extract("判断网站类型，只返回新闻类或机构类", f"url={url}\nhtml={home_html[:8000]}", SiteTypeModel)
        st = out.site_type if isinstance(out, SiteTypeModel) else "新闻类"
        return {"step_idx": 3, "site_type": st}

    def _step_4_nav(self, base_url: str, home_html: str, site_type: str) -> dict[str, Any]:
        self.log.info("step4 extract nav")
        page_md = self._make_page_md(base_url, home_html)[:9000]
        ins = "提取新闻相关导航nav_items" if site_type == "新闻类" else "提取政策法律新闻导航nav_items"
        out = self.lmc.extract(ins, page_md, NavListModel)
        navs = []
        if isinstance(out, NavListModel):
            for n in out.nav_items:
                u = urljoin(base_url, n.nav_href)
                if urlparse(u).netloc == urlparse(base_url).netloc:
                    navs.append({"label": n.nav_label, "url": u, "type": n.nav_type})
        return {"step_idx": 4, "homepage_navs": navs}

    def _step_5_lists(self, navs: list[dict[str, str]]) -> dict[str, Any]:
        self.log.info("step5 list pages")
        sampled, rules = [], []
        for n in navs[:3]:
            fr = self._fetch(n["url"])
            if not fr.ok:
                continue
            md = self._make_page_md(n["url"], fr.text)[:12000]
            out = self.lmc.extract("提取主体列表项title/href，不是列表页返回空", md, ListPageModel)
            items = []
            if isinstance(out, ListPageModel) and out.list_page_type != "不是列表页":
                items = [{"title": i.title, "href": urljoin(n["url"], i.href)} for i in out.list_items]
            soup = BeautifulSoup(fr.text, "html.parser")
            rule = self._infer_list_dom_rule(soup, items, n["url"])
            if rule:
                rules.append(rule)
            sampled.append({"url": n["url"], "items": items, "raw_html": fr.text})
        return {"step_idx": 5, "sampled_list_pages": sampled, "list_rules": rules}

    def _step_6_contents(self, list_pages: list[dict[str, Any]]) -> dict[str, Any]:
        self.log.info("step6 content pages")
        pages, rules = [], []
        for lp in list_pages:
            for it in lp.get("items", [])[:3]:
                fr = self._fetch(it["href"])
                if not fr.ok:
                    continue
                md = self._make_page_md(fr.url, fr.text)[:15000]
                fields = self.lmc.extract("提取title/date/body", md) or {}
                if not isinstance(fields, dict):
                    fields = {}
                soup = BeautifulSoup(fr.text, "html.parser")
                rule = self._infer_content_dom_rule(soup, {"title": str(fields.get("title", "")), "date": str(fields.get("date", "")), "body": str(fields.get("body", ""))})
                if rule:
                    rules.append(rule)
                pages.append({"url": fr.url, "fields": fields})
        return {"step_idx": 6, "sampled_content_pages": pages, "content_rules": rules}

    def _fetch(self, url: str) -> FetchResult:
        ret: dict[str, Any] = {}
        ok = load_page(url, ret, session=self.session, headers=headers, proxies=proxies_7890, timeout=self.timeout)
        code_raw = str(ret.get("code", "")).split(",")[-1].strip()
        status_code = int(code_raw) if code_raw.isdigit() else None
        return FetchResult(url=url, ok=bool(ok), status_code=status_code, text=str(ret.get("content", "") or ""), error=None if ok else str(ret.get("code", "failed")))

    def _infer_list_dom_rule(self, soup: BeautifulSoup, items: list[dict[str, str]], base_url: str) -> Optional[dict[str, Any]]:
        matched: list[Tag] = []
        href_map = {i["href"]: i for i in items}
        for a in soup.select("a[href]"):
            ah = urljoin(base_url, (a.get("href") or "").strip())
            if ah in href_map:
                matched.append(a)
        if len(matched) < 2:
            return None
        lca = self._lowest_common_ancestor(matched)
        if not lca:
            return None
        return {"root_selector": self._selector_with_identity(lca), "item_branch_selectors": [self._css_path(n) for n in matched[:10]], "item_link_selector": "a[href]"}

    def _infer_content_dom_rule(self, soup: BeautifulSoup, fields: dict[str, str]) -> Optional[dict[str, Any]]:
        chunks = [c.strip() for c in fields.get("body", "").split("\n") if len(c.strip()) > 12]
        if not chunks:
            return None
        matched = [p for p in soup.select("article p, main p, div p, p") if any(c[:20] in p.get_text(" ", strip=True) for c in chunks[:3])]
        if len(matched) < 2:
            return None
        lca = self._lowest_common_ancestor(matched)
        if not lca:
            return None
        return {"content_root_selector": self._selector_with_identity(lca), "title_selector": "h1", "date_selector": "time", "paragraph_selector": "p"}

    def _lowest_common_ancestor(self, nodes: list[Tag]) -> Optional[Tag]:
        paths = [self._ancestor_chain(n) for n in nodes]
        common = set(paths[0])
        for p in paths[1:]:
            common &= set(p)
        for anc in paths[0]:
            if anc in common:
                return anc
        return None

    def _ancestor_chain(self, node: Tag) -> list[Tag]:
        chain, cur = [], node
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
        parts, cur = [], node
        while isinstance(cur, Tag) and cur.name != "[document]":
            sibs = [s for s in cur.parent.find_all(cur.name, recursive=False)] if isinstance(cur.parent, Tag) else [cur]
            idx = sibs.index(cur) + 1 if cur in sibs else 1
            parts.append(f"{cur.name}:nth-of-type({idx})")
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return " > ".join(reversed(parts))
