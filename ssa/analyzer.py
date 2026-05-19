from __future__ import annotations

import json
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
        self.state.setdefault("cache", {})

    def _persist(self) -> None:
        save_json(self.state_path, self.state)

    def _clear_from_step(self, force_restep: int) -> None:
        keys = ["step1", "step2", "step3", "step4", "step5", "step6", "result"]
        for i, k in enumerate(keys, start=1):
            if i >= force_restep and k in self.state:
                self.state.pop(k, None)
        self.state["cache"] = {}

    def _cache_get(self, key: str) -> Any:
        return self.state.get("cache", {}).get(key)

    def _cache_set(self, key: str, value: Any) -> Any:
        self.state.setdefault("cache", {})[key] = value
        self._persist()
        return value

    def analyze_site(self, target: str, force_restep: int = 0) -> SiteAnalyzeResult:
        if force_restep > 0:
            self.log.info("force_restep=%s: clear state from this step", force_restep)
            self._clear_from_step(force_restep)
            self._persist()

        s1 = self.state.get("step1") or {"step_idx": 1}
        self.state["step1"] = s1
        s1 = self._step_1_validate_target(s1, target)
        self.log.debug("step1 done: %s", s1)
        self.state["step1"] = s1
        self._persist()

        s2 = self.state.get("step2") or {"step_idx": 2}
        self.state["step2"] = s2
        s2 = self._step_2_fetch_home(s2, s1["target"])
        self.log.debug("step2 done: normalized_url=%s home_html_len=%s", s2.get("normalized_url"), len(s2.get("home_html", "")))
        self.state["step2"] = s2
        self._persist()

        s3 = self.state.get("step3") or {"step_idx": 3}
        self.state["step3"] = s3
        s3 = self._step_3_site_type(s3, s2["normalized_url"], s2["home_html"])
        self.log.debug("step3 done: %s", s3)
        self.state["step3"] = s3
        self._persist()

        s4 = self.state.get("step4") or {"step_idx": 4}
        self.state["step4"] = s4
        s4 = self._step_4_nav(s4, s2["normalized_url"], s2["home_html"], s3["site_type"])
        self.log.debug("step4 done: nav_count=%d", len(s4.get("homepage_navs", [])))
        self.state["step4"] = s4
        self._persist()

        s5 = self.state.get("step5") or {"step_idx": 5}
        self.state["step5"] = s5
        s5 = self._step_5_lists(s5, s4.get("homepage_navs", []))
        self.log.debug("step5 done: list_pages=%d list_rules=%d", len(s5.get("sampled_list_pages", [])), len(s5.get("list_rules", [])))
        self.state["step5"] = s5
        self._persist()

        s6 = self.state.get("step6") or {"step_idx": 6}
        self.state["step6"] = s6
        s6 = self._step_6_contents(s6, s5.get("sampled_list_pages", []))
        self.log.debug("step6 done: content_pages=%d content_rules=%d", len(s6.get("sampled_content_pages", [])), len(s6.get("content_rules", [])))
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
        cache_key = f"md::{base_url}::{len(raw_html or '')}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, str):
            self.log.debug("cache hit md: %s", base_url)
            return cached
        ht.baseurl = base_url
        ht.images_to_alt = True
        page_md = ht.handle(raw_html or "")
        self.log.debug("make_page_md: url=%s raw_len=%d md_len=%d", base_url, len(raw_html or ""), len(page_md))
        return self._cache_set(cache_key, page_md)

    def _safe_extract(self, instruction: str, content: str, model: Any = None) -> Any:
        model_name = getattr(model, "__name__", "none")
        cache_key = f"llm::{model_name}::{hash(instruction)}::{hash(content[:4000])}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            self.log.debug("cache hit llm: model=%s", model_name)
            if model is None:
                return cached
            try:
                return model.model_validate(cached) if isinstance(cached, dict) else cached
            except Exception:  # noqa: BLE001
                return cached
        try:
            if model is None:
                out = self.lmc.extract(instruction, content)
                self._cache_set(cache_key, out)
                return out
            out = self.lmc.extract(instruction, content, model)
            if hasattr(out, "model_dump"):
                self._cache_set(cache_key, out.model_dump())
            else:
                self._cache_set(cache_key, out)
            return out
        except Exception as ex:  # noqa: BLE001
            self.log.exception("lmc.extract failed: %s", ex)
            return None

    def _step_1_validate_target(self, step: dict[str, Any], target: str) -> dict[str, Any]:
        self.log.info("step1 validate target")
        if step.get("target"):
            return step
        t = (target or "").strip()
        if not t or t.upper() in {"#N/A", "N/A", "NA", "NULL"}:
            raise ValueError("目标网址为空或无效")
        step["target"] = t
        return step

    def _step_2_fetch_home(self, step: dict[str, Any], target: str) -> dict[str, Any]:
        self.log.info("step2 fetch home")
        if step.get("normalized_url") and step.get("home_html"):
            return step
        candidates = [target] if RE_HTTP.match(target) else [f"https://{target}", f"http://{target}"]
        self.log.debug("step2 candidates=%s", candidates)
        step["candidates"] = candidates
        for url in candidates:
            fr = self._fetch(url)
            self.log.debug("step2 fetched url=%s ok=%s status=%s content_len=%d", url, fr.ok, fr.status_code, len(fr.text or ""))
            if fr.ok:
                step["normalized_url"] = fr.url
                step["home_html"] = fr.text
                return step
        raise RuntimeError("home fetch failed")

    def _step_3_site_type(self, step: dict[str, Any], url: str, home_html: str) -> dict[str, Any]:
        self.log.info("step3 detect site_type")
        if step.get("site_type"):
            return step
        out = self._safe_extract("判断网站类型，只返回新闻类或机构类", f"url={url}\nhtml={home_html[:8000]}", SiteTypeModel)
        st = out.site_type if isinstance(out, SiteTypeModel) else "新闻类"
        step["site_type"] = st
        return step

    def _step_4_nav(self, step: dict[str, Any], base_url: str, home_html: str, site_type: str) -> dict[str, Any]:
        self.log.info("step4 extract nav")
        if step.get("homepage_navs"):
            return step
        page_md = self._make_page_md(base_url, home_html)[:9000]
        ins = "提取新闻相关导航nav_items" if site_type == "新闻类" else "提取政策法律新闻导航nav_items"
        out = self._safe_extract(ins, page_md, NavListModel)
        navs = []
        if isinstance(out, NavListModel):
            for n in out.nav_items:
                u = urljoin(base_url, n.nav_href)
                if urlparse(u).netloc == urlparse(base_url).netloc:
                    navs.append({"label": n.nav_label, "url": u, "type": n.nav_type})
        self.log.debug("step4 nav candidates=%d accepted=%d", len(getattr(out, 'nav_items', []) if out else []), len(navs))
        step["homepage_navs"] = navs
        return step

    def _coerce_listpage_model(self, out: Any) -> Optional[ListPageModel]:
        if isinstance(out, ListPageModel):
            if isinstance(out.list_items, str):
                try:
                    fixed_items = json.loads(out.list_items.strip())
                    out = ListPageModel(list_page_type=out.list_page_type, list_items=fixed_items)
                except Exception:
                    self.log.warning("list_items was string but json parse failed")
                    return None
            return out
        if isinstance(out, dict):
            try:
                if isinstance(out.get("list_items"), str):
                    out["list_items"] = json.loads(out["list_items"].strip())
                return ListPageModel.model_validate(out)
            except Exception as ex:  # noqa: BLE001
                self.log.warning("coerce dict->ListPageModel failed: %s", ex)
                return None
        return None

    def _step_5_lists(self, step: dict[str, Any], navs: list[dict[str, str]]) -> dict[str, Any]:
        self.log.info("step5 list pages")
        if step.get("sampled_list_pages") is not None and step.get("list_rules") is not None:
            return step
        sampled, rules = step.get("sampled_list_pages", []), step.get("list_rules", [])
        for idx, n in enumerate(navs[:3], start=1):
            fr = self._fetch(n["url"])
            self.log.debug("step5[%d] fetch url=%s ok=%s status=%s", idx, n['url'], fr.ok, fr.status_code)
            if not fr.ok:
                continue
            md = self._make_page_md(n["url"], fr.text)[:12000]
            out_raw = self._safe_extract("提取主体列表项title/href，不是列表页返回空", md, ListPageModel)
            out = self._coerce_listpage_model(out_raw)
            items = []
            if out and out.list_page_type != "不是列表页":
                items = [{"title": i.title, "href": urljoin(n["url"], i.href)} for i in out.list_items]
            self.log.debug("step5[%d] list_page_type=%s items=%d", idx, out.list_page_type if out else None, len(items))
            soup = BeautifulSoup(fr.text, "html.parser")
            rule = self._infer_list_dom_rule(soup, items, n["url"])
            if rule:
                rules.append(rule)
            sampled.append({"url": n["url"], "items": items, "raw_html": fr.text})
        step["sampled_list_pages"] = sampled
        step["list_rules"] = rules
        return step

    def _step_6_contents(self, step: dict[str, Any], list_pages: list[dict[str, Any]]) -> dict[str, Any]:
        self.log.info("step6 content pages")
        if step.get("sampled_content_pages") is not None and step.get("content_rules") is not None:
            return step
        pages, rules = step.get("sampled_content_pages", []), step.get("content_rules", [])
        for lp_idx, lp in enumerate(list_pages, start=1):
            for it_idx, it in enumerate(lp.get("items", [])[:3], start=1):
                fr = self._fetch(it["href"])
                self.log.debug("step6[%d.%d] fetch url=%s ok=%s", lp_idx, it_idx, it['href'], fr.ok)
                if not fr.ok:
                    continue
                md = self._make_page_md(fr.url, fr.text)[:15000]
                fields = self._safe_extract("提取title/date/body", md) or {}
                if not isinstance(fields, dict):
                    fields = {}
                soup = BeautifulSoup(fr.text, "html.parser")
                normalized = {"title": str(fields.get("title", "")), "date": str(fields.get("date", "")), "body": str(fields.get("body", ""))}
                rule = self._infer_content_dom_rule(soup, normalized)
                if rule:
                    rules.append(rule)
                pages.append({"url": fr.url, "fields": normalized})
        step["sampled_content_pages"] = pages
        step["content_rules"] = rules
        return step

    def _fetch(self, url: str) -> FetchResult:
        cache_key = f"fetch::{url}"
        cached = self._cache_get(cache_key)
        if isinstance(cached, dict):
            self.log.debug("cache hit fetch: %s", url)
            return FetchResult(
                url=cached.get("url", url),
                ok=bool(cached.get("ok")),
                status_code=cached.get("status_code"),
                text=str(cached.get("text", "") or ""),
                error=cached.get("error"),
            )
        ret: dict[str, Any] = {}
        ok = load_page(url, ret, session=self.session, headers=headers, proxies=proxies_7890, timeout=self.timeout)
        code_raw = str(ret.get("code", "")).split(",")[-1].strip()
        status_code = int(code_raw) if code_raw.isdigit() else None
        fr = FetchResult(url=url, ok=bool(ok), status_code=status_code, text=str(ret.get("content", "") or ""), error=None if ok else str(ret.get("code", "failed")))
        self._cache_set(cache_key, fr.__dict__)
        return fr

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
