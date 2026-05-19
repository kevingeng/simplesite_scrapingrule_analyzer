from __future__ import annotations

import re
import random
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import html2text
from bs4 import BeautifulSoup, Tag
try:
    from selectolax.parser import HTMLParser
except Exception:  # noqa: BLE001
    HTMLParser = None

from tools.http_helper import headers, load_page, proxies_7890
from .models import ArticleModel, FetchResult, ListItemsFitModel, ListPageModel, NavListModel, SiteAnalyzeResult, SiteTypeModel
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
        self.state.setdefault("site_flags", {"with_bypass": False, "bypass_param": None})
        self._rand = random.Random(42)

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
            return model.model_validate(cached) if isinstance(cached, dict) else cached
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
                step["access_check"] = {"ok": True, "status": fr.status_code}
                return step
            access = self._classify_access_failure(fr)
            step["access_check"] = access
            if access.get("need_bypass"):
                bp = (access["reject_type"], access["block_kind"])
                self.state["site_flags"] = {"with_bypass": True, "bypass_param": list(bp)}
                self._persist()
                self.log.info("step2 retry with bypass_param=%s", bp)
                retry_fr = self._fetch(url, bypass_param=bp, force_refresh=True)
                if retry_fr.ok:
                    step["normalized_url"] = retry_fr.url
                    step["home_html"] = retry_fr.text
                    step["access_check"] = {"ok": True, "status": retry_fr.status_code, "bypass_used": True}
                    return step
                step["access_check"]["bypass_failed"] = True
                raise RuntimeError(f"home fetch failed after bypass: {step['access_check']}")
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
            md = self._make_page_md(n["url"], fr.text)[:8000]
            out = self._safe_extract("提取主体列表项title/href，不是列表页返回空", md, ListPageModel)
            items = []
            if out.list_page_type != "不是列表页":
                items = [{"title": i.title, "href": urljoin(n["url"], i.href)} for i in out.list_items]
            fit = self._validate_list_items_fit(items, self.state["step3"]["site_type"])
            if not fit.passed:
                sampled.append({"url": n["url"], "items": items, "raw_html": fr.text, "validated": False, "fit_reason": fit.reason})
                self.log.info("step5[%d] ignored by type-check: %s", idx, fit.reason)
                continue
            self.log.debug("step5[%d] list_page_type=%s items=%d", idx, out.list_page_type if out else None, len(items))
            soup = self._make_soup(fr.text)
            rule = self._infer_list_dom_rule(soup, items, n["url"])
            if rule:
                rules.append(rule)
            sampled.append({"url": n["url"], "items": items, "raw_html": fr.text, "validated": True, "fit_reason": fit.reason})
        if not any(p.get("validated") for p in sampled):
            raise RuntimeError("所有列表页都被类型校验过滤，无法继续")
        step["sampled_list_pages"] = sampled
        step["list_rules"] = rules
        return step

    def _step_6_contents(self, step: dict[str, Any], list_pages: list[dict[str, Any]]) -> dict[str, Any]:
        self.log.info("step6 content pages")
        if step.get("sampled_content_pages") is not None and step.get("content_rules") is not None:
            return step
        pages, rules = step.get("sampled_content_pages", []), step.get("content_rules", [])
        for lp_idx, lp in enumerate([p for p in list_pages if p.get("validated")], start=1):
            for it_idx, it in enumerate(lp.get("items", [])[:3], start=1):
                fr = self._fetch(it["href"])
                self.log.debug("step6[%d.%d] fetch url=%s ok=%s", lp_idx, it_idx, it['href'], fr.ok)
                if not fr.ok:
                    continue
                md = self._make_page_md(fr.url, fr.text)[:15000]
                fields = self._safe_extract("提取title/date/content", md, ArticleModel)
                soup = self._make_soup(fr.text)
                normalized = {"title": fields.title, "date": fields.date, "body": fields.content}
                rule = self._infer_content_dom_rule(soup, normalized)
                if rule:
                    rules.append(rule)
                pages.append({"url": fr.url, "fields": normalized})
        step["sampled_content_pages"] = pages
        step["content_rules"] = rules
        return step

    def _fetch(self, url: str, bypass_param: Optional[tuple[str, str]] = None, force_refresh: bool = False) -> FetchResult:
        if bypass_param is None:
            bp_saved = self.state.get("site_flags", {}).get("bypass_param")
            if isinstance(bp_saved, (list, tuple)) and len(bp_saved) == 2:
                bypass_param = (str(bp_saved[0]), str(bp_saved[1]))
        bp_key = f"{bypass_param[0]}::{bypass_param[1]}" if bypass_param else "none"
        cache_key = f"fetch::{url}::bp::{bp_key}"
        if not force_refresh:
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
        ok = load_page(
            url,
            ret,
            session=self.session,
            headers=headers,
            proxies=proxies_7890,
            timeout=self.timeout,
            bypass_param=bypass_param,
        )
        code_raw = str(ret.get("code", "")).split(",")[-1].strip()
        status_code = int(code_raw) if code_raw.isdigit() else None
        fr = FetchResult(url=url, ok=bool(ok), status_code=status_code, text=str(ret.get("content", "") or ""), error=None if ok else str(ret.get("code", "failed")))
        self._cache_set(cache_key, fr.__dict__)
        return fr

    def _classify_access_failure(self, fr: FetchResult) -> dict[str, Any]:
        status = fr.status_code
        text_low = (fr.text or "").lower()
        if status in {500, 503}:
            return {"ok": False, "status": status, "category": "server_error", "need_bypass": False}
        if status == 404:
            return {"ok": False, "status": status, "category": "not_found", "need_bypass": False}
        if status in {401, 403, 429}:
            block_kind = "generic_waf"
            if "cloudflare" in text_low:
                block_kind = "cloudflare"
            elif "akamai" in text_low:
                block_kind = "akamai"
            elif "incapsula" in text_low or "imperva" in text_low:
                block_kind = "imperva_incapsula"
            elif "captcha" in text_low:
                block_kind = "captcha"
            return {
                "ok": False,
                "status": status,
                "category": "rejected",
                "reject_type": "拒绝访问",
                "block_kind": block_kind,
                "need_bypass": True,
            }
        if fr.error:
            return {"ok": False, "status": status, "category": "connect_error", "need_bypass": False, "error": fr.error}
        return {"ok": False, "status": status, "category": "unknown_error", "need_bypass": False}

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
        return {
            "root_selector": self._selector_with_identity(lca),
            "root_xpath": self._xpath_path(lca),
            "item_branch_selectors": [self._css_path(n) for n in matched[:10]],
            "item_branch_xpaths": [self._xpath_path(n) for n in matched[:10]],
            "item_link_selector": "a[href]",
            "item_link_xpath": ".//a[@href]",
        }

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
        return {
            "content_root_selector": self._selector_with_identity(lca),
            "content_root_xpath": self._xpath_path(lca),
            "title_selector": "h1",
            "title_xpath": ".//h1",
            "date_selector": "time",
            "date_xpath": ".//time",
            "paragraph_selector": "p",
            "paragraph_xpath": ".//p",
        }

    def _validate_list_items_fit(self, items: list[dict[str, str]], site_type: str) -> ListItemsFitModel:
        if not items:
            return ListItemsFitModel(fit_count=0, total_count=0, passed=False, reason="空列表项")
        sample = self._sample_items(items)
        text = "\n".join([f"- {x.get('title','')} | {x.get('href','')}" for x in sample])
        ins = (
            "根据站点类型判断这些列表项是否符合目标新闻采集主题。新闻类关注政治/国际/战争军事/法律犯罪/灾难社会等；"
            "机构类关注政策/法律/新闻公告。返回fit_count,total_count,passed,reason。"
        )
        out = self._safe_extract(ins + f"\n站点类型:{site_type}", text, ListItemsFitModel)
        return out

    def _sample_items(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        n = len(items)
        if n <= 6:
            return items
        if n < 15:
            head, tail, midn = 3, 3, 5
        else:
            head, tail, midn = 5, 5, 5
        head_items = items[:head]
        tail_items = items[-tail:]
        middle = items[head : max(head, n - tail)]
        k = min(midn, len(middle))
        mid_items = self._rand.sample(middle, k) if k > 0 else []
        return head_items + mid_items + tail_items

    def _make_soup(self, html: str) -> BeautifulSoup:
        if HTMLParser is not None:
            try:
                tree = HTMLParser(html)
                return BeautifulSoup(tree.html or html, "html.parser")
            except Exception:  # noqa: BLE001
                pass
        return BeautifulSoup(html, "html.parser")

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
            if cur.name in {"html", "body"}:
                cur = cur.parent if isinstance(cur.parent, Tag) else None
                continue
            token = cur.name
            if cur.get("id"):
                token = f"#{cur['id']}"
                parts.append(token)
                break
            classes = [c for c in (cur.get("class") or []) if isinstance(c, str)]
            if classes:
                token = f"{cur.name}." + ".".join(classes[:2])
            else:
                attr_name = next((k for k in cur.attrs.keys() if isinstance(k, str) and k.startswith("data-")), None)
                if attr_name:
                    token = f'{cur.name}[{attr_name}="{cur.attrs[attr_name]}"]'
                elif isinstance(cur.parent, Tag):
                    sibs = [s for s in cur.parent.find_all(cur.name, recursive=False)]
                    idx = sibs.index(cur) + 1 if cur in sibs else 1
                    token = cur.name if idx == 1 else f"{cur.name}:nth-of-type({idx})"
            parts.append(token)
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return " > ".join(reversed(parts))

    def _xpath_path(self, node: Tag) -> str:
        parts: list[str] = []
        cur: Optional[Tag] = node
        while isinstance(cur, Tag) and cur.name != "[document]":
            if cur.name in {"html", "body"}:
                cur = cur.parent if isinstance(cur.parent, Tag) else None
                continue
            if cur.get("id"):
                parts.append(f'*[@id="{cur["id"]}"]')
                break
            classes = [c for c in (cur.get("class") or []) if isinstance(c, str)]
            if classes:
                parts.append(f'{cur.name}[contains(concat(" ", normalize-space(@class), " "), " {classes[0]} ")]')
            else:
                if isinstance(cur.parent, Tag):
                    sibs = [s for s in cur.parent.find_all(cur.name, recursive=False)]
                    idx = sibs.index(cur) + 1 if cur in sibs else 1
                    parts.append(cur.name if idx == 1 else f"{cur.name}[{idx}]")
                else:
                    parts.append(cur.name)
            cur = cur.parent if isinstance(cur.parent, Tag) else None
        return "//" + "/".join(reversed(parts))
