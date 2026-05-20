from __future__ import annotations

from collections import Counter
from itertools import groupby
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urljoin
from rapidfuzz import fuzz

from bs4 import BeautifulSoup, Tag


@dataclass
class PageRuleResult:
    page_url: str
    ok: bool
    reason: str
    rule: Optional[dict[str, Any]] = None


def infer_list_rules(list_pages: list[dict[str, Any]]) -> dict[str, Any]:
    per_page: list[PageRuleResult] = []
    for lp in list_pages:
        if not lp.get("validated"):
            per_page.append(PageRuleResult(lp.get("url", ""), False, "not_validated"))
            continue
        html = lp.get("raw_html") or ""
        items = lp.get("items") or []
        page_url = lp.get("url", "")
        rule = _infer_list_rule_from_page(html, items, page_url)
        if rule is None:
            per_page.append(PageRuleResult(page_url, False, "insufficient_matches"))
        else:
            per_page.append(PageRuleResult(page_url, True, "ok", rule))

    valid_rules = [x.rule for x in per_page if x.ok and x.rule]
    canonical = _pick_canonical_rule(valid_rules, "root_selector")
    return {
        "per_page": [x.__dict__ for x in per_page],
        "canonical": canonical,
        "valid_count": len(valid_rules),
        "total_count": len(per_page),
    }


def infer_content_rules(content_pages: list[dict[str, Any]]) -> dict[str, Any]:
    per_page: list[PageRuleResult] = []
    for cp in content_pages:
        html = cp.get("raw_html") or ""
        fields = cp.get("fields") or {}
        page_url = cp.get("url", "")
        rule = _infer_content_rule_from_page(html, fields)
        if rule is None:
            per_page.append(PageRuleResult(page_url, False, "insufficient_matches"))
        else:
            per_page.append(PageRuleResult(page_url, True, "ok", rule))
    valid_rules = [x.rule for x in per_page if x.ok and x.rule]
    canonical = _pick_canonical_rule(valid_rules, "content_root_selector")
    return {
        "per_page": [x.__dict__ for x in per_page],
        "canonical": canonical,
        "valid_count": len(valid_rules),
        "total_count": len(per_page),
    }


def _infer_list_rule_from_page(raw_html: str, items: list[dict[str, str]], base_url: str) -> Optional[dict[str, Any]]:
    soup = BeautifulSoup(raw_html, "html.parser")
    matched: list[Tag] = []

    href_items = sorted([i for i in items if i.get("href")], key=lambda i: i["href"])
    href_map: dict[str, dict[str, Any]] = {
        k: {"T": list(g)[0].get("title", ""), "M": []}
        for k, g in groupby(href_items, key=lambda i: i["href"])
    }

    for ai, a in enumerate(soup.select("a[href]")):
        ah = urljoin(base_url, (a.get("href") or "").strip())
        if ah in href_map:
            href_map[ah]["M"].append((a, ai))

    for ad in href_map.values():
        mm = ad.get("M") or []
        if not mm:
            continue
        title = str(ad.get("T", "") or "")

        # 优先直接文本匹配，再退化到全文本匹配；同分时优先更深节点
        direct = []
        for a, ai in mm:
            ds = a.string.strip() if isinstance(a.string, str) else ""
            if not ds:
                continue
            score = fuzz.ratio(title, ds)
            if score >= 80:
                depth = len(list(a.parents))
                direct.append((a, ai, score, depth))

        chosen: Optional[Tag] = None
        if direct:
            chosen = max(direct, key=lambda x: (x[2], x[3], x[1]))[0]
        else:
            full = []
            for a, ai in mm:
                ft = a.get_text(" ", strip=True)
                if not ft:
                    continue
                score = fuzz.ratio(title, ft)
                if score >= 80:
                    depth = len(list(a.parents))
                    full.append((a, ai, score, depth))
            if full:
                chosen = max(full, key=lambda x: (x[2], x[3], x[1]))[0]

        if chosen is None:
            raise ValueError("没有标题和HREF都匹配的超链")
        matched.append(chosen)

    if len(matched) < 2:
        return None
    lca = _lowest_common_ancestor(matched)
    if lca is None:
        return None
    root_selector = _make_unique_root_selector(soup, lca)
    paths = [_css_relative_path(lca, n) for n in matched[:20]]
    branch = _common_branch(paths) or "a[href]"
    branch = _refine_branch_selector(lca, matched[:20], branch)
    return {
        "root_selector": root_selector,
        "item_selector": branch,
        "item_link_selector": "a[href]",
    }


def _infer_content_rule_from_page(raw_html: str, fields: dict[str, str]) -> Optional[dict[str, Any]]:
    soup = BeautifulSoup(raw_html, "html.parser")
    chunks = [c.strip() for c in str(fields.get("body", "")).split("\n") if len(c.strip()) > 12]
    if not chunks:
        return None

    # 通过正文片段匹配段落，再做 LCA；标题/日期不参与该 LCA
    paragraph_nodes = [
        p
        for p in soup.select("article p, main p, div p, p")
        if any(c[:24] in p.get_text(" ", strip=True) for c in chunks[:5])
    ]
    if len(paragraph_nodes) < 2:
        return None

    lca = _lowest_common_ancestor(paragraph_nodes)
    if lca is None:
        return None

    # 优先选择更语义化的正文根（id/class/data-testid 显式指向 article/body/content）
    content_root = _promote_content_root(lca)
    root_selector = _make_unique_root_selector(soup, content_root)
    return {
        "content_root_selector": root_selector,
        "title_selector": "h1",
        "date_selector": "time",
        "paragraph_selector": "p",
    }



def _promote_content_root(node: Tag) -> Tag:
    keywords = ("article", "body", "content", "正文")
    cur: Optional[Tag] = node
    best = node
    while isinstance(cur, Tag):
        nid = str(cur.get("id", "") or "").lower()
        classes = " ".join([c for c in (cur.get("class") or []) if isinstance(c, str)]).lower()
        dt = str(cur.get("data-testid", "") or "").lower()
        text = f"{nid} {classes} {dt}"
        if any(k in text for k in keywords):
            best = cur
            if nid:
                break
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return best

def _make_unique_root_selector(soup: BeautifulSoup, node: Tag) -> str:
    base = _selector_with_identity(node)
    if _safe_select_count(soup, base) == 1:
        return base
    # fallback to full path to guarantee uniqueness as much as possible
    full = _css_path(node)
    if _safe_select_count(soup, full) == 1:
        return full
    return base


def _refine_branch_selector(root: Tag, targets: list[Tag], candidate: str) -> str:
    # 目标：包含所有 targets，且尽量不包含其他元素
    target_ids = {id(x) for x in targets}
    best = candidate
    best_extra = 10**9
    candidates = [candidate, "a[href]", ".//a[@href]"]
    for c in candidates:
        if c == ".//a[@href]":
            nodes = root.select("a[href]")
        else:
            nodes = root.select(c)
        node_ids = {id(x) for x in nodes}
        if not target_ids.issubset(node_ids):
            continue
        extra = len(node_ids - target_ids)
        if extra < best_extra:
            best = c if c != ".//a[@href]" else "a[href]"
            best_extra = extra
    return best


def _safe_select_count(soup: BeautifulSoup, selector: str) -> int:
    try:
        return len(soup.select(selector))
    except Exception:
        return 0

def _pick_canonical_rule(rules: list[dict[str, Any]], key: str) -> Optional[dict[str, Any]]:
    if not rules:
        return None
    counter = Counter([r.get(key, "") for r in rules if r.get(key)])
    if not counter:
        return rules[0]
    best = counter.most_common(1)[0][0]
    for r in rules:
        if r.get(key) == best:
            return r
    return rules[0]


def _common_branch(paths: list[str]) -> str:
    if not paths:
        return ""
    parts = [p.split(" > ") for p in paths if p]
    if not parts:
        return ""
    min_len = min(len(x) for x in parts)
    prefix = []
    for i in range(min_len):
        vals = {x[i] for x in parts}
        if len(vals) == 1:
            prefix.append(parts[0][i])
        else:
            break
    # 如果公共前缀里出现了 id 选择器，id 之前的层级是冗余的
    for i, seg in enumerate(prefix):
        if seg.startswith("#"):
            return " > ".join(prefix[i:])
    return " > ".join(prefix)


def _lowest_common_ancestor(nodes: list[Tag]) -> Optional[Tag]:
    chains = [_ancestor_chain(n) for n in nodes]
    common = set(chains[0])
    for c in chains[1:]:
        common &= set(c)
    for anc in chains[0]:
        if anc in common:
            return anc
    return None


def _ancestor_chain(node: Tag) -> list[Tag]:
    chain = []
    cur: Optional[Tag] = node
    while isinstance(cur, Tag):
        chain.append(cur)
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return chain


def _selector_with_identity(node: Tag) -> str:
    if node.get("id"):
        return f"#{node['id']}"
    classes = [c for c in (node.get("class") or []) if isinstance(c, str) and ':' not in c and '/' not in c]
    if classes:
        return f"{node.name}." + ".".join(classes[:2])
    return _css_path(node)


def _css_relative_path(root: Tag, node: Tag) -> str:
    parts = []
    cur: Optional[Tag] = node
    while isinstance(cur, Tag) and cur is not root:
        seg = _segment(cur)
        parts.append(seg)
        if seg.startswith("#"):
            return " > ".join(reversed(parts))
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    if cur is root:
        parts.append(":scope")
    return " > ".join(reversed(parts))


def _css_path(node: Tag) -> str:
    parts = []
    cur: Optional[Tag] = node
    while isinstance(cur, Tag) and cur.name != "[document]":
        if cur.name in {"html", "body"}:
            cur = cur.parent if isinstance(cur.parent, Tag) else None
            continue
        parts.append(_segment(cur))
        cur = cur.parent if isinstance(cur.parent, Tag) else None
    return " > ".join(reversed(parts))


def _segment(node: Tag) -> str:
    if node.get("id"):
        return f"#{node['id']}"
    cls = [c for c in (node.get("class") or []) if isinstance(c, str) and ':' not in c and '/' not in c]
    if cls:
        return f"{node.name}." + ".".join(cls[:2])
    data_attr = next((k for k in node.attrs if isinstance(k, str) and k.startswith("data-")), None)
    if data_attr:
        return f'{node.name}[{data_attr}="{node.attrs[data_attr]}"]'
    if isinstance(node.parent, Tag):
        sibs = [s for s in node.parent.find_all(node.name, recursive=False)]
        idx = sibs.index(node) + 1 if node in sibs else 1
        return node.name if idx == 1 else f"{node.name}:nth-of-type({idx})"
    return node.name
