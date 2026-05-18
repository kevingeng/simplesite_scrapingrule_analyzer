"""SiteRuleAnalyzer 调用示例（AP News）

运行前准备：
1) 确保项目里有可用的 `http_helper.py` 和 `llm_client.py`
2) `llm_client.py` 里有 `MyLLMClient`，并可正常调用 `extract`

执行：
    python example_apnews_usage.py
"""

from pprint import pprint

from llm_client import MyLLMClient
from site_rule_analyzer import SiteRuleAnalyzer


def main() -> None:
    # 你可以按自己的模型配置替换这里
    lmc = MyLLMClient(model=MyLLMClient.LLM_MODELS_DIC["qwen3-max"])

    analyzer = SiteRuleAnalyzer(
        lmc=lmc,
        list_page_limit=3,
        content_per_list_limit=3,
        timeout=(5, 20),
    )

    result = analyzer.analyze_site("https://apnews.com/")

    print("\n=== 基本信息 ===")
    print("input_url:", result.input_url)
    print("normalized_url:", result.normalized_url)
    print("site_type:", result.site_type)
    print("errors:", result.errors)

    print("\n=== 首页导航（前10）===")
    pprint(result.homepage_navs[:10])

    print("\n=== 抽样列表页 ===")
    for i, lp in enumerate(result.sampled_list_pages, 1):
        print(f"[{i}] {lp.get('url')} | nav_label={lp.get('nav_label')} | items={len(lp.get('items', []))}")

    print("\n=== 列表规则 ===")
    pprint(result.list_rules)

    print("\n=== 正文规则 ===")
    pprint(result.content_rules)

    print("\n=== 抽样正文页数量 ===")
    print(len(result.sampled_content_pages))


if __name__ == "__main__":
    main()
