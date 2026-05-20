from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from tools.llm_client import MyLLMClient
from ssa.analyzer import SiteRuleAnalyzer
from ssa.utils import make_logger, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="SimpleSite ScrapingRule Analyzer CLI")
    parser.add_argument("--site-name", required=True)
    parser.add_argument("--site-url", required=True)
    parser.add_argument("--work-root", default=str(ROOT / "work_sites"))
    parser.add_argument("--force-restep", type=int, default=0)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--llm-name", default="max")
    args = parser.parse_args()

    logger = make_logger(args.log_level)
    site_dir = Path(args.work_root) / args.site_name
    site_dir.mkdir(parents=True, exist_ok=True)

    lmc = MyLLMClient(model=MyLLMClient.LLM_MODELS_DIC[args.llm_name])
    analyzer = SiteRuleAnalyzer(lmc=lmc, work_dir=site_dir, log=logger, pool_size=args.pool_size)
    result = analyzer.analyze_site(args.site_url, force_restep=args.force_restep)

    final_rules = {
        "input_url": result.input_url,
        "normalized_url": result.normalized_url,
        "site_type": result.site_type,
        "list_rules": result.list_rules,
        "content_rules": result.content_rules,
        "errors": result.errors,
    }
    save_json(site_dir / "result.json", final_rules)
    logger.info("done: %s", site_dir / "result.json")


if __name__ == "__main__":
    main()
