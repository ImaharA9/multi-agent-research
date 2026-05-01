#!/usr/bin/env python3
"""
Multi-Agent Research System — CLI Entry Point

Usage:
    python main.py "你的研究问题"
    python main.py --demo          # 使用预构建响应演示完整流程（无需 API key）
    python main.py --demo --save   # 演示并保存报告到文件
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System — 多 Agent 协作式深度研究",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "Should we use microservices or monolith for an early-stage SaaS?"
    python main.py --demo
    python main.py --demo --save
        """,
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="Research question to investigate",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with simulated agent responses (no API key required)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the final report to report.md",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full reasoning trace",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo(args)
    elif args.query:
        run_live(args)
    else:
        parser.print_help()
        print("\n提示：提供一个研究问题，或用 --demo 查看演示。")


def run_demo(args):
    """Run the full pipeline with simulated agent responses."""
    from orchestrator import ResearchOrchestrator, DEMO_QUERY

    print("\n" + "=" * 60)
    print("  Multi-Agent Research System — DEMO MODE")
    print("  使用预构建响应演示完整的多 Agent 推理链")
    print("=" * 60)

    orch = ResearchOrchestrator()
    result = orch.research(DEMO_QUERY, demo=True)

    print("━" * 60)
    print("  最终报告")
    print("━" * 60)
    print(result["report"])

    if args.verbose:
        print("\n" + "━" * 60)
        print("  推理链路追踪")
        print("━" * 60)
        for i, step in enumerate(result["trace"], 1):
            print(f"\n  Step {i}: {step.get('step', '?')} [{step.get('agent', '?')}]")
            print(f"  Preview: {step.get('content', '')[:300]}...")

    if args.save:
        save_report(result["report"])
        print("\n报告已保存到 report.md")

    print(f"\n[Demo 完成 — 展示了 4 个 Agent 协作完成的长链推理过程]")


def run_live(args):
    """Run with real API calls via a provider-agnostic client."""
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o")

    if not api_key:
        print("错误：未找到 LLM_API_KEY。")
        print("请在 .env 文件中设置以下环境变量：")
        print("  LLM_API_KEY=your-api-key")
        print("  LLM_API_BASE=https://api.openai.com/v1  (可选)")
        print("  LLM_MODEL=gpt-4o                      (可选)")
        print()
        print("或使用 --demo 查看演示。")
        sys.exit(1)

    from orchestrator import ResearchOrchestrator
    from llm_client import OpenAICompatibleClient

    client = OpenAICompatibleClient(api_key=api_key, base_url=api_base, model=model)
    orch = ResearchOrchestrator(client=client)
    result = orch.research(args.query, demo=False)

    print("━" * 60)
    print("  最终报告")
    print("━" * 60)
    print(result["report"])

    if args.verbose:
        print("\n" + "━" * 60)
        print("  推理链路追踪")
        print("━" * 60)
        for i, step in enumerate(result["trace"], 1):
            print(f"\nStep {i}: {step.get('step', '?')} [{step.get('agent', '?')}]")

    print(f"\n总耗时: {result['elapsed_seconds']:.1f}s")

    if args.save:
        save_report(result["report"])


def save_report(report: str):
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
