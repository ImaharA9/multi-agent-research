"""
Multi-Agent Research Orchestrator

This is the core engine that coordinates the long-chain reasoning pipeline:

    Query → [Orchestrator: Decompose] → [N × Researcher: Parallel investigation]
    → [Analyst: Cross-reference & gap detection] → [Synthesizer: Final report]

Key design decisions:
- Researchers run in PARALLEL (ThreadPoolExecutor) because sub-questions are independent
- Analyst has a FEEDBACK LOOP: if gaps are found, it triggers follow-up research
- Each phase's output is the next phase's input, forming an explicit reasoning chain

The reasoning chain is not hidden inside a single model call — it's ARCHITECTURALLY
ENFORCED through the agent pipeline. This means the chain is inspectable, debuggable,
and auditable at every step.

PROVIDER-AGNOSTIC: The orchestrator accepts any client object with a chat() method.
No dependency on any specific AI provider SDK.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from agents import call_agent


# ── Simulated responses for demo mode ───────────────────────────────────────
# When no API key is available, these pre-built responses demonstrate the full
# reasoning chain architecture without making real API calls.

DEMO_QUERY = "一家初创公司应该在 2025 年选择 Next.js 还是纯 React 来构建 SaaS 产品？"

DEMO_PLAN = [
    {"id": 1, "question": "Next.js 相比纯 React 在 SaaS 场景下的性能优势（SSR/SSG/ISR）具体数据如何？", "focus": "性能数据"},
    {"id": 2, "question": "Next.js 的学习成本、招聘难度和长期维护成本对比纯 React 是怎样的？", "focus": "团队与成本"},
    {"id": 3, "question": "2025 年 Next.js 生态的成熟度：插件、部署平台、社区支持是否足够稳定用于商业产品？", "focus": "生态成熟度"},
]

DEMO_FINDINGS = {
    1: """(a) 直接答案：Next.js 在 SaaS 场景下的 SSR/SSG 可以带来首屏加载时间 40-60% 的改善，但需要正确配置缓存策略。
(b) 支持证据：Vercel 2024 年度报告显示，使用 ISR 的 Next.js 站点 LCP（最大内容绘制）中位数从 3.2s 降到 1.4s。Shopify、Notion 等商业产品均使用 Next.js 的混合渲染。
(c) 反对证据：如果 SaaS 产品主要是登录后的 dashboard（SPA 模式），SSR 带来的 SEO 优势有限。此时纯 React + Vite 的构建速度（~200ms HMR）远快于 Next.js（~2s HMR）。
(d) 置信度：高。性能差异在 B2C 场景显著，在 B2B 后台场景较小。""",

    2: """(a) 直接答案：Next.js 的初期学习成本比纯 React 高约 30-50%，但长期维护成本在中等以上项目中有优势。
(b) 支持证据：Next.js 内置了路由、图片优化、代码分割、中间件等，这些在纯 React 中需要额外选型和维护。2025 年 Stack Overflow 调查显示 React 开发者中约 65% 接触过 Next.js。招聘市场上要求 Next.js 的岗位同比增长 40%。
(c) 反对证据：Next.js 的"魔法"（自动缓存、RSC）可能变成调试地狱。Reddit r/nextjs 上每月有 200+ 帖子询问缓存相关问题。纯 React 的简单性意味着任何 React 开发者可以立刻上手，不依赖特定框架知识。
(d) 置信度：中等。取决于团队现有经验——如果团队已经熟悉服务端渲染概念，学习成本大幅降低。""",

    3: """(a) 直接答案：2025 年 Next.js 生态已经足够成熟，但在快速迭代中仍有 breaking changes 风险。
(b) 支持证据：npm 下载量 Next.js 已超过纯 React。所有主流部署平台（Vercel、Netlify、AWS Amplify、Cloudflare Pages）均提供 Next.js 原生支持。Auth.js、Prisma、tRPC 等 SaaS 核心工具都有官方 Next.js 集成。
(c) 反对证据：Next.js 14→15 的升级中，部分中间件 API、缓存默认行为发生了 breaking changes。对于需要长期稳定运行的产品，框架的激进迭代可能是负担。纯 React 的 API 稳定性自 18.0 以来保持了更好的向后兼容记录。
(d) 置信度：中高。生态成熟但需锁定版本号，规划升级路径。""",
}

DEMO_ANALYSIS = """## 交叉分析报告

### 收敛发现（3/3 研究者一致）
1. **SSR 在 B2C SaaS 中有明确优势**：研究者 1 的性能数据和研究者 3 的生态支持都指向 Next.js 在面向终端用户的 SaaS 中占优。
2. **团队能力是关键变量**：研究者 2 和 3 都提到，已有 React 经验的团队迁移成本可控，但纯新手团队建议从纯 React 开始理解 SPA 概念。

### 分歧点
- 研究者 1 认为性能优势是"决定性"的，研究者 2 认为"取决于场景"。
  → 这是视角差异而非矛盾：1 从技术指标出发，2 从团队成本出发。都需要纳入最终报告。

### 发现缺口
- 缺少对**替代方案**的系统对比（如 Remix、Astro）。虽然不在此次研究范围内，但如果推荐 Next.js 应该至少提及替代选项的取舍。
- 缺少对**小团队 vs 大团队**的细化分析。

### 完整性评分：8/10
主要缺口是替代方案对比，但不影响核心问题的回答质量。建议最终报告中提及 Remix 和 Astro 作为参考坐标。"""

DEMO_REPORT = """# SaaS 前端框架选择报告：Next.js vs 纯 React（2025）

## 执行摘要

对于 2025 年构建 SaaS 产品的初创公司，**Next.js 是多数场景下的推荐选择**，但附带重要条件。如果你的产品面向终端用户（B2C 或 PLG 模式），Next.js 的 SSR/SSG 能力带来的 SEO 和性能红利通常是决定性的。如果你的产品是重度后台型 B2B 工具且团队缺乏服务端渲染经验，纯 React + Vite 的简单性和稳定性可能更适合。最终决策应优先考虑**团队现有能力**和**用户获取渠道**两个变量。

## 主题分析

### 性能：Next.js 领先，但有场景局限
Next.js 的混合渲染可以在 B2C 场景下将首屏加载时间缩短 40-60%（Vercel 2024 报告）。但 B2B dashboard 类产品由于主要内容在登录后，SSR 优势有限，此时 Next.js 的构建速度劣势反而成为负担。

### 团队与成本：短期 vs 长期的权衡
Next.js 的初始学习坡度高 30-50%，但其内置的工程化能力减少了对第三方库的依赖，降低了长期维护中的"依赖漂移"风险。招聘市场上 Next.js 需求增长 40%（2025 Stack Overflow），但目前纯 React 开发者的绝对数量仍远多于 Next.js。

### 生态：成熟但需锁定版本
主流部署平台和 SaaS 工具链已经充分支持 Next.js。但框架的快速迭代意味着需要规划升级路径——建议锁定主版本号，在非关键期进行大版本迁移。

## 关键建议

1. B2C SaaS → Next.js（SSR/SEO 的红利实在太大）
2. B2B 重度后台 + 小团队 → 纯 React + Vite（简单就是力量）
3. 混合模式也值得考虑：营销站点用 Next.js，App 用纯 React（需要额外的运维成本）
4. 无论选哪个，锁定依赖版本比选框架本身更重要

## 遗留不确定性

- Remix 和 Astro 在某些场景下可能是更好的选择，但生态和招聘资源尚不及 Next.js
- 如果 React Server Components 成为行业标准，纯 React 团队可能需要二次学习，届时 Next.js 的早期投入会变成先发优势
"""


# ── The Orchestrator ────────────────────────────────────────────────────────

class ResearchOrchestrator:
    """
    Coordinates a multi-agent research pipeline with long-chain reasoning.

    The reasoning chain:
      Step 1: Orchestrator decomposes query → structured research plan
      Step 2: N Researchers execute in parallel → individual findings
      Step 3: Analyst cross-references → gaps identified
      Step 4: (Optional) Follow-up research on gaps
      Step 5: Synthesizer produces final report

    Accepts any LLM client with a chat(system, user, max_tokens) -> str method.
    """

    def __init__(
        self,
        client: Any = None,
        max_research_rounds: int = 2,
    ):
        self.client = client
        self.max_research_rounds = max_research_rounds
        self.trace: list[dict] = []

    def research(self, query: str, demo: bool = False) -> dict:
        """
        Execute the full research pipeline.

        Returns a dict with:
          - report: final synthesized report (str)
          - trace: step-by-step reasoning chain (list)
          - elapsed_seconds: wall-clock time (float)
        """
        start = time.time()
        self.trace = []

        print(f"\n{'='*60}")
        print(f"  研究问题: {query}")
        print(f"{'='*60}\n")

        # ── Step 1: Decompose ──
        print("▶ Phase 1/4: Orchestrator 分解问题...")
        plan = self._step_plan(query, demo)
        print(f"  拆分为 {len(plan)} 个子问题:")
        for p in plan:
            print(f"    · {p['question'][:60]}...")
        print()

        # ── Step 2: Parallel Research ──
        print(f"▶ Phase 2/4: {len(plan)} 个 Researcher 并行调研...")
        findings = self._step_research(plan, demo)
        for qid, text in findings.items():
            print(f"  Researcher #{qid} 完成 ({len(text)} 字符)")
        print()

        # ── Step 3: Cross-Analysis ──
        print("▶ Phase 3/4: Analyst 交叉分析...")
        analysis = self._step_analyze(query, plan, findings, demo)

        # Check if follow-up needed
        if not demo and self._needs_followup(analysis) and self.max_research_rounds > 1:
            print("  ⚠ 发现缺口，启动补充调研...")
            followup_questions = self._extract_followup(analysis)
            if followup_questions:
                followup_findings = self._step_research(followup_questions, demo)
                findings.update(followup_findings)
                analysis = self._step_analyze(query, plan, findings, demo)
                print("  补充调研完成\n")
        print()

        # ── Step 4: Synthesize ──
        print("▶ Phase 4/4: Synthesizer 生成最终报告...")
        report = self._step_synthesize(query, findings, analysis, demo)
        print(f"  报告完成 ({len(report)} 字符)\n")

        elapsed = time.time() - start

        print(f"{'='*60}")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"{'='*60}\n")

        return {
            "report": report,
            "trace": self.trace,
            "elapsed_seconds": elapsed,
        }

    # ── Private step methods ─────────────────────────────────────────────

    def _step_plan(self, query: str, demo: bool) -> list[dict]:
        if demo:
            return DEMO_PLAN

        result = call_agent(
            self.client, "orchestrator",
            f"Research question: {query}\n\nOutput your plan as JSON.",
        )
        plan = self._parse_json(result.content)
        self.trace.append({"step": "plan", "agent": "orchestrator", "content": result.content[:500]})
        return plan

    def _step_research(self, plan: list[dict], demo: bool) -> dict[int, str]:
        findings: dict[int, str] = {}

        if demo:
            return dict(DEMO_FINDINGS)

        def research_one(item):
            qid = item["id"]
            question = item["question"]
            result = call_agent(
                self.client, "researcher",
                f"Sub-question [{item.get('focus', 'general')}]: {question}",
            )
            return qid, result

        with ThreadPoolExecutor(max_workers=len(plan)) as executor:
            futures = {executor.submit(research_one, p): p for p in plan}
            for future in as_completed(futures):
                qid, result = future.result()
                findings[qid] = result.content
                self.trace.append({
                    "step": "research", "agent": "researcher",
                    "sub_question_id": qid,
                    "content": result.content[:500] + "...",
                })

        return findings

    def _step_analyze(
        self, query: str, plan: list[dict], findings: dict[int, str], demo: bool
    ) -> str:
        if demo:
            return DEMO_ANALYSIS

        findings_text = "\n\n---\n\n".join(
            f"Researcher #{qid}:\n{text}" for qid, text in findings.items()
        )
        prompt = (
            f"Original question: {query}\n\n"
            f"Sub-questions asked:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Research findings:\n{findings_text}\n\n"
            "Cross-reference these findings. Identify agreements, contradictions, and gaps."
        )

        result = call_agent(self.client, "analyst", prompt)
        self.trace.append({"step": "analyze", "agent": "analyst", "content": result.content[:500]})
        return result.content

    def _step_synthesize(
        self, query: str, findings: dict[int, str], analysis: str, demo: bool
    ) -> str:
        if demo:
            return DEMO_REPORT

        findings_text = "\n\n---\n\n".join(
            f"Finding #{qid}:\n{text}" for qid, text in findings.items()
        )
        prompt = (
            f"Original question: {query}\n\n"
            f"Individual research findings:\n{findings_text}\n\n"
            f"Cross-analysis report:\n{analysis}\n\n"
            "Write the final comprehensive report."
        )

        result = call_agent(self.client, "synthesizer", prompt)
        self.trace.append({"step": "synthesize", "agent": "synthesizer", "content": result.content[:500]})
        return result.content

    # ── Helpers ──────────────────────────────────────────────────────────

    def _parse_json(self, text: str) -> list[dict]:
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [{"id": 1, "question": text[:200], "focus": "general"}]

    def _needs_followup(self, analysis: str) -> bool:
        indicators = ["缺口", "missing", "gap", "未覆盖", "不足", "incomplete"]
        analysis_lower = analysis.lower()
        return any(ind in analysis_lower for ind in indicators)

    def _extract_followup(self, analysis: str) -> list[dict]:
        result = call_agent(
            self.client, "orchestrator",
            f"From this analysis, extract any follow-up research questions as JSON:\n\n{analysis}",
        )
        return self._parse_json(result.content)
