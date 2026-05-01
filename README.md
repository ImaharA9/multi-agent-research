# Multi-Agent Research System

**多 Agent 协作式深度研究系统** — 使用 4 个专职 AI Agent 模拟人类研究团队的分工协作，将复杂问题通过长链推理转化为结构化深度报告。

## 项目解决的核心痛点

### 单 Agent 研究的三大缺陷

| 问题 | 表现 | 后果 |
|------|------|------|
| **视角单一** | 一个模型一次回答，缺乏多角度审视 | 结论偏颇，遗漏关键反方观点 |
| **无自我纠错** | 模型无法有效检验自己的输出 | 错误在推理链中逐级放大 |
| **浅层回答** | 复杂问题被简化为单步推理 | 缺少深度分析所需的"研究→质疑→补充→综合"循环 |

### 本系统的解决方案

**将人类研究团队的分工模式映射为 Agent 协作架构：**

- **Orchestrator（策划者）**：分解复杂问题 → 生成可执行的研究计划
- **Researcher（研究员）× N**：并行调研子问题，每人专注于一个角度
- **Analyst（分析师）**：交叉审查所有发现，标记矛盾和缺口
- **Synthesizer（综合者）**：将分散发现整合为自包含的最终报告

**核心创新**：不是让一个模型"想得更久"，而是让多个模型"从不同角度想"，然后让另外的模型来**检验**它们的思考。

## 核心逻辑流

```
用户提出复杂研究问题
         │
         ▼
┌─────────────────────┐
│  Phase 1: 问题分解    │  Orchestrator Agent
│  1 个问题 → 2-4 个子问题 │  · 识别隐含假设
│  输出结构化 JSON 计划  │  · 确保 MECE（互斥且穷尽）
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 2: 并行调研    │  N × Researcher Agent（并行执行）
│  每个子问题独立深度调研  │  · 每个 Researcher 有独立视角
│  输出：观点 + 证据 + 反方 │  · ThreadPoolExecutor 并行
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 3: 交叉分析    │  Analyst Agent
│  识别：一致 / 矛盾 / 缺口 │  · 如果发现缺口 → 触发补充调研
│  输出：交叉验证报告     │  · 反馈循环：研究→审查→再研究
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 4: 综合输出    │  Synthesizer Agent
│  生成结构化最终报告     │  · 执行摘要 → 主题分析 → 建议
│                      │  · 明确标注遗留不确定性
└─────────────────────┘
```

### 长链推理体现在哪里？

1. **问题分解链**：复杂问题 → 子问题 → 每个子问题的证据标准（三级展开）
2. **研究-审查反馈环**：Analyst 发现缺口后自动触发补充调研（闭环推理）
3. **全链路追踪**：每一步的输入输出都被记录，推理路径可审计

### 多 Agent 协作体现在哪里？

- **专业化分工**：4 种 Agent 各有独立的 System Prompt 和认知约束
- **并行执行**：Researcher 阶段使用 ThreadPoolExecutor 同时调研
- **对抗性审查**：Analyst 的角色是"挑刺"，而非"润色"

## 快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 演示模式（无需 API Key）

```bash
python main.py --demo
```

使用预构建的 Agent 响应演示完整的多 Agent 协作流程，包括：
- 问题分解
- 3 个 Researcher 的并行调研结果
- Analyst 的交叉分析报告
- 最终综合报告

### 3. 真实 API 模式

```bash
# 设置 API Key（支持任意 OpenAI 兼容接口）
cp .env.example .env
# 编辑 .env 填入你的 API Key 和 endpoint

# 运行你自己的研究
python main.py "Should a startup use microservices or monolith?"

# 保存报告
python main.py --save "Is Rust ready for web backends in 2025?"

# 查看完整推理链路
python main.py --verbose "What are effective strategies for reducing production bugs?"
```

## 项目结构

```
multi-agent-research/
├── README.md           # 项目文档
├── requirements.txt    # Python 依赖
├── .env.example        # API Key 配置模板
├── .gitignore
├── main.py             # CLI 入口
├── agents.py           # Agent 定义（4 个 System Prompt + 调用逻辑）
├── llm_client.py       # LLM 客户端适配器（可替换任意提供商）
└── orchestrator.py     # 多 Agent 编排引擎（协调整条推理链）
```

## 技术架构

- **Agent 调用**：每个 Agent 通过不同 System Prompt 定义角色，LLM 后端可替换（支持任意 OpenAI 兼容 API）
- **并行执行**：`concurrent.futures.ThreadPoolExecutor` 实现多 Researcher 并行调研
- **反馈循环**：Analyst 的缺口检测触发补充调研（最多 2 轮）
- **推理追踪**：全链路 `trace` 记录每步的输入/输出/token 消耗
- **Demo 模式**：使用预构建响应演示完整架构，无需 API Key

## License

MIT
