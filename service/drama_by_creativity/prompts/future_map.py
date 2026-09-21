"""Future Map 及分集贡献映射相关 prompt。"""

GENERATE_FUTURE_MAP_PROMPT = r"""
你是一名长篇连续剧的叙事规划师。请严格依据完整集大纲，从故事最终结局开始向前倒推，建立一棵“未来叙事目标需求分解树”（Future Map）。

## 完整集大纲
{{EpisodeOutline}}

## 核心定义
- 根节点是完整集大纲最终实际达成的核心结局，不是宽泛主题。
- 子节点回答：“为了使父节点在剧情中可信地成立，必须先满足哪些叙事目标或条件？”
- 持续向前分解，直到叶节点成为足够明确的基础铺垫、初始条件、人物动机、关键信息或关系基础。
- 这是一棵按剧情因果和叙事依赖组织的抽象目标树，不是分集目录、时间线或剧情复述。

## 约束
1. 只能使用集大纲已经明确给出的剧情、结局、人物、冲突和因果，不得另创剧情。
2. 不要让一集对应一个节点，也不要在节点中写“第X集”。节点粒度由叙事目标本身决定。
3. 每个非根节点必须恰好有一个 parent_id；父节点依赖子节点，方向是“子目标支撑父目标”。
4. 根节点固定为 `FM001`，parent_id 为 null，depth 为 0；其他节点使用不重复的 `FM002`、`FM003`……，depth 必须等于父节点 depth + 1。
5. 同一父节点下的子目标应相互区分、共同解释父目标成立所需的关键条件；避免同义重复、琐碎动作和空泛表达。
6. goal 用一句明确、可验证的完成状态描述；requirement 说明该目标为何是父目标成立的必要条件；evidence 概括集大纲中支持该节点的具体剧情依据，但不要添加集号。
7. 只输出 JSON，不要输出解释文字。

## 输出格式
```json
{
  "map_type": "backward_requirement_tree",
  "root_id": "FM001",
  "nodes": [
    {
      "node_id": "FM001",
      "parent_id": null,
      "depth": 0,
      "goal": "最终核心结局",
      "requirement": "全剧最终要达成的叙事结果",
      "evidence": "集大纲中的结局依据"
    },
    {
      "node_id": "FM002",
      "parent_id": "FM001",
      "depth": 1,
      "goal": "支撑结局的必要子目标",
      "requirement": "该子目标为何是父目标成立的必要条件",
      "evidence": "集大纲中的剧情依据"
    }
  ]
}
```
"""


GENERATE_EPISODE_CONTRIBUTIONS_PROMPT = r"""
你是一名长篇连续剧的叙事规划师。请严格依据完整集大纲和已经生成的 Future Map，判断每一集实际为哪些未来叙事目标作出贡献。

## 完整集大纲
{{EpisodeOutline}}

## Future Map
{{FutureMap}}

## 核心定义
“贡献”是指本集中的明确剧情内容为某个节点目标提供了必要的铺垫、推进、转折、达成或结果兑现。仅仅人物相同、主题相似或时间上接近，不算贡献。

## 约束
1. 必须为完整集大纲中的每一集输出且只输出一条 episode 记录，按 episode_id 升序排列。
2. 每集可以贡献一个或多个节点，也可以确实不贡献任何节点；不要为了覆盖节点而强行关联。
3. node_id 只能引用给定 Future Map 中真实存在的节点。
4. contribution_type 只能是以下之一：
   - `setup`：建立该目标所需的前提、动机、信息或关系；
   - `advance`：实质推进该目标；
   - `turn`：使该目标发生关键转向、升级或受阻；
   - `complete`：直接达成该节点目标；
   - `payoff`：兑现该节点此前积累的叙事效果。
5. contribution 必须具体说明本集的什么内容如何贡献该节点，不能只复述节点标题。
6. importance 只能是 `primary` 或 `secondary`；每集只将最核心的贡献标为 primary。
7. 同一集不得重复关联同一 node_id。
8. 只能依据集大纲，不得补写大纲中不存在的情节。
9. 只输出 JSON，不要输出解释文字。

## 输出格式
```json
{
  "episodes": [
    {
      "episode_id": 1,
      "contributions": [
        {
          "node_id": "FM010",
          "contribution_type": "setup",
          "importance": "primary",
          "contribution": "本集的具体剧情如何为该目标建立前提"
        }
      ]
    }
  ]
}
```
"""
