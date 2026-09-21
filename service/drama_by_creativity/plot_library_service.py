"""
剧情桥段库服务模块

提供桥段库查询和格式化功能，用于集大纲生成时从桥段库检索参考剧情。

设计说明：
- 每次请求独立查询桥段库，无状态设计
- 前3集大纲生成和后续集大纲生成时各自独立调用 search_plot_from_library() 查询
"""

PLOT_SEARCH_SIZE = 5
PLOT_SEARCH_SIZE_OUTLINE = 20
PLOT_SEARCH_TYPE = "scene_summary"


async def search_plot_from_library(ctx, query, size=PLOT_SEARCH_SIZE, plot_type=PLOT_SEARCH_TYPE):
    """Local generation has no external trope retrieval."""
    return []


def format_plot_references(plots: list) -> str:
    """
    将召回的情节列表格式化为参考文本，用于注入到集大纲生成的 reference 中。
    
    参数:
        plots: 从情节库中召回的情节列表
    
    返回:
        str: 格式化后的参考文本
    """
    if not plots:
        return ""
    
    reference_parts = []
    reference_parts.append("## 情节库参考剧情\n以下是从情节库中召回的相关剧情片段，可以作为本集剧本创作的参考和灵感来源：\n")
    
    for i, plot in enumerate(plots, 1):
        # PlotInfo proto 字段: plot_text(剧情摘要), bridge_name(桥段名称), 
        # bridge_plot(桥段概述), script_name(剧集名称), script_text(剧本原文), score
        plot_text = plot.get("plot_text", "")
        bridge_name = plot.get("bridge_name", "")
        bridge_plot = plot.get("bridge_plot", "")
        script_name = plot.get("script_name", "")
        score = plot.get("score", 0)
        
        part = f"### 参考片段 {i}"
        if bridge_name:
            part += f"（{bridge_name}）"
        part += "\n"
        if script_name:
            part += f"来源: 《{script_name}》\n"
        if bridge_plot:
            part += f"桥段概述: {bridge_plot}\n"
        part += f"剧情摘要: {plot_text}\n"
        reference_parts.append(part)
    
    return "\n".join(reference_parts)
