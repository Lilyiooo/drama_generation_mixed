import os
import yaml

def post_process(ret):
    ret = (
        ret
        .replace("```json\n", "")
        .replace("\n```", "")
        .strip()
    )
    return ret

def read_txt(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        content = file.readlines()
    return "".join(content)

def read_yaml(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        configs = yaml.safe_load(file)
    return configs


# 模型配置文件路径
MODEL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs.yaml")

def calc_scene_nums_by_per_scene_word(drama_config, min_word_count, max_word_count):
    """
    根据每场字数范围和整集字数上下限，动态计算场次数范围。
    适用于动漫类型逐场生成时的场次数计算。
    
    计算逻辑：
        scene_nums_min = ceil(min_word_count / per_scene_word_max)
        scene_nums_max = ceil(max_word_count / per_scene_word_min)
    
    Args:
        drama_config: 剧集类型配置字典（需包含 per_scene_word_range）
        min_word_count: 每集字数下限
        max_word_count: 每集字数上限
    
    Returns:
        tuple: (scene_nums_min, scene_nums_max)
    """
    import math
    per_scene_config = drama_config.get("per_scene_word_range")
    
    if not per_scene_config or not min_word_count or not max_word_count:
        return drama_config.get("scene_nums_min", 1), drama_config.get("scene_nums_max", 10)
    
    per_scene_min = per_scene_config["min"]
    per_scene_max = per_scene_config["max"]
    
    # 字数少 / 每场多 = 最少场次；字数多 / 每场少 = 最多场次
    scene_nums_min = max(1, math.ceil(min_word_count / per_scene_max))
    scene_nums_max = max(scene_nums_min, math.ceil(max_word_count / per_scene_min))
    
    return scene_nums_min, scene_nums_max



def calc_word_nums_by_word_count(drama_config, min_word_count, max_word_count):
    """
    根据请求中的每集字数上下限生成字数范围字符串。
    如果上下限有效则直接使用，否则降级使用配置默认值。
    
    Args:
        drama_config: 剧集类型配置字典
        min_word_count: 每集字数下限
        max_word_count: 每集字数上限
    
    Returns:
        str: 字数范围字符串，如 "1000-1400"
    """
    if min_word_count and max_word_count and min_word_count > 0 and max_word_count > 0:
        return f"{int(min_word_count)}-{int(max_word_count)}"
    elif min_word_count and min_word_count > 0:
        # 只有下限，上限按下限的1.4倍估算
        return f"{int(min_word_count)}-{int(min_word_count * 1.4)}"
    elif max_word_count and max_word_count > 0:
        # 只有上限，下限按上限的0.7倍估算
        return f"{int(max_word_count * 0.7)}-{int(max_word_count)}"
    else:
        # 都无效，使用配置默认值
        return drama_config.get("word_nums", "300")


def get_model_config(model_type):
    """
    从 model_configs.yaml 中读取模型配置
    
    Args:
        model_type: 模型类型，可选值：
            - "world_view_model"：世界观生成模型
            - "role_model"：人设生成模型
            - "plot_point_model"：卡点规划生成模型
            - "story_outline_model"：故事大纲生成模型
            - "episode_outline_model"：集大纲生成模型
            - "scene_outline_model"：场次大纲生成模型
            - "script_model"：剧本生成模型
            - "auxiliary_model"：辅助模型（JSON格式修正、摘要提取）
    
    Returns:
        dict: 模型配置字典，包含 model, system_prompt, temperature, top_p, top_k, max_length, max_new_tokens
    """
    configs = read_yaml(MODEL_CONFIG_PATH)
    return configs[model_type]


# 剧集类型配置文件路径
DRAMA_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drama_configs.yaml")


def get_drama_config_by_plot_type(plot_type):
    """
    根据 plot_type 获取对应的 drama_config 配置。
    
    Args:
        plot_type: pb.PlotType 枚举值（SHORT=1, CARTOON=2, LONG=3, SHORT_CARTOON=4）
    
    Returns:
        dict: 对应的 drama_config 配置字典
    """
    all_drama_configs = read_yaml(DRAMA_CONFIG_PATH)
    
    # plot_type 到配置 key 的映射
    plot_type_map = {
        1: "short",          # pb.PlotType.SHORT
        2: "cartoon",        # pb.PlotType.CARTOON
        3: "long",           # pb.PlotType.LONG
        4: "short_cartoon",  # pb.PlotType.SHORT_CARTOON
    }
    
    config_key = plot_type_map.get(plot_type, "default")
    return all_drama_configs["drama_types"].get(config_key, all_drama_configs["drama_types"]["default"])


def get_role_desc_by_plot_type(plot_type):
    """
    根据 plot_type 获取对应的角色描述（role_desc），用于提示词中的 RoleDescription 变量。
    
    Args:
        plot_type: pb.PlotType 枚举值
    
    Returns:
        str: 角色描述字符串
    """
    drama_config = get_drama_config_by_plot_type(plot_type)
    return drama_config.get("role_desc", "你是一位资深的商业编剧，擅长创作快节奏、强冲突、视觉冲击力强的剧本。")


def get_drama_type_name_by_plot_type(plot_type):
    """
    根据 plot_type 获取对应的剧集类型名称（如"短剧"、"动漫"等），用于提示词中的 DramaType 变量。
    
    Args:
        plot_type: pb.PlotType 枚举值
    
    Returns:
        str: 剧集类型名称
    """
    drama_config = get_drama_config_by_plot_type(plot_type)
    return drama_config.get("name", "短剧")


def calc_dialogue_ratio(script: str) -> dict:
    """
    计算剧本中台词与旁白的字数比例。
    
    解析规则：
    - 台词行：匹配 "角色名：台词内容" / "角色名OS：" / "角色名VO：" 格式的行
    - 旁白行：以 "△" 开头的行
    - 忽略行：场头行（如 "1-1. 地点 时间 人物：..."）、分隔符（---）、空行、结束标记（第x集完）
    
    Args:
        script: 剧本文本内容
    
    Returns:
        dict: {
            "dialogue_chars": int,      # 台词字数（去除空白）
            "narration_chars": int,     # 旁白字数（去除空白）
            "total_chars": int,         # 台词+旁白总字数
            "dialogue_ratio": float,    # 台词占比（0.0-1.0）
            "narration_ratio": float,   # 旁白占比（0.0-1.0）
            "is_ratio_ok": bool,        # 比例是否达标（台词占比 >= 65%）
        }
    """
    import re
    
    dialogue_chars = 0  # 台词字数
    narration_chars = 0  # 旁白字数
    
    lines = script.strip().split('\n')
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过空行
        if not stripped:
            continue
        
        # 跳过分隔符
        if stripped == '---':
            continue
        
        # 跳过场头行（格式：数字-数字. 地点 时间 人物：...）
        if re.match(r'^\d+-\d+\.', stripped):
            continue
        
        # 跳过结束标记（第x集完）
        if re.match(r'^（第\d+集完）$', stripped):
            continue
        
        # 跳过闪回标记
        if stripped.startswith('【闪回') or stripped.startswith('接'):
            continue
        
        # 计算当前行的有效字数（去除空白）
        line_chars = len(re.sub(r'\s+', '', stripped))
        
        if line_chars == 0:
            continue
        
        # 旁白行：以 △ 开头
        if stripped.startswith('△'):
            narration_chars += line_chars
        # 台词行：匹配 "角色名：" / "角色名OS：" / "角色名VO：" / "角色名（...）：" 格式
        elif re.match(r'^.{1,10}(OS|VO|（[^）]*）)?：', stripped):
            dialogue_chars += line_chars
        else:
            # 无法明确分类的行，归入旁白
            narration_chars += line_chars
    
    total_chars = dialogue_chars + narration_chars
    
    if total_chars == 0:
        return {
            "dialogue_chars": 0,
            "narration_chars": 0,
            "total_chars": 0,
            "dialogue_ratio": 0.0,
            "narration_ratio": 0.0,
            "is_ratio_ok": True,  # 无内容时不触发调整
        }
    
    dialogue_ratio = dialogue_chars / total_chars
    narration_ratio = narration_chars / total_chars
    
    # 台词占比 >= 65% 视为达标（给一定容差，目标是70%，65%以上不触发修复）
    DIALOGUE_RATIO_THRESHOLD = 0.65
    
    return {
        "dialogue_chars": dialogue_chars,
        "narration_chars": narration_chars,
        "total_chars": total_chars,
        "dialogue_ratio": round(dialogue_ratio, 3),
        "narration_ratio": round(narration_ratio, 3),
        "is_ratio_ok": dialogue_ratio >= DIALOGUE_RATIO_THRESHOLD,
    }


def calc_dialogue_ratio_per_scene(script: str) -> dict:
    """
    按场次计算剧本中台词与旁白的字数比例。
    基于 --- 分隔符将整集剧本拆分为多场，对每场分别计算比例。
    
    Args:
        script: 整集剧本文本内容（场次之间用 --- 分隔）
    
    Returns:
        dict: {
            "episode_ratio": dict,          # 整集的比例信息（同 calc_dialogue_ratio 返回值）
            "scene_ratios": list[dict],     # 每场的比例信息列表，每项包含：
                                            #   - scene_index: int (场次序号，从1开始)
                                            #   - scene_header: str (场头信息，用于标识场次)
                                            #   - dialogue_chars: int
                                            #   - narration_chars: int
                                            #   - total_chars: int
                                            #   - dialogue_ratio: float
                                            #   - narration_ratio: float
                                            #   - is_ratio_ok: bool
            "scenes_not_ok": list[dict],    # 比例不达标的场次列表
            "all_scenes_ok": bool,          # 是否所有场次都达标
        }
    """
    import re
    
    # 整集比例
    episode_ratio = calc_dialogue_ratio(script)
    
    # 按 --- 分隔符拆分场次（兼容不同格式的分隔：\n\n\n---\n\n\n 或 \n\n---\n\n 等）
    scenes = re.split(r'\n+---\n+', script.strip())
    
    scene_ratios = []
    scenes_not_ok = []
    
    for idx, scene_text in enumerate(scenes):
        scene_text = scene_text.strip()
        if not scene_text:
            continue
        
        # 提取场头信息（第一行如果匹配场头格式）
        first_line = scene_text.split('\n')[0].strip()
        scene_header = first_line if re.match(r'^\d+-\d+\.', first_line) else f"第{idx+1}场"
        
        # 计算该场的比例
        scene_ratio_info = calc_dialogue_ratio(scene_text)
        
        scene_data = {
            "scene_index": idx + 1,
            "scene_header": scene_header,
            **scene_ratio_info,
        }
        scene_ratios.append(scene_data)
        
        # 收集不达标的场次（排除无内容的场次）
        if not scene_ratio_info["is_ratio_ok"] and scene_ratio_info["total_chars"] > 0:
            scenes_not_ok.append(scene_data)
    
    return {
        "episode_ratio": episode_ratio,
        "scene_ratios": scene_ratios,
        "scenes_not_ok": scenes_not_ok,
        "all_scenes_ok": len(scenes_not_ok) == 0,
    }
