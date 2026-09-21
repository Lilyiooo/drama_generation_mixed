"""NLP utilities for episode number extraction and scene script processing."""
import re
from cn2an import cn2an  # 需要安装 cn2an 库：pip install cn2an
import re
from typing import Tuple, Optional


def split_episode_and_content(text: str) -> Tuple[Optional[int], str]:
    """
    从字符串中识别“第x集”，然后将其与内容切分。

    函数会查找形如 "第<数字>集" 的模式，并将其与后面的内容分开。
    它能处理 "集" 字后面不同的分隔符，如中文冒号、英文冒号或空格。

    Args:
        text: 包含剧集信息的原始字符串。

    Returns:
        一个元组 (episode_number, content)。
        - 如果找到匹配项, episode_number 是提取出的整数, content 是剩余的文本。
        - 如果未找到匹配项, episode_number 是 None, content 是完整的原始文本。

    示例:
    >>> s = '第18集：洛桐对每晚...'
    >>> num, content = split_episode_and_content(s)
    >>> num
    18
    >>> content
    '洛桐对每晚...'
    """
    # 正则表达式模式解释:
    # ^           - 匹配字符串的开头 (可选，但通常更精确)
    # 第          - 匹配文字 "第"
    # (\d+)       - 捕获组1: 匹配并捕获一个或多个数字 (这就是我们想要的x)
    # 集          - 匹配文字 "集"
    # [:：\s]*    - 匹配分隔符：可以是英文冒号(:), 中文冒号(：), 或任何空白字符(\s)。
    #               星号(*)表示匹配0次或多次，这使得分隔符是可选的。
    # (.*)        - 捕获组2: 贪婪匹配剩余的所有字符 (这就是内容)
    #               re.DOTALL 标志让 . 也能匹配换行符
    pattern = r'^第(\d+)集[:：\s]*(.*)'
    
    # re.match() 从字符串的开头开始匹配，如果你的"第x集"总是在开头的话，用它更好。
    # 如果可能在中间，就用 re.search()。这里我们假设它在开头。
    match = re.match(pattern, text, re.DOTALL)
    
    if match:
        # group(1) 是第一个括号里的内容，即数字
        episode_number_str = match.group(1)
        episode_number = int(episode_number_str)
        
        # group(2) 是第二个括号里的内容，即正文
        content = match.group(2).strip() # .strip() 用于移除可能存在的前后多余空格
        
        return episode_number, content
    else:
        # 如果没有找到匹配项，返回None和原始文本
        return None, text


def extract_episode_number(text: str) -> Optional[int]:
    """
    从字符串中识别并提取“第x集”中的数字x。

    这个函数使用正则表达式来查找 "第" 和 "集" 之间的数字部分。

    Args:
        text: 包含剧集信息的字符串。

    Returns:
        如果找到匹配的剧集编号，则返回该编号（作为整数）。
        如果没有找到，则返回 None。
    
    示例:
    >>> s = '第18集：洛桐对每晚仅止于接吻的关系感到不满...'
    >>> extract_episode_number(s)
    18
    >>> s = '这是一个不包含剧集信息的故事。'
    >>> extract_episode_number(s)
    None
    """
    # 正则表达式模式：
    # r'...'   - 表示这是一个原始字符串 (raw string)，推荐用于正则表达式
    # 第       - 匹配文字“第”
    # (\d+)    - 这是一个捕获组（用括号括起来）。
    #   \d     - 匹配任意一个数字 (0-9)
    #   +      - 匹配前面一个元素（\d）一次或多次。这能处理像 "1", "18", "102" 这样的多位数。
    # 集       - 匹配文字“集”
    pattern = r'第(\d+)集'
    
    # re.search() 会在字符串中查找第一个匹配该模式的位置
    match = re.search(pattern, text)
    
    # 如果找到了匹配项
    if match:
        # match.group(0) 是整个匹配的字符串，例如 '第18集'
        # match.group(1) 是第一个捕获组（括号里的内容），例如 '18'
        episode_number_str = match.group(1)
        return int(episode_number_str)
    else:
        # 如果没有找到匹配项，返回 None
        return None

def extract_number(text):
    """Extract episode number from various formats.

    Args:
        text: Input string that may contain episode number in formats like
              '9.', '第9集', '第九集'

    Returns:
        Integer episode number, or -1 if not found.
    """
    # 先尝试直接提取阿拉伯数字
    # '9.' '第9集' '第九集' 提取其中的数字，并转换成阿拉伯的9
    num = re.findall(r'\d+', text)
    if num:
        return int(num[0])
    # 再尝试提取中文数字
    cn_num = re.findall(r'第?([一二三四五六七八九十百千万零〇两]+)[集\.]?', text)
    if cn_num:
        try:
            return cn2an(cn_num[0], "cn")
        except (ValueError, TypeError):
            pass
    return -1


def fix_scenescript(scenescript, epsi, scenid):
    """Fix and normalize scene script format.

    Args:
        scenescript: Original scene script text.
        epsi: Episode ID or number.
        scenid: Scene ID or number.

    Returns:
        Formatted scene script with episode-scene prefix.
    """
    # 判断是否以"数字-数字"开头
    if re.match(r'^\d+-\d+\s*', scenescript):
        # 删除开头的"数字-数字"
        scenescript = re.sub(r'^\d+-\d+\s*', '', scenescript)
    # 在前面加上"epsi-scenid"
    return f"{epsi}-{scenid} {scenescript}"


if __name__ == "__main__":
    print(extract_number('9.'))      # 输出: 9
    print(extract_number('第9集'))   # 输出: 9
    print(extract_number('第九集'))  # 输出: 9
