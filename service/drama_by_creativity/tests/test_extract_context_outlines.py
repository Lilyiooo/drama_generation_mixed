"""
测试 _extract_context_outlines 函数的单元测试

测试场景：
1. generated_data 为空 -> 返回"暂无"
2. 修改第1集（头部边界）-> 只有后面的集
3. 修改第2集（头部附近）-> 前1集 + 后2集
4. 修改中间集 -> 前2集 + 后2集
5. 修改倒数第2集（尾部附近）-> 前2集 + 后1集
6. 修改最后1集（尾部边界）-> 只有前面的集
7. 只有1集的极端情况 -> 返回"暂无"
8. episode_id 找不到匹配 -> 回退到索引方式

使用方式：
    cd drama_operator/service/drama_by_creativity/tests
    python test_extract_context_outlines.py
"""

import sys
import os

# 设置项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from drama_local import models as pb

# 直接导入被测试的函数
from service.drama_by_creativity.regenerate_episode_outline import _extract_context_outlines


def create_test_episode_outline(num_episodes=10):
    """创建测试用的 GeneratedData，包含指定数量的集大纲。
    
    Args:
        num_episodes: 集数，默认10集
    
    Returns:
        pb.GeneratedData 对象
    """
    episodes = []
    for i in range(num_episodes):
        episode = pb.EpisodeOutline.Episode(
            season_id=0,
            episode_id=i,
            chapter_from=i * 10 + 1,
            chapter_to=(i + 1) * 10,
            content=f"第{i+1}集大纲内容：这是测试用的第{i+1}集剧情描述。"
        )
        episodes.append(episode)
    
    season = pb.EpisodeOutline.Season(
        episodes=episodes,
        season_id=0
    )
    
    episode_outline = pb.EpisodeOutline(seasons=[season])
    generated_data = pb.GeneratedData(episode_outline=episode_outline)
    
    return generated_data


def test_empty_generated_data():
    """测试1: generated_data 为空"""
    print("=" * 60)
    print("测试1: generated_data 为 None")
    result = _extract_context_outlines(None, 3)
    assert result == "暂无", f"期望'暂无'，实际得到: {result}"
    print("✅ 通过: 返回'暂无'")
    
    print("\n测试1b: generated_data.episode_outline 为空")
    empty_data = pb.GeneratedData()
    result = _extract_context_outlines(empty_data, 3)
    assert result == "暂无", f"期望'暂无'，实际得到: {result}"
    print("✅ 通过: 返回'暂无'")
    
    print("\n测试1c: episode_outline.seasons 为空列表")
    empty_outline = pb.GeneratedData(episode_outline=pb.EpisodeOutline(seasons=[]))
    result = _extract_context_outlines(empty_outline, 3)
    assert result == "暂无", f"期望'暂无'，实际得到: {result}"
    print("✅ 通过: 返回'暂无'")


def test_first_episode():
    """测试2: 修改第1集（episode_id=0，头部边界）"""
    print("\n" + "=" * 60)
    print("测试2: 修改第1集（头部边界，episode_id=0）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=0)
    
    print(f"结果:\n{result}\n")
    
    # 应该只有后面的集（后1集、后2集）
    assert "后1集" in result, "应该包含后1集"
    assert "后2集" in result, "应该包含后2集"
    assert "前" not in result, "不应该包含'前'字（因为前面没有集）"
    print("✅ 通过: 只包含后2集的大纲")


def test_second_episode():
    """测试3: 修改第2集（episode_id=1，头部附近）"""
    print("\n" + "=" * 60)
    print("测试3: 修改第2集（头部附近，episode_id=1）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=1)
    
    print(f"结果:\n{result}\n")
    
    # 应该有前1集 + 后2集
    assert "前1集" in result, "应该包含前1集"
    assert "前2集" not in result, "不应该包含前2集（因为前面只有1集）"
    assert "后1集" in result, "应该包含后1集"
    assert "后2集" in result, "应该包含后2集"
    print("✅ 通过: 包含前1集 + 后2集")


def test_middle_episode():
    """测试4: 修改中间集（episode_id=5）"""
    print("\n" + "=" * 60)
    print("测试4: 修改中间集（episode_id=5）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=5)
    
    print(f"结果:\n{result}\n")
    
    # 应该有前2集 + 后2集
    assert "前1集" in result, "应该包含前1集"
    assert "前2集" in result, "应该包含前2集"
    assert "后1集" in result, "应该包含后1集"
    assert "后2集" in result, "应该包含后2集"
    # 不应该包含当前集本身的内容
    assert "第6集大纲内容" not in result, "不应该包含当前集本身的内容"
    print("✅ 通过: 包含前2集 + 后2集，不包含当前集")


def test_second_to_last_episode():
    """测试5: 修改倒数第2集（episode_id=8，尾部附近）"""
    print("\n" + "=" * 60)
    print("测试5: 修改倒数第2集（episode_id=8，尾部附近）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=8)
    
    print(f"结果:\n{result}\n")
    
    # 应该有前2集 + 后1集
    assert "前1集" in result, "应该包含前1集"
    assert "前2集" in result, "应该包含前2集"
    assert "后1集" in result, "应该包含后1集"
    assert "后2集" not in result, "不应该包含后2集（因为后面只有1集）"
    print("✅ 通过: 包含前2集 + 后1集")


def test_last_episode():
    """测试6: 修改最后1集（episode_id=9，尾部边界）"""
    print("\n" + "=" * 60)
    print("测试6: 修改最后1集（episode_id=9，尾部边界）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=9)
    
    print(f"结果:\n{result}\n")
    
    # 应该只有前面的集
    assert "前1集" in result, "应该包含前1集"
    assert "前2集" in result, "应该包含前2集"
    assert "后" not in result, "不应该包含'后'字（因为后面没有集）"
    print("✅ 通过: 只包含前2集的大纲")


def test_single_episode():
    """测试7: 只有1集的极端情况"""
    print("\n" + "=" * 60)
    print("测试7: 只有1集的极端情况")
    
    generated_data = create_test_episode_outline(1)
    result = _extract_context_outlines(generated_data, episode_id=0)
    
    print(f"结果: {result}\n")
    
    # 只有1集，没有前后集可取
    assert result == "暂无", f"期望'暂无'，实际得到: {result}"
    print("✅ 通过: 返回'暂无'")


def test_episode_id_not_found():
    """测试8: episode_id 找不到匹配，回退到索引方式"""
    print("\n" + "=" * 60)
    print("测试8: episode_id 找不到匹配（episode_id=100，超出范围）")
    
    generated_data = create_test_episode_outline(10)
    result = _extract_context_outlines(generated_data, episode_id=100)
    
    print(f"结果: {result}\n")
    
    # episode_id=100 找不到匹配，且100 >= len(all_episodes)=10，所以返回"暂无"
    assert result == "暂无", f"期望'暂无'，实际得到: {result}"
    print("✅ 通过: 返回'暂无'")


def test_two_episodes_modify_first():
    """测试9: 只有2集，修改第1集"""
    print("\n" + "=" * 60)
    print("测试9: 只有2集，修改第1集（episode_id=0）")
    
    generated_data = create_test_episode_outline(2)
    result = _extract_context_outlines(generated_data, episode_id=0)
    
    print(f"结果:\n{result}\n")
    
    # 应该只有后1集
    assert "后1集" in result, "应该包含后1集"
    assert "前" not in result, "不应该包含'前'字"
    print("✅ 通过: 只包含后1集")


def test_two_episodes_modify_last():
    """测试10: 只有2集，修改第2集"""
    print("\n" + "=" * 60)
    print("测试10: 只有2集，修改第2集（episode_id=1）")
    
    generated_data = create_test_episode_outline(2)
    result = _extract_context_outlines(generated_data, episode_id=1)
    
    print(f"结果:\n{result}\n")
    
    # 应该只有前1集
    assert "前1集" in result, "应该包含前1集"
    assert "后" not in result, "不应该包含'后'字"
    print("✅ 通过: 只包含前1集")


def test_content_correctness():
    """测试11: 验证提取的内容是否正确"""
    print("\n" + "=" * 60)
    print("测试11: 验证提取的内容正确性")
    
    generated_data = create_test_episode_outline(5)
    # 修改第3集（episode_id=2），应该取到第1、2、4、5集
    result = _extract_context_outlines(generated_data, episode_id=2)
    
    print(f"结果:\n{result}\n")
    
    # 验证内容
    assert "第1集大纲内容" in result, "应该包含第1集的内容"
    assert "第2集大纲内容" in result, "应该包含第2集的内容"
    assert "第3集大纲内容" not in result, "不应该包含第3集（当前集）的内容"
    assert "第4集大纲内容" in result, "应该包含第4集的内容"
    assert "第5集大纲内容" in result, "应该包含第5集的内容"
    print("✅ 通过: 内容正确，包含前2集和后2集的实际内容")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("  开始测试 _extract_context_outlines 函数")
    print("🧪" * 30 + "\n")
    
    tests = [
        test_empty_generated_data,
        test_first_episode,
        test_second_episode,
        test_middle_episode,
        test_second_to_last_episode,
        test_last_episode,
        test_single_episode,
        test_episode_id_not_found,
        test_two_episodes_modify_first,
        test_two_episodes_modify_last,
        test_content_correctness,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"❌ 失败: {test_func.__name__} - {e}")
        except Exception as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"💥 异常: {test_func.__name__} - {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败, 共 {passed + failed} 个测试")
    
    if errors:
        print("\n失败详情:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
