# -*- coding: utf-8 -*-
"""
第一个自动化测试文件
====================

测的是 hello_llm.py 里那两个"纯函数"——不联网、不读文件、只做计算的那种。

为什么先测它们？
    因为它们不依赖网络和 API Key，跑一次 0.01 秒，永远不会因为限流而失败。
    自动化测试的第一条原则：**先把计算逻辑和外部依赖分开，然后从计算逻辑开始测。**

运行方式（在 C:\\qiao\\ai 目录下，且已经 activate 过虚拟环境）：
    python -m pytest -v

约定：
    文件名必须以 test_ 开头；函数名也必须以 test_ 开头。两者缺一，pytest 都会视而不见。
"""

import pytest

from hello_llm import clean_answer, summarize_latencies, _clean


# ============================================================
# 一、输出净化：防"假失败"
# ============================================================

def test_净化掉闭合的think标签():
    """实测真实案例：DeepSeek-V4-Flash 回答'1+1'时返回的是 '2</think>2'。"""
    assert clean_answer("2</think>2") == "22"


def test_净化掉开头结尾的空白():
    """实测真实案例：Qwen3.5-9B 返回 '\\n\\n2'，前面挂着一堆换行。"""
    assert clean_answer("\n\n2") == "2"


def test_标签大小写都能识别():
    assert clean_answer("答案<THINK>x</Think>是4") == "答案x是4"


def test_没有杂质时原样返回():
    assert clean_answer("答案是 4") == "答案是 4"


def test_空输入不会崩():
    """模型偶尔会返回空 content，净化函数必须扛得住，不能抛异常。"""
    assert clean_answer("") == ""
    assert clean_answer(None) is None


def test_清理Key文件里的常见脏东西():
    """BOM + 换行 + 误加的引号，这三个是新手 Key 报 401 的三大元凶。"""
    assert _clean("\ufeff sk-abc123 \n") == "sk-abc123"
    assert _clean('"sk-abc123"') == "sk-abc123"
    assert _clean("'sk-abc123'") == "sk-abc123"


# ============================================================
# 二、延迟统计：这是一个"回归测试"
# ============================================================

def test_P95在样本很少时等于最大值():
    """
    这条测试是为了锁住一个真实踩过的坑。

    原来的公式是 int(len*0.95)-1，int() 向下截断，3 条样本会被算成中位数：
        [2.0, 5.0, 27.0] -> 原公式得 5.0，正确答案是 27.0
    也就是说，最慢的那条被藏起来了 —— 而 P95 存在的意义恰恰是暴露长尾。

    现在改成 math.ceil(len*0.95)-1 之后，这条测试会一直守着它，
    以后谁再把公式改回去，测试立刻变红。
    """
    stats = summarize_latencies([2.0, 5.0, 27.0])
    assert stats["p95"] == 27.0


def test_两条样本时P95也是最大值():
    stats = summarize_latencies([2.0, 27.0])
    assert stats["p95"] == 27.0


def test_P50是排序后的中间值():
    stats = summarize_latencies([10.0, 1.0, 5.0, 3.0, 2.0])
    assert stats["p50"] == 3.0          # 排序后 [1,2,3,5,10]，中间是 3


def test_一百分样本时P95落在第95位():
    """样本足够大时，P95 应该精确落在第 95 个位置上（不是最大值）。"""
    stats = summarize_latencies([float(i) for i in range(1, 101)])
    assert stats["p95"] == 95.0
    assert stats["max"] == 100.0


def test_失败的条目要被剔除():
    """
    调用失败的条目耗时是 None。如果不剔除，sorted() 会直接抛
    TypeError: '<' not supported between 'NoneType' and 'float'。
    这是批量跑几百条时必然出现的情况（总会有几条限流失败）。
    """
    stats = summarize_latencies([3.0, None, 1.0, None, 2.0])
    assert stats["count"] == 3
    assert stats["max"] == 3.0


def test_全部失败时不炸():
    stats = summarize_latencies([None, None])
    assert stats["count"] == 0
    assert stats["p50"] is None
    assert stats["p95"] is None


def test_空列表不炸():
    stats = summarize_latencies([])
    assert stats["count"] == 0
    assert stats["p95"] is None


# ============================================================
# 三、留给你的作业：把下面这条补完
# ============================================================


def test_四个值应该完全相等():
    """
    要求：验证"输入顺序不影响统计结果"。
    提示：把同一组数字用两种不同的顺序传进去，断言两个结果完全一样。
    想要更严格，可以用 pytest 的 parametrize 一次覆盖多组数据（第 5 周讲）。
    """
    stats = summarize_latencies([10.0])
    assert stats['max'] == stats['min'] == stats['p95'] == stats['p50']

def test_P95大于等于P50():

    stats = summarize_latencies([20.0, 16.0, 55.0, 2])
    assert stats['p95'] >= stats['p50']

def test_最大值大于等于p95最小值小于等于p50():
    stats = summarize_latencies([22, 66, 10, -60, 33])
    assert stats['max'] >= stats['p95']
    assert stats['min'] <= stats['p50']