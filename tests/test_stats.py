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
# ---- A. 模型输出杂质 → 用 clean_answer ----
# 注意：这里只能放"模型答案里的杂质"（think 标签、首尾空白）。
# Key 文件的脏东西（BOM、引号）属于另一个函数 _clean，见下面 B 组。
@pytest.mark.parametrize("dirty, expected", [
    ("2</think>2", "22"),                    # 实测：DeepSeek-V4-Flash 真实返回
    ("\n\n2", "2"),                          # 实测：Qwen3.5-9B 真实返回
    ("答案<THINK>x</Think>是4", "答案x是4"),   # 标签大小写混合
    ("答案是 4", "答案是 4"),                  # 无杂质时原样返回
    ("", ""),                                # 空输入不崩
    (None, None),                            # 模型偶尔返回 null，也不能崩
], ids=["think标签", "首尾换行", "标签大小写", "无杂质", "空字符串", "None"])
def test_净化模型输出(dirty, expected):
    assert clean_answer(dirty) == expected


# ---- B. Key 文件脏东西 → 用 _clean ----
# BOM + 换行 + 误加的引号，这三个是新手 Key 报 401 的三大元凶。
@pytest.mark.parametrize("dirty, expected", [
    ("\ufeff sk-abc123 \n", "sk-abc123"),    # BOM + 空格
    ('"sk-abc123"', "sk-abc123"),            # 误加了双引号
    ("'sk-abc123'", "sk-abc123"),            # 误加了单引号
], ids=["BOM与空白", "双引号", "单引号"])
def test_清理Key文件的脏东西(dirty, expected):
    assert _clean(dirty) == expected


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


# ------------------------------------------------------------
# 第 5 周改造（一）："精确值"类 → 子集字典 + parametrize
# ------------------------------------------------------------
# 规矩：parametrize 第二栏必须是**数据**，不能是判断表达式。
# 期望值也不该是"完完整整一个 dict"，而是"我这次关心的那几个键"——
# 只比对的键，其余的键不参与，以后新增字段不会把老用例连带打挂。
@pytest.mark.parametrize(
    "data, expected",
    [
        ([2.0, 27.0],                          {"p95": 27.0}),                  # 2 条样本，P95 必须是最大值
        ([10.0, 1.0, 5.0, 3.0, 2.0],           {"p50": 3.0}),                   # 排序后 [1,2,3,5,10]，中间是 3
        ([float(i) for i in range(1, 101)],    {"p95": 95.0, "max": 100.0}),    # 大样本落在第 95 位
        ([3.0, None, 1.0, None, 2.0],          {"count": 3, "max": 3.0}),       # 失败条目(None)要被剔除
        ([None, None],                         {"count": 0, "p50": None, "p95": None}),   # 全失败不炸
        ([],                                   {"count": 0, "p95": None}),      # 空列表不炸
    ],
    ids=["2条样本P95取最大", "P50取中间值", "100条样本落第95位", "剔除失败条目", "全部失败不炸", "空列表不炸"],
)
def test_统计结果(data, expected):
    """只断言 expected 里写出来的键，值必须精确相等。"""
    stats = summarize_latencies(data)
    for key, want in expected.items():
        # 加一句失败说明：断言挂了的时候，你会直接看到"哪个键、期望多少、实际多少"
        assert stats[key] == want, f"{key} 期望 {want}，实际 {stats[key]}"


# ------------------------------------------------------------
# 第 5 周改造（二）："关系"类 → 检查函数 + parametrize
# ------------------------------------------------------------
# 有些断言期望的不是某个具体数字，而是"两个值的大小关系"。这种把**判断本身**
# 当成数据传进去就是最省事的写法——Python 里函数也是普通的值，可以放进列表。
@pytest.mark.parametrize(
    "data, check",
    [
        ([10.0], lambda s: s["p95"] == s["p50"] ==s["max"] == s["min"]),
        ([20.0, 16.0, 55.0, 2.0], lambda s: s["p95"] >= s["p50"]),
        ([22.0, 66.0, 10.0, 33.0], lambda s: s["max"] >= s["p95"] >= s["p50"] >= s["min"]),
        ([5.0, 5.0, 5.0],         lambda s: s["max"] == s["min"]),
    ],
    ids=["单个值四个相等", "P95不小于P50", "大小顺序成立", "全部相同"],
)
def test_统计不变量(data, check):
    """不管输入什么数据，这些大小关系永远成立——这叫"不变量"（invariant）。"""
    assert check(summarize_latencies(data))

@pytest.fixture
def sample_latencies():
    return [10.0, 1.0, 5.0, 3.0, 2.0]

def test_count(sample_latencies):
    stats = summarize_latencies(sample_latencies)
    assert stats["count"] == 5