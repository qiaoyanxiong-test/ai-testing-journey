# -*- coding: utf-8 -*-
"""
这个文件本身什么都不做，只负责一件事：让 pytest 知道"项目的根目录在这儿"。

pytest 默认只把测试文件所在的目录加进 import 搜索路径，那个目录是 tests\，
而 hello_llm.py 在它的上一层。根目录放一个 conftest.py，pytest 就会把根目录
也加进搜索路径，于是测试文件里可以直接写 `from hello_llm import ...`。

文件名不能改，conftest.py 是 pytest 的约定。
"""
