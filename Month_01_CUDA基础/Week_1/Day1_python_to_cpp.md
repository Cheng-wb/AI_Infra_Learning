# Day 1：从 Python 到 C++

> **今日目标：** 建立 C++ 基本语法认知，纠正三个关键思维差异（变量是内存 / 值语义 / 静态类型），能写出并编译运行第一个 C++ 程序。

## 环境准备（先做，只此一次）

C++ 只需一个编译器，不需要 GPU。三选一：

- **AutoDL 4090（推荐，Linux）**：镜像里自带 `g++`，直接 `g++ --version` 确认。
- **本地 Windows**：装 [w64devkit](https://github.com/skeeto/w64devkit)（解压即用）或 MinGW-w64，把 `g++` 加进 PATH；或装 WSL。
- **本地有 VS**：用 `cl`，但本计划统一按 `g++` 写命令。

验证命令（任选一个环境）：

```bash
g++ --version      # 应看到 GCC 版本
```

## 任务清单

### 任务 1：第一个 C++ 程序
1. 新建目录 `Month_01_CUDA基础/Week_1/`，写 `hello.cpp`：

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CUDA!" << std::endl;
    return 0;
}
```

2. 编译运行：

```bash
g++ hello.cpp -o hello
./hello            # Linux/AutoDL
hello.exe          # Windows（若用 MinGW）
```

3. 逐行理解：`#include <iostream>` 引入标准输入输出；`main` 是程序入口；`std::cout` 是输出流；`return 0` 表示正常结束。

### 任务 2：基本类型与字节数
写 `types.cpp`，打印每种类型的字节数（`sizeof`）：

```cpp
#include <iostream>
#include <cstdint>

int main() {
    std::cout << "int:     " << sizeof(int) << " bytes\n";
    std::cout << "float:   " << sizeof(float) << " bytes\n";
    std::cout << "double:  " << sizeof(double) << " bytes\n";
    std::cout << "char:    " << sizeof(char) << " bytes\n";
    std::cout << "bool:    " << sizeof(bool) << " bytes\n";
    std::cout << "size_t:  " << sizeof(std::size_t) << " bytes\n";
    std::cout << "int64_t: " << sizeof(std::int64_t) << " bytes\n";
    return 0;
}
```

**要记住：** `int` 通常 4 字节、`float` 4 字节、`double` 8 字节、`char` 1 字节。对比 Python 里 `int` 是任意精度大整数，C++ 的 `int` 是定长、会溢出。

### 任务 3：`const` / `auto` / 命名空间
写 `basics.cpp`：

```cpp
#include <iostream>

int main() {
    const int N = 10;          // 编译期常量，不可修改
    auto x = 3.14;             // auto 自动推导为 double
    auto y = 42;               // 推导为 int

    // N = 20;                 // 取消注释会编译报错，体会 const 的约束

    std::cout << "x = " << x << ", y = " << y << "\n";
    std::cout << "N = " << N << "\n";
    return 0;
}
```

**要记住：** `std::` 是标准库的命名空间，避免名字冲突；`const` 表示「只读」，是 C++ 里高频关键字；`auto` 让编译器推导类型，但类型仍是静态确定的。

### 任务 4：值语义（纠正 Python 的引用直觉）
这是 Python → C++ 最关键的一步。写 `value_semantics.cpp`：

```cpp
#include <iostream>
#include <vector>

int main() {
    // 基本类型：赋值 = 拷贝一份
    int a = 5;
    int b = a;      // b 是独立的一块内存，拷贝了 a 的值
    b = 10;
    std::cout << "a = " << a << ", b = " << b << "\n";  // a=5, b=10

    // 容器也一样是值语义（Python list 是引用语义，这里完全不同）
    std::vector<int> v1 = {1, 2, 3};
    std::vector<int> v2 = v1;   // 深拷贝
    v2[0] = 99;
    std::cout << "v1[0] = " << v1[0] << ", v2[0] = " << v2[0] << "\n"; // v1[0]=1, v2[0]=99

    return 0;
}
```

**要记住：** C++ 里 `=` 默认是**拷贝（值语义）**，不是 Python 的「引用同一个对象」。变量名就是一块内存的别名，赋值是把值复制过去。

### 任务 5：控制流与函数
写 `fib.cpp`（迭代版斐波那契，避免递归爆栈）：

```cpp
#include <iostream>

int fib(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int c = a + b;
        a = b;
        b = c;
    }
    return b;
}

int main() {
    for (int i = 0; i <= 20; ++i)
        std::cout << "fib(" << i << ") = " << fib(i) << "\n";
    return 0;
}
```

对比 Python 版你熟悉的写法，注意三点差异：函数参数和返回值都要写类型；`for` 用 `int` 计数；`++i` 是自增。

## 练习与验收

1. 写 `factorial.cpp`，用函数 `int fact(int n)` 计算阶乘，打印 `fact(0)` 到 `fact(12)`。核对 `fact(12) = 479001600`。
2. 写 `swap_by_value.cpp`：先写一个「按值传递」的 `swap(int a, int b)` 交换函数，观察主函数里的两个变量**没有**被交换——写下你看到的输出，理解「按值传递 = 拷贝」。（Day 2 会解决这个问题。）
3. 故意把 `int` 溢出：写 `int x = 2147483647; std::cout << x + 1;`，观察结果是负数——理解定长整数会溢出（对比 Python 不会）。

## 自检

- [ ] 能独立编译运行一个 C++ 程序（hello.cpp）。
- [ ] 能说出 `int/float/double/char` 的字节数。
- [ ] 能解释「值语义」：`int b = a;` 后改 `b` 不影响 `a`。
- [ ] 能解释为什么 C++ 的 `int` 会溢出而 Python 不会。
- [ ] 能写出带 `const` / `auto` 的小程序。
- [ ] 能写出迭代版斐波那契，并知道 Python 版和 C++ 版的三处差异。

> 今天的核心不是「会写语法」，而是**真正理解「变量是一块内存」**。理解了这个，Day 2 的指针就是水到渠成。
