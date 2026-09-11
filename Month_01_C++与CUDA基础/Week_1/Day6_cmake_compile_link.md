# Day 6：CMake 与编译链接

> **今日目标：** 理解「声明 vs 定义」「头文件 vs 源文件」，掌握多文件 `g++` 编译与 CMake 构建，认识静态库/动态库，能定位「未定义引用 / 重复定义」两个经典报错。

## 任务清单

### 任务 1：搭建多文件项目目录

先建立如下目录结构（今天所有任务都用它）：

```text
Week_1/Day6/
├── CMakeLists.txt
├── include/
│   └── math_utils.h
└── src/
    ├── math_utils.cpp
    └── main.cpp
```

### 任务 2：头文件（声明）与源文件（定义）

`include/math_utils.h`（**只声明**，不写函数体）：

```cpp
#pragma once            // 防止头文件被重复包含

int add(int a, int b);
int multiply(int a, int b);
```

`src/math_utils.cpp`（**定义**，写函数体）：

```cpp
#include "math_utils.h"

int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
```

`src/main.cpp`（使用）：

```cpp
#include <iostream>
#include "math_utils.h"   // 只需包含声明

int main() {
    std::cout << add(3, 4) << "\n";       // 7
    std::cout << multiply(3, 4) << "\n";  // 12
    return 0;
}
```

**要记住：**
- **声明**（declaration）告诉编译器「有这个东西，长这样」；**定义**（definition）是真正生成代码/分配内存。
- 头文件放**声明**，源文件放**定义**。多个 `.cpp` 可以 `#include` 同一个头文件，但一个函数的定义只能有一处。
- `#pragma once` 防止头文件被重复包含（等价于传统 `#ifndef` include guard）。

### 任务 3：手写多文件编译（理解编译 vs 链接）

分步执行，理解「先编译成目标文件，再链接」：

```bash
cd Week_1/Day6

# 第一步：编译（.cpp → .o，只检查语法和声明，不检查定义在哪）
g++ -c src/math_utils.cpp -o math_utils.o -Iinclude
g++ -c src/main.cpp       -o main.o        -Iinclude

# 第二步：链接（把多个 .o 拼成可执行文件，检查所有定义都能找到）
g++ main.o math_utils.o -o day6

# 运行
./day6
```

**要记住：** `-c` 只编译不链接；`-Iinclude` 指定头文件搜索路径；`-o` 指定输出。**编译错误**（语法错）在第一步报，**链接错误**（找不到定义）在第二步报。

### 任务 4：用 CMake 构建

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.16)
project(day6 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(day6
    src/main.cpp
    src/math_utils.cpp
)
target_include_directories(day6 PRIVATE include)
```

构建（在 `Day6/` 下）：

```bash
mkdir build && cd build
cmake ..
make            # Linux/AutoDL
# Windows + MinGW：cmake -G "MinGW Makefiles" .. 然后 mingw32-make
./day6
```

**要记住：** CMake 是跨平台构建工具，声明「源文件 + 头文件路径 + 标准」，生成 Makefile/VS 工程。后面所有 CUDA 项目都用 CMake 组织（M1 Week 4 会加 CUDA 支持）。

### 任务 5：静态库与动态库

```bash
cd Week_1/Day6

# 静态库（.a）：编译进可执行文件里
g++ -c src/math_utils.cpp -o math_utils.o -Iinclude
ar rcs libmath_utils.a math_utils.o
g++ src/main.cpp -Iinclude -L. -lmath_utils -o main_static
./main_static

# 动态库（.so，Linux）：运行时才加载
g++ -shared -fPIC src/math_utils.cpp -Iinclude -o libmath_utils.so
g++ src/main.cpp -Iinclude -L. -lmath_utils -o main_dynamic
LD_LIBRARY_PATH=. ./main_dynamic
```

**要记住：** 静态库编译时把代码拷进可执行文件；动态库运行时加载（`.so`/`.dll`）。`-L.` 指定库搜索路径，`-lmath_utils` 对应 `libmath_utils.a` 或 `.so`。

### 任务 6：制造并修复两个经典报错

1. **未定义引用（undefined reference）**：把 `math_utils.cpp` 里的 `multiply` 函数删掉，重新链接。你会看到 `undefined reference to 'multiply(int, int)'`——这是「声明了但没定义」。恢复它。

2. **重复定义（multiple definition）**：在 `main.cpp` 顶部也写一个 `int add(int,int){...}` 定义，重新链接。你会看到 `multiple definition of 'add(int,int)'`——这是「同一符号定义了两处」。删掉它。

把这两个报错信息截图/记进笔记，这是 C++ 最常见的两类链接错误。

## 练习与验收

1. 把 Day 4 的 `File` 类拆成 `file.h`（声明）+ `file.cpp`（定义）+ `main.cpp`（使用），用 CMake 构建。
2. 写一个 `add.h` 用传统 include guard（`#ifndef` / `#define` / `#endif`）重写，替代 `#pragma once`，验证效果一样。
3. 回答（写进笔记）：为什么函数体不能写在头文件里（除非是 `inline` 或模板）？——因为头文件被多个 `.cpp` 包含会导致重复定义。

## 自检

- [ ] 能区分「声明」和「定义」。
- [ ] 能手动 `g++ -c` + `g++` 分步编译链接一个多文件项目。
- [ ] 能用 CMake 构建项目。
- [ ] 能创建并使用静态库（`.a`）和动态库（`.so`）。
- [ ] 能定位并修复 `undefined reference` 和 `multiple definition`。
- [ ] 理解 `#pragma once` / include guard 的作用。

> 编译链接是「从代码到可运行程序」的完整链路。今天搭好的 CMake 多文件结构，M1 Week 4 会原样扩展成「CUDA + CMake」，是你所有后续项目的骨架。
