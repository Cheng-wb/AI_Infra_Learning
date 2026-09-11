# Day 7：周收口（Matrix/Buffer 模块 + 单测）

> **今日目标：** 用一周所学做一个完整的综合项目——「矩阵类 + 单元测试」，覆盖类、模板、RAII、值语义、编译链接，并建立「CPU 参考实现对拍」的习惯（这正是你后面验证每个 CUDA kernel 的方法）。

## 任务清单

### 任务 1：搭项目结构

```text
Week_1/Day7/
├── CMakeLists.txt
├── include/
│   └── matrix.h
├── src/
│   └── matrix.cpp
└── tests/
    └── test_matrix.cpp
```

### 任务 2：实现矩阵类（头文件）

`include/matrix.h`：

```cpp
#pragma once
#include <cstddef>
#include <vector>

class Matrix {
public:
    Matrix(std::size_t rows, std::size_t cols);

    double& operator()(std::size_t r, std::size_t c);
    double  operator()(std::size_t r, std::size_t c) const;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    Matrix add(const Matrix& other) const;       // 逐元素相加
    Matrix multiply(const Matrix& other) const;  // 矩阵乘法

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;   // 行主序存储，data_[r*cols_+c] 对应 (r,c)
};
```

### 任务 3：实现矩阵类（源文件）

`src/matrix.cpp`：

```cpp
#include "matrix.h"
#include <stdexcept>

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    return data_[r * cols_ + c];       // 行主序：第 r 行第 c 列
}
double Matrix::operator()(std::size_t r, std::size_t c) const {
    return data_[r * cols_ + c];
}

Matrix Matrix::add(const Matrix& o) const {
    if (rows_ != o.rows_ || cols_ != o.cols_)
        throw std::invalid_argument("add: shape mismatch");
    Matrix m(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i)
        m.data_[i] = data_[i] + o.data_[i];
    return m;
}

Matrix Matrix::multiply(const Matrix& o) const {
    if (cols_ != o.rows_)
        throw std::invalid_argument("multiply: shape mismatch");
    Matrix m(rows_, o.cols_);
    for (std::size_t i = 0; i < rows_; ++i)
        for (std::size_t j = 0; j < o.cols_; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < cols_; ++k)
                s += (*this)(i, k) * o(k, j);   // 三重循环：GEMM 的最朴素形态
            m(i, j) = s;
        }
    return m;
}
```

**重点理解：** `multiply` 里的三重循环 `i,j,k` 就是矩阵乘法（GEMM）的**最朴素形态**。你 3 个月后在 M3 写的 CUDA GEMM，本质就是在 GPU 上并行化这三重循环。今天先记住这个 CPU 版，它就是将来 CUDA 版本的「CPU 参考实现」。

### 任务 4：写单元测试（对拍）

`tests/test_matrix.cpp`：

```cpp
#include "matrix.h"
#include <cassert>
#include <cmath>
#include <iostream>

bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) < eps;
}

int main() {
    // 准备 2x2 矩阵
    Matrix a(2, 2), b(2, 2);
    a(0,0)=1; a(0,1)=2; a(1,0)=3; a(1,1)=4;
    b(0,0)=5; b(0,1)=6; b(1,0)=7; b(1,1)=8;

    // 测试 add
    Matrix c = a.add(b);
    assert(approx(c(0,0), 6));
    assert(approx(c(0,1), 8));
    assert(approx(c(1,0), 10));
    assert(approx(c(1,1), 12));

    // 测试 multiply（手算结果）
    // [1 2] [5 6]   [19 22]
    // [3 4] [7 8] = [43 50]
    Matrix d = a.multiply(b);
    assert(approx(d(0,0), 19));
    assert(approx(d(0,1), 22));
    assert(approx(d(1,0), 43));
    assert(approx(d(1,1), 50));

    // 测试形状不匹配抛异常
    bool threw = false;
    try {
        Matrix e(2, 3);
        a.multiply(e);   // 2x2 * 2x3 不匹配
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "all tests passed\n";
    return 0;
}
```

**重点理解：**
- `assert(条件)` 条件为假就中止——这是最简单的单测。
- `approx` 用容差 `1e-9` 比较浮点数（**永远不要用 `==` 比较浮点数**）。
- 测试「抛异常」用 `try/catch` 包裹，验证异常确实被抛出。
- 这套「手算期望值 → assert 对拍」的模式，就是你后面验证每个 CUDA kernel 正确性的模板（GPU 结果 vs CPU 参考）。

### 任务 5：用 CMake 构建并运行测试

`CMakeLists.txt`：

```cmake
cmake_minimum_required(VERSION 3.16)
project(day7 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(matrix src/matrix.cpp)
target_include_directories(matrix PUBLIC include)

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PRIVATE matrix)

enable_testing()
add_test(NAME matrix_test COMMAND test_matrix)
```

构建运行：

```bash
cd Week_1/Day7
mkdir build && cd build
cmake ..
make
./test_matrix                 # 应输出 all tests passed
ctest                         # 用 CTest 跑测试
```

### 任务 6：写本周总结笔记

写一份 `Week_1/Week1_总结.md`，包含：
1. 一周学到的知识点清单（类型/指针/STL/RAII/模板/编译链接）。
2. 你踩过的坑（如溢出、`undefined reference`、双重释放等）。
3. 一句话回答「C++ 和 Python 最本质的区别是什么」。
4. 用「矩阵类的 `multiply` 三重循环」记下：这就是你 3 个月后要并行化的 GEMM。

## 自检（Week 1 完成标准）

- [ ] 能独立写出 `Matrix` 类的头文件 + 源文件 + 测试，CMake 构建通过、测试全绿。
- [ ] 能解释 `operator()` 的行主序索引 `r*cols+c`。
- [ ] 理解矩阵乘法三重循环，并知道它将来对应 CUDA GEMM。
- [ ] 会用 `assert` + 浮点容差做对拍，会用 `try/catch` 测异常。
- [ ] 一周所有 `Day1`–`Day6` 的自检项都能打勾。

> 恭喜完成 M1 Week 1。你已经从「Python 思维」切换到「C++ 思维」：理解内存、值语义、指针、RAII、模板、编译链接。下周（Week 2）开始进入 CUDA 编程模型——你会发现前面所有东西（指针、内存、值语义、连续数组）都在为 GPU 编程铺路。
