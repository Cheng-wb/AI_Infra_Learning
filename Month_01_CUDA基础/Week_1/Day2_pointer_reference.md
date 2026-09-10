# Day 2：指针与引用

> **今日目标：** 掌握指针（地址）与引用（别名），理解三种传参方式（值/指针/引用）的差异，会画内存布局。这是写 CUDA 的前提——CUDA 里到处是指针。

## 任务清单

### 任务 1：取址与解引用
写 `pointer_basic.cpp`：

```cpp
#include <iostream>

int main() {
    int x = 42;
    int* p = &x;          // & 取地址，p 存的是 x 的内存地址
    std::cout << "x = " << x << "\n";
    std::cout << "&x = " << &x << "\n";
    std::cout << "p  = " << p << "\n";       // p 打印出来是地址
    std::cout << "*p = " << *p << "\n";      // * 解引用，*p 就是 x 本身

    *p = 100;            // 通过指针修改 x
    std::cout << "after *p=100, x = " << x << "\n";   // x 变成 100
    return 0;
}
```

**要记住：** `int* p` 声明一个「指向 int 的指针」；`&x` 拿地址；`*p` 解引用（通过地址读写值）。指针本身也占内存，存的是一个地址值。

### 任务 2：指针运算与数组
写 `pointer_arith.cpp`，验证 `a[i]` 等价于 `*(a+i)`：

```cpp
#include <iostream>

int main() {
    int a[5] = {10, 20, 30, 40, 50};
    int* p = a;                 // 数组名退化为指向首元素的指针
    std::cout << "a[0] = " << a[0] << ", *p = " << *p << "\n";
    std::cout << "a[2] = " << a[2] << ", *(a+2) = " << *(a + 2) << "\n";

    for (int i = 0; i < 5; ++i)
        std::cout << "*(a+" << i << ") = " << *(a + i) << "\n";
    return 0;
}
```

**要记住：** 数组在内存里是连续排布；`a[i]` 只是 `*(a+i)` 的语法糖。理解「连续内存 + 指针偏移」是理解 CUDA 全局内存访问（M1 Week 3 的合并访问）的地基。

### 任务 3：空指针与指针初始化
写 `nullptr.cpp`：

```cpp
#include <iostream>

int main() {
    int* p = nullptr;          // 空指针，不指向任何内存
    // *p = 1;                 // 取消注释会崩溃（解引用空指针）
    if (p == nullptr) std::cout << "p is null\n";

    int x = 5;
    p = &x;
    if (p != nullptr) std::cout << "p points to " << *p << "\n";
    return 0;
}
```

**要记住：** 未初始化或悬空的指针是 C++ 最常见的 bug 来源；`nullptr` 表示「不指向任何东西」，解引用它是未定义行为（会崩）。

### 任务 4：三种传参方式（核心）
写 `passing.cpp`，对比三种 `swap`：

```cpp
#include <iostream>

// 按值传递：交换的是副本，外面不变
void swap_by_value(int a, int b) {
    int tmp = a; a = b; b = tmp;
}

// 按指针传递：拿到地址，能真正交换
void swap_by_pointer(int* a, int* b) {
    int tmp = *a; *a = *b; *b = tmp;
}

// 按引用传递：引用是别名，能真正交换，写法最干净
void swap_by_reference(int& a, int& b) {
    int tmp = a; a = b; b = tmp;
}

int main() {
    int x = 3, y = 7;

    swap_by_value(x, y);
    std::cout << "value:     x=" << x << ", y=" << y << "\n";   // 3, 7（没交换）

    swap_by_pointer(&x, &y);
    std::cout << "pointer:   x=" << x << ", y=" << y << "\n";   // 7, 3

    swap_by_reference(x, y);
    std::cout << "reference: x=" << x << ", y=" << y << "\n";   // 3, 7（又换回来）

    return 0;
}
```

**要记住：** 三者区别——按值传「拷贝」；按指针传「地址」；按引用传「别名」。后两者能修改原变量。CUDA kernel 里常通过指针把结果写回，理解指针传参是必须的。

### 任务 5：引用 vs 指针
写 `ref_vs_ptr.cpp`：

```cpp
#include <iostream>

int main() {
    int x = 10;
    int& r = x;      // r 是 x 的别名，初始化后绑定 x，不能改绑
    r = 20;          // 等价于 x = 20
    std::cout << "x = " << x << "\n";   // 20

    int y = 30;
    int* p = &x;     // 指针可以重新指向别的变量
    p = &y;
    std::cout << "*p = " << *p << "\n"; // 30

    return 0;
}
```

**要记住：** 引用是「别名」，初始化后不能改变绑定对象；指针是「地址变量」，可以重新指向。引用更安全（不能为空），指针更灵活。

## 练习与验收

1. 手动画出 `int a[5] = {10,20,30,40,50};` 在内存中的布局图（连续 5 个 int，每个 4 字节，首地址记为 `&a[0]`），标注 `a`、`&a[0]`、`*(a+3)` 分别是什么。
2. 写 `array_sum.cpp`：函数 `int sum(int* arr, int n)` 用指针遍历数组求和（`for` 里用 `*(arr+i)`），主函数传入 `{1,2,3,4,5}`，验证结果 15。
3. 写 `double_it.cpp`：函数 `void double_it(int& x)` 用引用把参数翻倍，验证原变量被修改。
4. 预测并验证：`swap_by_value` 为什么交换不了？（把答案写进笔记。）

## 自检

- [ ] 能解释 `&`（取址）和 `*`（解引用）各是什么。
- [ ] 能写出 `a[i] == *(a+i)` 并解释。
- [ ] 能解释三种传参（值/指针/引用）的差异，并说出 `swap` 为什么用指针或引用才有效。
- [ ] 知道 `nullptr` 是什么，解引用空指针会发生什么。
- [ ] 能画出数组的内存布局图。
- [ ] 能区分引用和指针（别名 vs 地址变量）。

> 指针是 C++ 的「分水岭」。今天务必把「地址 / 解引用 / 传参」练到不假思索，后面 CUDA 里 `float* d_out`、`&d_out`、`*d_out` 会反复出现。
