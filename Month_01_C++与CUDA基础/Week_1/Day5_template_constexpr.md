# Day 5：模板与 constexpr

> **今日目标：** 掌握函数模板 / 类模板 / 模板特化 / 重载决议 / `constexpr` / `static`。模板是 C++ 泛型编程的核心，CUTLASS（M4/M8 你会用到）就是重度模板元编程的库，今天先打基础。

## 任务清单

### 任务 1：函数模板
写 `func_template.cpp`：

```cpp
#include <iostream>

template <typename T>
T max_of(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    std::cout << max_of(3, 7) << "\n";          // 实例化为 int 版
    std::cout << max_of(3.14, 2.71) << "\n";    // 实例化为 double 版
    std::cout << max_of('a', 'z') << "\n";      // 实例化为 char 版
    return 0;
}
```

**要记住：** 模板不是「运行时多态」，而是**编译期**为每种类型各生成一份代码（实例化）。你写一份 `max_of`，编译器生成 `int` 版、`double` 版、`char` 版。这是零运行时开销的泛型。

### 任务 2：类模板
写 `class_template.cpp`：

```cpp
#include <iostream>
#include <string>

template <typename T>
class Box {
public:
    explicit Box(T v) : value_(v) {}
    T get() const { return value_; }
private:
    T value_;
};

int main() {
    Box<int>    ib(42);
    Box<double> db(3.14);
    Box<std::string> sb("hello");

    std::cout << ib.get() << ", " << db.get() << ", " << sb.get() << "\n";
    return 0;
}
```

**要记住：** 类模板让一个类能容纳不同类型。你之前用的 `std::vector<int>`、`std::unique_ptr<int>` 都是类模板。CUTLASS 里几乎每个核心类型都是类模板（用 tile 尺寸、数据类型做模板参数）。

### 任务 3：函数重载与重载决议
写 `overload.cpp`：

```cpp
#include <iostream>

void print(int x)    { std::cout << "int: " << x << "\n"; }
void print(double x) { std::cout << "double: " << x << "\n"; }
void print(int x, int y) { std::cout << "two ints: " << x << "," << y << "\n"; }

int main() {
    print(1);          // 选 int 版
    print(1.0);        // 选 double 版
    print(1, 2);       // 选两个参数版
    return 0;
}
```

**要记住：** 重载 = 同名函数，靠参数类型/个数区分。编译器根据实参「决议」调用哪个。模板 + 重载可以组合出非常灵活的接口。

### 任务 4：模板特化
写 `specialization.cpp`：

```cpp
#include <iostream>
#include <cstring>

template <typename T>
T max_of(T a, T b) { return (a > b) ? a : b; }

// 全特化：const char* 不能按指针地址比大小，要按字符串内容比
template <>
const char* max_of(const char* a, const char* b) {
    return (std::strcmp(a, b) > 0) ? a : b;
}

int main() {
    std::cout << max_of(3, 7) << "\n";                    // 7
    const char* s = max_of("apple", "banana");            // "banana"
    std::cout << s << "\n";
    return 0;
}
```

**要记住：** 特化 = 对某个具体类型给出「特殊实现」。上面的例子如果不特化，`max_of("apple","banana")` 会比较两个**指针地址**（错误）；特化后按字符串内容比。

### 任务 5：`constexpr`（编译期计算）
写 `constexpr.cpp`：

```cpp
#include <iostream>

constexpr int square(int x) { return x * x; }

int main() {
    constexpr int a = square(5);          // 编译期就算好，等于常量 25
    int arr[square(4)];                   // 可以用作数组大小（编译期常量）
    std::cout << "a = " << a << "\n";
    std::cout << "arr size = " << sizeof(arr)/sizeof(arr[0]) << "\n"; // 16
    return 0;
}
```

**要记住：** `constexpr` 表示「能在编译期求值」。结果可直接用作模板参数、数组大小等需要编译期常量的地方。现代 CUDA 代码常用 `constexpr` 定义 tile 尺寸、block 大小。

### 任务 6：`static`
写 `static.cpp`：

```cpp
#include <iostream>

int counter() {
    static int n = 0;    // 函数内 static：只初始化一次，跨调用保留
    return ++n;
}

int main() {
    std::cout << counter() << "\n";   // 1
    std::cout << counter() << "\n";   // 2
    std::cout << counter() << "\n";   // 3
    return 0;
}
```

**要记住：** `static` 三个常见含义：①函数内局部 `static`（只初始化一次、生命周期到程序结束）；②类内 `static` 成员（属于类而非某个对象）；③全局 `static`（限制在本文件可见，Day 6 会用到）。

## 练习与验收

1. 写 `sum_template.cpp`：函数模板 `T sum(const std::vector<T>& v)` 求和，分别对 `vector<int>` 和 `vector<double>` 调用，验证结果。
2. 写 `pair_template.cpp`：类模板 `Pair<A,B>` 存两个不同类型的值（如 `Pair<int,std::string>`），带 `first()`/`second()` 访问。
3. 写 `area.cpp`：函数重载 `double area(double r)`（圆）和 `double area(double w, double h)`（矩形），验证重载决议。
4. 回答（写进笔记）：模板和重载的区别？模板是「一份代码多种类型」，重载是「多份同名代码不同签名」。

## 自检

- [ ] 能手写函数模板并解释「编译期实例化」。
- [ ] 能手写类模板并实例化不同类型。
- [ ] 能解释函数重载的决议规则。
- [ ] 能写模板全特化并解释为什么 `const char*` 需要特化。
- [ ] 会用 `constexpr` 做编译期常量。
- [ ] 能说出 `static` 的三种含义。

> 模板今天先「会用」即可。等你 M4/M8 读 CUTLASS 时，会看到 `template <typename ElementA, typename LayoutA, ...>` 这种深度模板元编程——今天理解「模板 = 编译期代码生成」就够用了。
