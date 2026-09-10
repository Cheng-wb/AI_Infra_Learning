# Day 3：数组与 STL

> **今日目标：** 掌握 C++ 标准库最常用的容器（vector/array/string/map），理解栈 vs 堆内存，并动手实现一个简化版 `vector` 来理解其扩容机制。

## 任务清单

### 任务 1：`std::vector`（动态数组，最常用）
写 `vector.cpp`：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> v = {1, 2, 3};
    v.push_back(4);                    // 尾部追加
    std::cout << "size=" << v.size()
              << " capacity=" << v.capacity() << "\n";

    for (int x : v) std::cout << x << " ";   // 范围 for 遍历
    std::cout << "\n";

    v[0] = 100;                        // 下标访问
    std::cout << "v[0]=" << v[0] << ", v.at(1)=" << v.at(1) << "\n";
    return 0;
}
```

**要记住：** `size()` 是元素个数，`capacity()` 是已分配容量（≥ size）。`operator[]` 不检查越界，`at()` 会抛异常。对比 Python 的 `list`：`vector` 是连续内存、只能存同一类型、值语义。

### 任务 2：`std::array`（定长数组）
写 `array.cpp`：

```cpp
#include <iostream>
#include <array>

int main() {
    std::array<int, 4> a = {1, 2, 3, 4};   // 长度编译期固定
    for (int i = 0; i < a.size(); ++i)
        std::cout << a[i] << " ";
    std::cout << "\n";
    return 0;
}
```

**要记住：** `std::array` 是定长、栈上、比裸数组 `int a[4]` 更安全（带 `size()`、可整体赋值）。适合长度已知的小数组。

### 任务 3：`std::string`
写 `string.cpp`：

```cpp
#include <iostream>
#include <string>

int main() {
    std::string s = "hello";
    s += " world";                       // 拼接
    std::cout << s << "\n";
    std::cout << "len=" << s.size() << ", s[1]=" << s[1] << "\n";
    std::cout << "substr(0,5)=" << s.substr(0, 5) << "\n";
    return 0;
}
```

### 任务 4：`std::unordered_map`（哈希表）
写 `map.cpp`：

```cpp
#include <iostream>
#include <unordered_map>
#include <string>

int main() {
    std::unordered_map<std::string, int> m;
    m["one"] = 1;
    m["two"] = 2;
    m["one"] = 111;                      // 覆盖
    std::cout << "m[\"one\"] = " << m["one"] << "\n";

    for (const auto& kv : m)             // 遍历 key-value
        std::cout << kv.first << " -> " << kv.second << "\n";
    return 0;
}
```

**要记住：** `unordered_map` 对应 Python 的 `dict`（哈希）；`map` 是红黑树（有序，`operator<` 排序）。默认用 `unordered_map` 即可。

### 任务 5：栈 vs 堆
写 `stack_heap.cpp`，观察两种内存的分配方式：

```cpp
#include <iostream>
#include <vector>

int main() {
    int stack_int = 5;              // 栈上：作用域结束自动释放
    int stack_arr[1000];            // 栈上定长数组

    int* heap_int = new int(5);     // 堆上：手动 new
    std::vector<int> heap_vec;      // 内部数据在堆上，对象自动管理

    delete heap_int;                // 堆内存必须手动 delete，否则泄漏

    std::cout << "stack int: " << stack_int
              << ", heap vec size: " << heap_vec.size() << "\n";
    return 0;
}
```

**要记住：** 栈（stack）由编译器自动管理、容量小、快；堆（heap）用 `new` 分配、必须 `delete`（或用智能指针，Day 4 讲）。`vector` 的**对象**在栈上，但它的**元素数据**在堆上——它帮你管理堆内存，这就是 RAII 的雏形。

### 任务 6：手写简化版 `vector`（理解扩容）
写 `intvec.cpp`，实现 `push_back` / `operator[]` / `size` / `capacity` / 析构：

```cpp
#include <iostream>
#include <cstddef>

class IntVec {
public:
    void push_back(int v) {
        if (size_ == cap_) {                          // 满了 → 扩容
            std::size_t new_cap = (cap_ == 0) ? 1 : cap_ * 2;   // 倍增
            int* new_data = new int[new_cap];
            for (std::size_t i = 0; i < size_; ++i)
                new_data[i] = data_[i];               // 拷贝旧数据
            delete[] data_;                           // 释放旧内存
            data_ = new_data;
            cap_ = new_cap;
        }
        data_[size_++] = v;
    }

    int& operator[](std::size_t i) { return data_[i]; }
    std::size_t size()     const { return size_; }
    std::size_t capacity() const { return cap_; }

    ~IntVec() { delete[] data_; }                     // 析构释放

private:
    int* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t cap_ = 0;
};

int main() {
    IntVec v;
    for (int i = 1; i <= 10; ++i) {
        v.push_back(i);
        std::cout << "push " << i << ": size=" << v.size()
                  << " capacity=" << v.capacity() << "\n";
    }
    for (std::size_t i = 0; i < v.size(); ++i)
        std::cout << v[i] << " ";
    std::cout << "\n";
    return 0;
}
```

**要记住：** 观察输出里 `capacity` 的倍增（1→2→4→8→16），这就是 `std::vector` 的扩容策略。`delete[]` 配 `new[]`、`delete` 配 `new`，不能混用。

## 练习与验收

1. 写 `count_words.cpp`：用 `unordered_map<string,int>` 统计一段话里每个单词出现次数。
2. 写 `vec_grow.cpp`：连续 `push_back` 100 个元素到 `std::vector<int>`，打印每次 `capacity` 变化，验证「倍增扩容」。
3. 给 `IntVec` 加一个 `at(std::size_t i)` 方法，越界时 `throw std::out_of_range("...")`，并测试它抛异常。
4. 回答（写进笔记）：`IntVec` 的析构函数为什么必须有？如果删掉析构会发生什么（内存泄漏）？

## 自检

- [ ] 能说出 `vector` 的 `size` 与 `capacity` 区别。
- [ ] 能用 `vector`/`array`/`string`/`unordered_map` 写出常用操作。
- [ ] 能解释栈 vs 堆：谁自动释放、谁要手动/自动管理。
- [ ] 能手写 `push_back` 的倍增扩容逻辑。
- [ ] 知道 `new` 配 `delete`、`new[]` 配 `delete[]`。
- [ ] 理解析构函数是「释放资源」的钩子（为 Day 4 RAII 铺路）。

> `vector` 是你在算子开发里最常用的容器（存数据、存中间结果）。今天手写一遍它的扩容，是为了明天理解 RAII 和智能指针——那是避免内存泄漏的关键。
