# Day 4：类、结构体与 RAII

> **今日目标：** 掌握类/结构体、构造/析构、拷贝与移动语义，理解 RAII 与智能指针——这是 C++ 内存安全的基石，也是 CUDA 里「封装资源 + 自动释放」的思维来源。

## 任务清单

### 任务 1：`struct` 与 `class`
写 `struct_vs_class.cpp`：

```cpp
#include <iostream>
#include <string>

struct Point {
    int x;      // struct 默认成员 public
    int y;
};

class Person {
    std::string name_;   // class 默认成员 private
    int age_;
public:
    Person(std::string name, int age) : name_(name), age_(age) {}
    void introduce() const {
        std::cout << name_ << ", " << age_ << "\n";
    }
};

int main() {
    Point p{3, 4};                       // 聚合初始化
    std::cout << "p = (" << p.x << "," << p.y << ")\n";

    Person person("Alice", 30);
    person.introduce();
    return 0;
}
```

**要记住：** `struct` 和 `class` 在 C++ 里**几乎一样**，唯一区别是默认访问权限（`struct` 默认 `public`，`class` 默认 `private`）。`Point p{3,4}` 是列表初始化。

### 任务 2：构造与析构
写 `ctor_dtor.cpp`，观察构造/析构的调用时机：

```cpp
#include <iostream>

class Demo {
public:
    Demo()  { std::cout << "constructor\n"; }
    ~Demo() { std::cout << "destructor\n"; }
};

void f() {
    Demo d;            // 进入 f：构造
}                       // 离开 f：自动析构

int main() {
    std::cout << "before f\n";
    f();
    std::cout << "after f\n";
    return 0;
}
```

**要记住：** 构造函数负责初始化，析构函数负责清理。栈上对象的析构在作用域结束**自动**发生——这正是 RAII 的基础。

### 任务 3：RAII（资源获取即初始化）
写 `file_raii.cpp`，用类封装文件句柄，保证自动关闭：

```cpp
#include <cstdio>
#include <iostream>

class File {
public:
    explicit File(const char* path) {
        f_ = std::fopen(path, "w");
        if (!f_) throw std::runtime_error("open failed");
    }
    ~File() {                         // 析构自动 fclose
        if (f_) std::fclose(f_);
        std::cout << "file closed\n";
    }
    void write(const char* s) { if (f_) std::fputs(s, f_); }

    File(const File&) = delete;                    // 禁止拷贝（避免双重关闭）
    File& operator=(const File&) = delete;

private:
    std::FILE* f_ = nullptr;
};

int main() {
    File f("test.txt");
    f.write("hello RAII\n");
    return 0;                       // 离开作用域，析构自动关闭文件
}
```

**要记住：** RAII = 把资源（文件/内存/句柄）的获取放在构造、释放放在析构，让「生命周期」自动管理资源。这是 C++ 内存安全的核心，`std::vector`、`std::string`、`std::unique_ptr` 都是 RAII。

### 任务 4：拷贝语义与「三/五法则」
写 `copy_semantics.cpp`：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = a;         // 拷贝构造：深拷贝
    b[0] = 999;
    std::cout << "a[0]=" << a[0] << ", b[0]=" << b[0] << "\n";  // 1, 999

    std::vector<int> c;
    c = a;                          // 拷贝赋值
    c[0] = 777;
    std::cout << "a[0]=" << a[0] << ", c[0]=" << c[0] << "\n";  // 1, 777
    return 0;
}
```

**要记住：** 拷贝构造 `T b = a` 和拷贝赋值 `c = a` 都会复制数据。对于自己管理裸指针的类，不写拷贝构造/赋值会导致「浅拷贝 → 双重释放」的经典 bug，所以要么禁用拷贝（上面 `File` 那样 `= delete`），要么正确实现（Day 6 会涉及）。

### 任务 5：移动语义与右值引用
写 `move_semantics.cpp`：

```cpp
#include <iostream>
#include <vector>
#include <utility>

int main() {
    std::vector<int> src = {1, 2, 3, 4, 5};
    std::vector<int> dst = std::move(src);   // 移动：把 src 的内部数据"搬"给 dst

    std::cout << "dst size=" << dst.size() << "\n";
    std::cout << "src size=" << src.size() << "\n";   // 通常变空（未规定，但一般被掏空）
    return 0;
}
```

**要记住：** 移动（`std::move`）把资源「转移」而非「拷贝」，省一次深拷贝，对 `vector`/`string` 等大对象很有用。移动后原对象处于「已移动」状态，不应再依赖它的内容。

### 任务 6：智能指针（告别手动 delete）
写 `smart_ptr.cpp`：

```cpp
#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<int> p = std::make_unique<int>(42);   // 独占所有权
    std::cout << "*p = " << *p << "\n";
    // 离开作用域自动 delete，无需手动 delete

    std::shared_ptr<int> q = std::make_shared<int>(7);    // 共享所有权，引用计数
    std::shared_ptr<int> q2 = q;                          // q2 和 q 共享
    std::cout << "*q = " << *q << ", use_count=" << q.use_count() << "\n"; // 2

    return 0;
}
```

**要记住：** `unique_ptr` 独占、不可拷贝（可移动）；`shared_ptr` 共享、引用计数为 0 才释放。优先用 `unique_ptr`，需要共享时才用 `shared_ptr`。在现代 C++ 里，几乎不该出现裸 `new`/`delete`。

## 练习与验收

1. 写 `counter.cpp`：类 `Counter` 有成员 `int count_`，构造函数初始化为 0，方法 `increment()`，析构打印最终计数。在主函数里创建它并调用，观察析构时机。
2. 写 `vec_holder.cpp`：类 `VecHolder` 内部持有 `std::vector<int>`，构造函数接收一个 `vector`，用**移动语义**把它搬进来，验证外部原 `vector` 被掏空。
3. 回答（写进笔记）：为什么 `File` 类要 `= delete` 拷贝构造？如果允许拷贝会发生什么（两个对象指向同一个 `FILE*`，第一个析构 `fclose` 后第二个再用 → 未定义行为）。
4. 把 Day 3 的 `IntVec` 改成用 `std::unique_ptr<int[]>` 管理内部数组，删掉手动 `delete[]`，验证还能正常工作。

## 自检

- [ ] 能说出 `struct` 和 `class` 的唯一区别。
- [ ] 能解释构造/析构的调用时机。
- [ ] 能用 RAII 封装一个资源（文件/内存）并自动释放。
- [ ] 能解释拷贝构造 vs 移动构造（深拷贝 vs 转移）。
- [ ] 会用 `unique_ptr` / `shared_ptr`，知道何时用哪个。
- [ ] 理解「自己管理裸指针的类必须正确处理拷贝，否则双重释放」。

> RAII + 智能指针让你「几乎不用手动管理内存」。这对写 CUDA 尤其重要：后面你要手动 `cudaMalloc`/`cudaFree`，会自己封装 RAII 的显存管理类——今天的思维就是地基。
