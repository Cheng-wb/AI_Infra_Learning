# Month 1：C++ 与 CUDA 编程基础

> **主题：** 从 Python 切到 C++，建立 CUDA 编程模型认知，能写、编译、验证第一个 CUDA Kernel。
> **前置：** 会 Python（有参考项目的调度算法基础即可），在 AutoDL 租了 RTX 4090（sm_89）。
> **月度目标：** 理解「线程 / Block / Grid」三级并行和「寄存器 / 共享内存 / 全局内存」三级内存；能独立写一个 element-wise 算子并用 CPU 参考实现对拍验证。

## 月度里程碑

- ✅ 用 C++ 独立实现一个带单元测试的小模块（如动态数组 / 矩阵）。
- ✅ 写出第一个 CUDA kernel（向量加法），理解 `<<<grid, block>>>` 启动语法。
- ✅ 能解释 SIMT、warp、SM 三个概念，画出内存层次图。
- ✅ 写一个 element-wise 算子 + CPU 对拍 + 计时，形成可复用框架（P1 启动）。
- ✅ 用 CMake / nvcc 组织编译，理解 `.cu` 与 `.cpp` 的编译流程。

## 环境准备（第一天先做）

1. 在 AutoDL 租一台 4090，选带 CUDA + PyTorch 的镜像；`nvcc --version`、`nvidia-smi` 确认是 4090（sm_89）。
2. 熟悉 `nvcc file.cu -o out && ./out` 的编译运行流程（也可在 JupyterLab 里写）。
3. 建 GitHub 仓库 `AI-Infra-Operator-Learning`，用参考项目同样的「月/周/天」目录结构提交笔记与代码。

---

## Week 1：C++ 基础（为 CUDA 补齐语言能力）

**周目标：** 掌握指针、引用、模板、STL 容器、RAII、编译链接，能看懂并写出 C++ 代码；把 Python 思维（list 动态、引用语义）纠正为 C++ 思维（内存、值语义、手动管理）。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 从 Python 到 C++ | 变量/类型/`const`、值语义 vs 引用语义、`for`/`auto`、`std::cout`、命名空间；理解「变量是一块内存」 | 用 C++ 重写一个你熟的小逻辑（如斐波那契、阶乘），加 `-Wall -O2` 编译 | 能编译运行，能解释 `int`/`float`/`double` 字节与溢出 |
| Day 2 | 指针与引用 | 指针、取址 `&`、解引用 `*`、指针运算、`nullptr`、引用 vs 指针、值传递/指针传递/引用传递 | 手写 `swap(int*, int*)` 与 `swap(int&, int&)`，验证地址变化 | 能画出一段数组在内存中的布局，解释 `a[i] == *(a+i)` |
| Day 3 | 数组与 STL | `std::vector`、`std::array`、`std::string`、`std::map`/`unordered_map`、迭代器、范围 for；栈 vs 堆 | 实现一个动态数组类（`push_back`/`operator[]`），与 `std::vector` 对比行为 | 能说清 `vector` 扩容机制、栈/堆对象的生命周期 |
| Day 4 | 类、结构体、RAII | `struct`/`class`、构造/析构、拷贝构造、移动语义（`std::move`、右值引用）、`unique_ptr`/`shared_ptr` | 写一个 RAII 文件/资源句柄类，构造打开析构释放 | 能解释为什么析构要 `virtual`、`unique_ptr` 如何避免泄漏 |
| Day 5 | 模板与函数重载 | 函数模板、类模板、特化、重载决议、`constexpr`、`static` | 写一个泛型 `max`/`sum` 模板 + 对 `int`/`float` 特化 | 能解释模板在编译期展开，特化与重载的区别 |
| Day 6 | 编译与链接 | 头文件/源文件分离、`#include`、声明 vs 定义、`extern`、静态库/动态库、`g++` 多文件编译、CMake 入门 | 把前面的类拆成 `.h`/`.cpp` 多文件，写 `CMakeLists.txt` 构建 | 能解释「未定义引用」「重复定义」两个经典报错 |
| Day 7 | 复习 + 综合 | 用 C++ 实现一个「矩阵」类（元素存储 + `operator()` + 加法/乘法），带 `assert` 单测 | 综合练习：矩阵类 + 简单单测框架（`assert` 或自己写 `EXPECT_EQ`） | 交出「可复用 C++ 模块 + 单测」作为 Week 1 产出 |

**Week 1 自检：** 能不查资料写出指针解引用、`vector` 遍历、一个带析构的类、一个函数模板、一个多文件 CMake 项目。

---

## Week 2：GPU 架构与 CUDA 编程模型

**周目标：** 建立「GPU 为什么适合并行」的直觉，理解线程层次（Thread/Block/Grid）、SIMT、warp、SM，写出第一个 kernel。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | GPU vs CPU | 吞吐 vs 延迟、大量轻量线程、SIMT（Single Instruction Multiple Threads）、数据并行思想 | 读 PMPP 第 1–2 章；画一张 CPU/GPU 对比表 | 能用「吞吐 vs 延迟」解释 GPU 适合什么任务 |
| Day 2 | CUDA 编程模型 | `__global__`/`__device__`/`__host__`、`<<<grid, block>>>`、`threadIdx`/`blockIdx`/`blockDim`/`gridDim`、`cudaMalloc`/`cudaMemcpy`/`cudaFree` | 写「打印线程索引」kernel：`printf` 或存数组回传，验证索引公式 `idx = blockIdx.x*blockDim.x + threadIdx.x` | 能说出 1D/2D/3D 索引计算公式 |
| Day 3 | 第一个 kernel：向量加法 | kernel 函数、数据拷贝（Host→Device→Host）、`cudaDeviceSynchronize`、错误检查宏 `cudaGetErrorString` | 写 `vector_add`，n=1M，与 CPU 循环结果对拍 | 跑通 + 结果一致，能解释为什么需要同步 |
| Day 4 | warp 与线程束 | warp = 32 线程，warp 内锁步执行，分支发散（divergence） | 写一个「if (tid%2==0)」的 kernel，用 `__syncthreads` 观察，理解发散代价 | 能解释「同一个 warp 内不同分支会串行执行」 |
| Day 5 | SM 与线程调度 | Streaming Multiprocessor、一个 SM 上多个 warp、延迟隐藏（occupancy）、Block 如何分配到 SM | 读 PMPP 硬件章节；用不同 grid/block 尺寸跑同一 kernel 计时对比 | 能解释「更多活跃 warp → 更好隐藏访存延迟」 |
| Day 6 | 内存层次概览 | 寄存器（每线程私有）、共享内存（block 内共享）、全局内存（全局）、常量/纹理（只读） | 写一个 kernel 分别用全局 vs 寄存器存中间结果，体会速度差异 | 能画出内存层次金字塔，标出访问延迟量级 |
| Day 7 | 复习 + 综合 | 综合：向量加法 + 缩放 `y = a*x + b`（saxpy）+ 点乘的 CPU 对拍框架 | 搭一个「CPU 参考 vs GPU 结果对拍」的通用小函数，作为后续模板 | 交出「saxpy + 对拍 + 计时」模板 |

**Week 2 自检：** 能手写出 `__global__ void add(...)` 完整骨架；能解释 Thread/Block/Grid、warp、SM 关系；能算任意 1D/2D 索引。

---

## Week 3：CUDA 内存与访存合并

**周目标：** 理解全局内存「合并访问（coalesced）」这一核心优化，认识共享内存、寄存器、常量内存的适用场景。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 全局内存与合并访问 | 一次访存按 128 字节事务读，连续线程访问连续地址才合并；stride 访问的惩罚 | 写 `a[i] = b[i]`（连续）vs `a[i] = b[i*stride]`（跨步），stride=2/4/8 计时对比 | 画出两种访问模式的内存事务图，解释速度差 |
| Day 2 | 矩阵按行 vs 按列访问 | 行主序存储 `A[i][j]=A[i*N+j]`，列访问不合并 | 写「矩阵求和」两种遍历（行主序/列主序），对比时间 | 能解释为什么列主序遍历慢数倍 |
| Day 3 | 共享内存（shared memory） | `__shared__`、`__syncthreads()`、生命周期、bank 概念初识 | 写一个用共享内存做 block 内归约的小例子 | 能解释 `__syncthreads` 的必要性与死锁风险 |
| Day 4 | Bank Conflict（初识） | 32 个 bank，同 bank 同地址广播、不同地址串行化；padding 消除 | 写「转置」naive 版（共享内存 + 不同索引顺序），观察 conflict | 能解释「bank conflict 让访存翻倍」 |
| Day 5 | 常量内存与只读 | `__constant__`、广播、`__ldg`/`const __restrict__`、纹理内存适用场景 | 写一个查表 kernel，对比常量内存 vs 全局内存 | 能说清常量内存适合「所有线程读同一地址」 |
| Day 6 | 寄存器与本地内存 | 寄存器溢出（spill）到 local memory、`--ptxas-options=-v` 看寄存器用量 | 写一个大量临时变量的 kernel，编译时打印寄存器数，观察 spill | 能解释寄存器耗尽 → 溢出 → 变慢 |
| Day 7 | 复习 + 综合 | 综合：转置 naive + shared-memory 版 + bank conflict 消除，计时对比 | 完成「转置三版本性能对比」并记录数字 | 交出 Week 3 的访存优化对比笔记 |

**Week 3 自检：** 能画出合并 vs 非合并访问的示意图；能写出用 `__shared__` + `__syncthreads` 的 kernel；能解释 bank conflict。

---

## Week 4：工具链、计时与算子框架

**周目标：** 掌握错误检查、计时（CPU timer / cudaEvent）、编译选项，搭建可复用的「CPU 对拍 + 计时」算子框架，完成第一个里程碑。

| 天 | 主题 | 学习内容（要点） | 动手练习 | 完成标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 错误检查宏 | `cudaGetLastError`、`cudaGetErrorString`、把每次 API 调用包进 `CHECK` 宏 | 写 `CHECK(cudaMalloc(...))` 宏，故意制造错误看输出 | 所有后续代码都套上错误检查 |
| Day 2 | 计时（event vs 墙钟） | `cudaEvent` 计时、`clock64`、CPU `std::chrono`、warmup、多次取中位数 | 写一个计时封装：warmup + 多次运行 + 报告均值/带宽 | 能解释为什么要 warmup、为什么取多次 |
| Day 3 | 编译选项与 ptxas | `-arch=sm_89`（4090）、`-O3`、`--ptxas-options=-v`、`-use_fast_math`、`-G`（debug） | 对比 `-O3` vs `-G`、`-use_fast_math` 对结果和速度的影响 | 能说清 `-arch` 选错会怎样、fast_math 的精度代价 |
| Day 4 | CMake + CUDA | `find_package(CUDA)` 或 `enable_language(CUDA)`、`.cu` 编译、`set_target_properties` 设 arch | 用 CMake 组织「CPU 参考 + CUDA kernel + 测试」三文件项目 | 交出一个可 `cmake && make` 的项目骨架 |
| Day 5 | element-wise 算子 | 统一接口：`template<T> void launch_xxx(T* out, const T* in, ...)`，支持 relu/sigmoid/add/mul | 实现 relu / add / mul 三个 element-wise 算子 + 对拍 | 三个算子通过对拍，形成算子库雏形 |
| Day 6 | 带宽与 Roofline（初识） | 算术强度、访存受限 vs 计算受限、Roofline 图 | 测 element-wise 的实际带宽，与 4090 峰值带宽（~1008 GB/s）对比 | 能算出自己的 kernel 达到峰值带宽的百分比 |
| Day 7 | 月度复习 + 里程碑 | 复盘 M1 全部知识点，整理 portfolio：基础算子库 + 对拍框架 + 性能数字 | 完成「月度自检清单」，写 M1 总结笔记 | 交出 M1 产出：可复用框架 + element-wise 算子 |

**Week 4 自检 / M1 月度自检清单：**
- [ ] 能不看模板写出 vector_add kernel 完整代码（含内存分配/拷贝/释放/错误检查）。
- [ ] 能解释 Thread/Block/Grid、warp、SM、SIMT。
- [ ] 能画出内存层次图并标出全局/共享/寄存器访问量级。
- [ ] 能解释合并访问与 bank conflict，并给出消除方法。
- [ ] 拥有一个「CPU 对拍 + 计时」的可复用框架。
- [ ] 至少 3 个 element-wise 算子通过正确性对拍，并记录了带宽数据。

---

## 本月背书行动（在职）

- 建 GitHub 公开仓库，采用「月/周/天」目录结构，规范 commit 信息（`feat:`/`bench:`/`docs:`）。
- 把 Week 4 的 element-wise 算子库写成带 README + benchmark 数字的 demo，作为首个可复现作品。
- 报名 1 门 NVIDIA DLI 课程（CUDA C/C++ 或加速计算），花 1–2 天拿证书。
- 产出口径：**首个公开仓库 + 首个可复现 demo + 1 张 DLI 证书**。

## Month 1 参考资源

- PMPP（Hwu）第 1–5 章（编程模型、内存、性能）
- CUDA C++ Programming Guide：第 1–3 章（模型、内存、接口）
- CUDA C++ Best Practices Guide：第 1–3 章（合并访问、occupancy）
- NVIDIA 博客《An Even Easier Introduction to CUDA》

> 完成 Month 1 后，你已经具备「写 CUDA kernel 的最基本能力」。Month 2 将把这些能力用于真正有难度的并行原语（归约/扫描/转置），并系统掌握共享内存优化。
