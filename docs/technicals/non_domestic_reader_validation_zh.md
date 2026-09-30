# 既有空间平台自动与单独读取准确性核验

日期：2026-09-30。基线：`82002ba0910a0fa29874f1da92e49a76bc9186a7`。

本轮排除上一轮新增的 SeekSpace、BMKMANU、Salus STS、CeleScope space 四种平台，逐项核验其余已有空间 IO。原有 Stereo-seq V1/V2 也纳入复测。验证从原生 H5、MEX、CSV、GEM、GEF 等文件开始，不用预先整理好的 H5AD 充当输入；H5AD 仅用于输出往返检查。

## 结果与范围

完成一轮最终全量 IO 回归：**405 项通过、1 项跳过**，其中 **168 项是本轮新增**。开发中针对缺陷进行了多次定向修复和复测；这些重复运行不算更多独立样本。跳过项是需要环境变量指定真实 Visium 路径的既有可选测试，真实 Visium 已在本轮另行完成全量原生数据审计。

| 技术 / 表示 | 本轮新增用例 | 单独读取 | 自动读取 | 本轮数据证据 |
|---|---:|---|---|---|
| Visium | 18 | 通过 | 通过 | 原生格式测试 + 完整真实小鼠脑数据 |
| Visium HD bin | 19 | 通过 | 通过 | H5、MEX、Parquet 原生格式测试 |
| Visium HD cell segmentation | 10 | 通过 | 通过 | H5 + GeoJSON 原生格式测试 |
| Xenium | 17 | 通过 | 通过 | H5 + CSV.gz / Parquet 原生格式测试 |
| Atera in Situ | 9 | 通过 | 通过 | 当前兼容格式测试，保留 preview 状态 |
| MERFISH | 14 | 通过 | 通过 | 独立构造的原生表格与图像 |
| seqFISH | 15 | 通过 | 通过 | 原生表格、矩阵转置、XY/XYZ、图像 |
| CosMx / NanoString | 16 | 通过 | 通过 | FOV + cell ID、局部/全局坐标、图像 |
| STARmap PLUS | 16 | 通过 | 通过 | raw / processed 表达、TYPE 行、XYZ |
| Slide-seq / Slide-seqV2 格式 | 15 | 通过 | 通过 | 原生格式测试 + 完整真实 Puck 数据 |
| Stereo-seq V1/V2 | 11 | 通过 | 通过 | GEM/GEF、bin/cell、exon、旧接口语义 |
| Seq-Scope | 8 | 通过 | **当前不支持** | MEX + 五列坐标、按 barcode / bin 输出 |
| **合计** | **168** | | | |

测试通过率是这些明确用例的结果，不是总体平台识别准确率、测序准确率或所有厂商版本的兼容率。GEM 格式本身不一定能证明 V1/V2 化学版本；测试核验显式 chemistry 来源信息，不把 GEF/GEM schema 版本当成化学版本。

## 如何判断读取正确

测试文件事先独立写定计数、ID、坐标和图像像素；两条读取路径分别与这些预期比较，而不是只比较它们是否相同。主要检查：

- 大于 `2**24`、`2**31`，以及适用表格中的 `2**53` 的整数计数精确保留，零表达基因保留。
- 元数据行顺序故意打乱，仍按 ID 对齐；前导零 ID、重复 gene symbol 与稳定 gene ID 分开验证。
- 小数 XY、XYZ 精度保留，必需坐标完整且有限；两条路径各自进行 H5AD 往返。
- 缺必需文件、缺失/重复/错误 ID、NaN 坐标、非数值/负数/小数原始计数等输入被拒绝，不补零、不默默取交集。
- 可选图像坏损保留正确核心数据并警告；能加载的图像与预期像素比较。
- 对独立实现的旧 reader，禁用自动入口后仍可执行。`read_stereoseq` 当前仍是自动入口的已知平台包装，**不能将这条路径的相等当成两套独立实现的证明**；另测独立的旧 `read_bgi`。

对应测试：`tests/io/test_tenx_reader_accuracy.py`、`test_imaging_reader_accuracy.py`、`test_other_reader_accuracy.py`。既有国内平台、lazy 和错误处理测试也包含在最终全量回归中。

## 本轮发现并修复的问题

| 问题 | 修复后的行为 |
|---|---|
| 10x H5 的 int32 经过 float32 转换、MEX 默认 float32，可能丢失整数精度 | 保留 int64 原始计数；检查稀疏索引、数值、ID 和聚合溢出 |
| Slide-seq 独立读取 int32、无效值补零、重复坐标取首行、缺坐标取交集 | 两条路径共用严格的流式矩阵解析；拒绝不完整核心输入，支持 gzip 默认发现 |
| MERFISH int32 溢出；seqFISH / STARmap float32 精度损失 | 保留安全数值精度，保留 XYZ；STARmap 默认 `dtype=None`，拒绝显式有损转换 |
| seqFISH 按行序匹配或生成 ID，多个平台默默删除未匹配细胞 | 必需 ID 唯一且完整，明确按 ID 连接 |
| Xenium / Atera 坐标强转 float32；Visium/HD 可保留缺失坐标 | 保留 float64 坐标，拒绝缺失或非有限值 |
| HD 的有效 hires 被缺失 lowres 牵连；坏图像阻断核心；坏 H5 被替换或误报 | 独立处理可选资产；只有 H5 不存在才尝试 MEX，已有损坏 H5 明确报错 |
| CosMx 可选全局坐标损坏被静默忽略 | 保留局部坐标，警告并记录 `optional_global_coordinates='invalid'` |
| Seq-Scope 缺坐标删除 barcode、数值无符号转换可能丢失信息 | 严格检查原始计数和五列坐标，不静默删除矩阵观测 |

原始计数格式现在拒绝负数或小数计数；处理后的 STARmap 表达仍允许有限的负值和小数。无效数据需要修正源文件，不通过强制转换伪装成功。

## 完整真实数据核验

| 数据集 | 完整矩阵 | 独立源数据核对 | 结果 |
|---|---|---|---|
| 10x V1 Adult Mouse Brain | 2,702 spots × 32,285 features | 原始 H5 全部 16,031,101 非零记录；总 UMI 85,825,294；全部 ID、XY、hires/lowres、scalefactors | 两条路线一致，两个 H5AD 往返通过 |
| Slide-seq Puck_180413_7 | 38,666 beads × 19,869 genes | 原始 CSV 全部 768,254,754 元素（含零）；4,442,379 非零；总 UMI 5,152,436；全部 ID、XY | 两条路线一致，两个 H5AD 往返通过 |

这是 **2 套真实数据 × 2 种读取方式 = 4 次最终读取**，对应 4 次输出往返。原始计数审计使用独立的 h5py/scipy 或 csv/numpy 解析，不复用 Spateo 解析函数作为答案。Slide-seq 从用户下载的原始 tar 中原样取出完整矩阵、完整坐标和主 BeadImage，不包含其他可选通道/worker 图像。所有输入记录 SHA-256。

数据来源：[10x 小鼠脑](https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0)、[SCP354 Slide-seq](https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study#study-download)。其他平台本轮没有新增真实生产样本全量审计；MOSTA/ARTISTA 的早先结果不计入本轮样本数。

## 不能误判为读取错误的差异

1. **特征筛选**：部分 10x 独立接口默认仅保留 Gene Expression；自动读取保留控制探针等全部来源特征。另有用例按各自明确的预期特征集合验证，不强行要求所有默认对象形状相同。
2. **坐标定义**：原生 Stereo-seq 接口给出 bin 起点；旧 `read_bgi(..., binsize=50, add_props=True)` 给出 bin 中心，相差 25，且 ID 命名不同。按同一 bin 和 gene 对齐后计数一致。STARmap 默认保留源坐标；显式 `reorient_xy=True` 是另外的坐标变换。
3. **FOV**：CosMx 的 `spatial` 保留 FOV 局部像素；`spatial_fov` 仅来自实际全局列。额外 FOV 位置信息不会被隐式重复平移。缺少全局坐标不等于核心局部坐标缺失。
4. **图像预算**：Slide-seq BeadImage 解码为 6,030 × 6,030，共 36,360,900 字节，超过自动读取的 32 MiB 可选图像预算。自动保留原始路径和 `deferred_resource`；独立读取载入的全部像素与原图相同。这里的延期属于图像资产，核心 AnnData 已就绪。
5. **附加元数据**：独立接口可能组织 FOV 图像、形态、transcript 或 boundary 信息；自动读取使用统一且有预算的资产组织。验证核心数值和对应像素，不宣称整个 `uns` 结构完全相同。
6. **支持边界**：Seq-Scope 尚无自动识别合约；Open-ST 没有本轮可测的专用原生自动 reader。已有 H5AD 使用 H5AD 接口，这不等于支持其原始输出自动识别。旧独立接口也没有自动结果对象的 lazy/resume API。

## 复现与质量记录

在源码根目录运行，并明确指定导入路径，避免已有 editable 安装指向其他 checkout：

```bash
PYTHONPATH=. python -m pytest -q tests/io/test_tenx_reader_accuracy.py tests/io/test_imaging_reader_accuracy.py tests/io/test_other_reader_accuracy.py
PYTHONPATH=. python -m pytest -q tests/io
```

本轮 18 个修改/新增 Python 文件的编译、isort、Black、空白检查通过。仓库整体 `make check` 仍因 3 个未修改文件的既有 isort 问题失败：`test_slideseq_stream.py`、`test_spatial_auto.py`、`test_tissue_loss_v3.py`。没有将该检查报告为通过。

机器可读结果、最终 IO 源码签名、输入和输出签名、真实数据核验及限制见 [验证记录](non_domestic_reader_validation_20260930.json)。可复跑真实审计脚本和完整本机日志保存在本轮 `spateo_other_readers_20260930/real_validation` 交付目录。开发期关于旧 bin 中心、重复 gene symbol 和图像预算的测试预期修正已记录；它们不被计为额外真实样本，也不掩盖失败记录。
