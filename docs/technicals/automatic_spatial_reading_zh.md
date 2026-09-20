# Spateo 无阈值自动读取：完整修改说明

## 1. 修改目的与兼容性

新增公开入口 `st.io.read_spatial(path)`。正常调用只需要输入路径，不需要设置匹配分数、最低分数或严格模式。内部不再使用旧自动检测器的固定评分、候选评分排序和近似并列分差。

原有 `read_auto_spatial`、`read_spatial_auto`、`detect_spatial_technology`、`detect_spatial_technologies` 及各平台直接 reader 保持原有行为。已有脚本不会因为本次修改突然改变返回类型或阈值含义。新入口是增量功能，不是把旧入口的默认分数改高或把所有候选设成满分。

## 2. 最简单的用法

```python
import spateo as st

result = st.io.read_spatial("/path/to/Visium/outs")
print(result.status)
adata = result.adata
```

对于单个有效数据集，`result.adata` 给出完成核心读取与核验的 AnnData。如果目录中存在多个样本、多个分辨率、失败输入或范围不完整，该属性会明确报错，避免用户无意间只拿到第一个对象。可先查看 `result.report`。

```python
result.write_report("spatial_io_report.json")
```

写报告是显式动作；读取函数本身不写入缓存、不修改原始文件，也不静默修复数据。

## 3. 实际执行流程

1. **统一文件发现。** 解析路径，在限定深度与数量内建立文件清单。识别常见 root、outs、binned_outputs、segmented_outputs 和样本容器。不跟随输入目录内的符号链接。
2. **提出可能的输入布局。** 文件名仅用于发现，不代表已经能读取。不完整的必需文件组合也保留，以便指出缺少哪个文件。
3. **检查平台格式要求。** H5 必需 group/dataset、稀疏矩阵形状元信息、表头、ID／坐标列和适用的身份元数据必须满足相应契约。预检查不宣称全量数据已经通过。
4. **按逻辑输入分组。** 原始与过滤矩阵、不同样本、HD 不同 bin 尺度和细胞分割结果分别保留。同一矩阵的多个坐标文件／多种存储编码不能仅因文件名相似就视为等价。
5. **处理 reader 重叠。** 只使用明确的兼容关系和正面证据。例如完整 MERFISH 文件对可覆盖同一文件对的泛化 seqFISH 识别。Atera 依赖识别字段中明确的 Atera/WTA 元信息与兼容核心结构，不再单凭染色图文件名判断。
6. **读取与全量核验。** 每个确定输入使用严格核心适配器读取。检查全部核心 ID、数值和坐标；不按文件行号强行拼接，不把非法值填零，不默默删除缺失坐标的细胞。
7. **处理可选资产。** 在资源预算内优先载入 hires／lowres，再处理 QC 等其他可支持的小型单帧图片；缺失、损坏、多帧或过大图片有明确状态。可选图片问题不改变已核验的核心数据或平台身份。
8. **保存来源和结果。** 每个成功对象记录 IO provenance，集合报告包含成功、失败、无法唯一识别和延迟读取的所有条目。

完整流程图见[英文技术文档](automatic_spatial_reading.md)中的 Mermaid 图，GitHub 可直接显示。

## 4. 为什么增加严格核心适配器

检查旧实现时发现，部分 table reader 为兼容历史文件会进行数值强制转换、填零、按行号对齐或取 ID 交集。这些操作可能让一个不完整输入产生看似正常的对象。

新入口的 `spateo.io.spatial.auto._contracts.read_core` 对支持的核心格式提供严格读取，不直接调用上述宽松路径。真实使用的适配器名称写入 provenance，不声称调用了未调用的旧 reader。旧 reader 保留，以便用户显式使用更广的历史格式与复杂可选资产。

## 5. 支持的核心数据

| 技术／表示 | 新入口支持内容 |
|---|---|
| Visium | 10x v3 H5／MEX；现代 CSV／Parquet 和旧六列无表头位置表；完整像素 XY |
| Visium HD bin | 从 square 目录确定 bin 尺度；矩阵＋位置表；所有发现的尺度独立保留 |
| Visium HD cellseg | 细胞矩阵＋包含 cell_id/cellid 的有效 Polygon/MultiPolygon GeoJSON；几何质心坐标 |
| Xenium 兼容核心格式 | 细胞矩阵＋唯一 cell ID 的 cells 表及质心 |
| Atera | 兼容细胞核心格式＋明确 Atera/WTA 元数据；保留 preview 标记 |
| MERFISH | 同组 cell_by_gene 与 cell_metadata；完整 ID 关联；支持识别到的 Z 坐标 |
| seqFISH | 同组 counts 与 cell-coordinate 表；通过完整 ID 关系判断矩阵方向 |
| CosMx | 表达与元数据；cell ID＋FOV 组成唯一标识；保留局部坐标和可用全局坐标 |
| Slide-seq | gene-by-bead 矩阵与具名 bead 坐标 |
| STARmap PLUS | raw／processed 表达与对应空间表；明确 ID 关联 |
| Stereo-seq／BGI | GEM 或具有支持表头的 TSV/TXT 分子表；原始整数 XY 位置聚合的总表达矩阵 |

BGI 的输出观察单位标为原生坐标 bin，不推断细胞分割。CosMx 局部 FOV 不自动变成全局已配准坐标。无法从格式确定的坐标单位记录为“来源未声明”，不虚构物理单位。表格中已有且支持的 Z 列保存在三维 `obsm['spatial']`。

本次自动化针对核心矩阵、ID 和空间坐标。旧 H5 格式、H5AD、Seq-Scope 等仍通过已有直接入口读取；不会被新入口用泛化规则强行猜测。

## 6. 返回值与错误状态

返回类型始终是 `SpatialReadResult`，不随单输入／多输入改变为不同 Python 类型。

每个 `result.datasets[key]` 是 `SpatialDataset`，包含：

- `technology`、`source`、`representation`：格式、路径及表示方式。
- `status`：当前状态。
- `adata`：仅 ready 条目具有完整对象。
- `evidence`：文件和身份依据。
- `validation`：结构检查、全量检查及候选诊断。
- `diagnostics`：错误／警告及历史尝试。
- `estimated_bytes`：读取的保守内存估计。

| 条目状态 | 含义 |
|---|---|
| ready | 核心内容已读取且通过完整核验 |
| failed | 格式、数据内容、依赖或读取失败 |
| unresolved | 同一输入存在多个无法排除的有效解释／编码 |
| deferred | 未请求载入或超出资源预算，核心内容尚未完整读取 |

集合 `result.status` 为：全部输入 ready 且发现范围无错误时 `ok`；混合结果为 `partial`；全部仅延迟读取且无发现错误时 `pending`；其他没有可读成功结果的情况为 `failed`。

一个样本失败不会抹掉另一个独立样本的成功结果。反过来，存在失败样本时也不会把整个集合标为 `ok`。可选图片警告不使完整核心对象自动失败。

```python
for key, entry in result.datasets.items():
    print(key, entry.technology, entry.representation, entry.status)
    if entry.status == "ready":
        adata = entry.adata
    else:
        print(entry.diagnostics)
        print(entry.validation)
```

## 7. 用户是否还要设置阈值

**不需要设置识别阈值。** 新入口没有 `min_confidence` 或 `strict` 参数，也没有候选概率或分数排名。

仍然存在工程上的资源边界：默认约 1 GiB 分配预算、10,000 个目录条目、最大四级发现深度和有界图片读取。这些参数用于控制内存与扫描范围，不能改变平台识别证据，也不是“匹配分数阈值”。

```python
result = st.io.read_spatial("/path/to/dataset", load=False)
entry = next(iter(result.datasets.values()))
print(entry.validation)  # content 尚未加载
entry.load()
```

资源不足时不会冒充读取成功，也不会依次尝试其他平台。明确知道大数据能够载入时，可对延迟条目使用 `entry.load(max_memory_bytes=4 * 1024**3)`。内存数值是保守分配估计，不是操作系统强制 RSS 上限；不能保证所有 reader 已具备 backed 模式。

延迟期间核心文件的大小或修改时间发生变化，读取会失败并要求重新发现，而不是沿用过期识别结果。该检查不是密码学哈希。

## 8. 元数据组织

`uns['spatial'][library]`：

- 核心坐标的单位／表示信息与受支持的 H5 来源元数据；
- 已加载图像、可选图像路径及每个资产的状态；
- 原文件提供且有效的比例信息，不生成虚构 scale factor。

`uns['spateo_io']`：

- technology、实际严格适配器名称、source；
- policy_version、resolution_reason；
- evidence、读取表示与参数；
- 完整核心验证结果、警告；
- 明确说明覆盖范围的有界文件 manifest。

新入口不生成 `confidence`。集合级失败输入没有 AnnData，因此其错误保存在 `result.report`，不伪造对象的 uns。图片文件存在或 scale JSON 存在，都不等于已经验证图像配准。

## 9. 具体行为变化示例

| 情况 | 新入口行为 |
|---|---|
| 真实完整 Visium | 自动读取，逐 barcode 对齐坐标，返回 ready |
| 空文件仅取了正确的 H5／CSV 文件名 | 必需结构失败；不生成高分候选 |
| 表达文件有非法字符串或 NaN | 拒绝，不填零 |
| 元数据 ID 与表达 ID 不同，但行数相同 | 拒绝，不按行号匹配 |
| 部分矩阵细胞没有坐标 | 拒绝，不删掉那些细胞继续宣称成功 |
| 存在重复观察 ID | 拒绝，不自行去重 |
| HD 有多个 bin 尺度和 cellseg | 分别返回具名条目 |
| 同一矩阵有两套可用坐标文件 | unresolved，不按文件优先顺序猜测 |
| 可选图片损坏 | 保留有效核心数据并记录图片诊断 |
| 一个独立样本失败 | 集合 partial，成功样本仍可访问 |
| 数据过大或 load=False | deferred／pending，不标成已读完 |

## 10. 文件改动

- `spateo/io/spatial/auto/_result.py`：稳定结果类型、状态计算、报告导出、延迟读取入口。
- `spateo/io/spatial/auto/_discovery.py`：有界文件发现、布局候选和文件组组织。
- `spateo/io/spatial/auto/_contracts.py`：严格 H5／MEX／表格／GeoJSON／GEM 核心读取与验证。
- `spateo/io/spatial/auto/_automatic.py`：无评分决策、结果管理、资源边界、可选图片及 provenance。
- `spateo/io/__init__.py` 与 `spateo/io/spatial/__init__.py`：导出新入口与结果类型。
- `tests/io/test_automatic_reading.py`：行为测试与可选择启用的真实 Visium 完整对照。
- `scripts/verify_automatic_spatial_reading.py`：真实 Visium 全量对照、H5AD 往返与 SHA-256 验证记录。
- 中英文技术文档及技术目录、README：使用方式、兼容性和限制。

没有修改旧自动入口的分数、阈值或排序，也没有为了让新入口“看起来成功”放宽旧测试。

## 11. 验证与实际能力边界

测试覆盖所有上述核心技术的小型合成文件、HD 多尺度、MEX、Parquet、三维坐标、空文件、非法数值、缺失／重复 ID、多对象、延迟读取、图片失败、源文件变化和符号链接。

真实数据使用 10x 公共 `V1_Adult_Mouse_Brain`（Space Ranger 1.1.0）。验证必须比较全部矩阵值、全部 barcode 顺序和逐 barcode XY 坐标，不能用“能够运行”替代内容一致性检查。本次 IO 与预处理回归共 **77 项通过**，仓库 `make check` 通过。真实对象为 **2,702 × 32,285**，包含 **16,031,101** 个非零元素、总 UMI **85,825,294**；全部核心对照与 H5AD 写入／重读检查通过。详见[完整验证记录](automatic_spatial_reading_validation.md)。

公开数据页：https://www.10xgenomics.com/datasets/mouse-brain-section-coronal-1-standard-1-1-0

复现命令：

```bash
python -m pytest -q tests/io
SPATEO_VISIUM_DATA=/path/to/V1_Adult_Mouse_Brain python -m pytest -q tests/io tests/preprocessing
python scripts/verify_automatic_spatial_reading.py /path/to/V1_Adult_Mouse_Brain --output /path/to/report
make check
```

各平台的合成测试不等于对所有厂商历史版本都完成真实数据验证。新入口并未完整载入可选转录本表、边界集合、多帧金字塔、FOV 复合图和变换矩阵等丰富资产；相关需求继续使用平台专用 reader。表格核心矩阵的 dtype／附加元数据可能不同于旧 reader，不承诺旧 AnnData 的字节级一致性；本次承诺的是受支持核心内容的完整性与显式诊断。

## 12. 如何迁移已有脚本

旧脚本可以继续运行。如果希望使用新行为，将单输入场景改为：

```python
# 旧入口仍有效
# adata = st.io.read_auto_spatial(path)

result = st.io.read_spatial(path)
if result.status != "ok":
    result.write_report("spatial_io_report.json")
    raise RuntimeError("Spatial input is incomplete; inspect the report.")
adata = result.adata
```

对多输入应显式遍历 `result.datasets`。不要在迁移时随意取 `next(...)` 的第一个 ready 对象当成整个数据集；也不应通过调高／调低分数掩盖文件或身份冲突。
