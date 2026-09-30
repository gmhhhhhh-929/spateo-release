# 国产空间平台原生 IO、延迟加载与恢复说明

更新：2026-09-30。本次消费的是平台计数/解码流程的**原生下游输出**，不是 FASTQ 比对流程，也不把已整理 H5AD 作为原生读取证明。国内产品的组织适用性、分辨率和成本不同，未核实统一价格，不承诺无条件替代其他实验技术。

## 本次可用接口与文件

| 技术/来源 | 必需文件与字段 | 接口 |
|---|---|---|
| SeekSpace / 寻因 SeekGene | `matrix.mtx[.gz]`、`features.tsv[.gz]`、`barcodes.tsv[.gz]`；`cell_locations.tsv[.gz]`，表头 `Cell_Barcode,X,Y`，以 tab 分隔 | `st.io.read_seekspace` |
| BMKMANU S1000 / 百迈客 BSTMatrix 聚合输出 | 同上 MEX 三件套；**复数** `barcodes_pos.tsv[.gz]`，无表头3列 barcode、pos_w、pos_h | `st.io.read_bmkmanu` |
| Salus STS / 赛陆，论文配套工作流 | MEX 三件套；`spatial.txt[.gz]`，无表头、空白分隔 barcode、x、y | `st.io.read_salus` |
| Singleron / 新格元 CeleScope `space` 导出 | `filtered_feature_bc_matrix.h5` 或 `raw_feature_bc_matrix.h5`；H5 属性 `chemistry_description=Spatial3`；`spatial/positions_list.csv` 无表头6列 | `st.io.read_singleron` |
| Stereo-seq / Stereo-seq V2（已有） | 支持的 GEM/GEM2 文本或 bin/cell GEF；具体字段见原生格式说明 | `st.io.read_stereoseq`，统一自动入口不变 |

MEX 为基因×barcode，AnnData 为 barcode×基因。四个新增 reader 都按 barcode 连接坐标，保留矩阵中全部 observation、feature 和原始整数计数，不执行组织内筛选或归一化。`var_names` 使用稳定 feature ID；重复 gene symbol 原样保存在 `var['gene_name']`，不靠添加后缀制造新身份。各坐标表必须有完整、唯一 ID 和有限坐标。

CeleScope CSV 六列为 `barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres`，输出 X 为像素列、Y 为像素行。该 pipeline 也可导出 Parquet，但本次专属 reader 以核实过的 CSV 合约为准。`Spatial3` 是软件导出证据，不能据此承诺所有 PyxiSCOPE 化学版本。普通10x矩阵没有足够证据区分实验品牌。

SeekSpace 保留 chip/image pixels；BMK 保留 BSTMatrix 标准图像显示坐标；Salus 保留工作流原生像素；CeleScope 保留 full-resolution image pixels。**不从营销分辨率猜微米换算、不自动配准图像、不生成 Z。** 三维分析需要额外的切片顺序、间距或来源明确的3D坐标。

BMK 旧版**单数** `barcode_pos.tsv` 的5列是 barcode、子区列/行索引、区内列/行索引，不是 XY。本次明确拒绝直接解释，提示先用 BSTMatrix 导出所需聚合层级的三列位置和匹配矩阵。SeekSpace README 中部分单数/扩展名写法与实际 writer 不同，本实现采用核实的 writer 合约，不把任意 TSV 当 MTX。

DynaSpatial / DynamicST 已调研；公开软件文档尚缺足以落实专属自动识别的空间列定义和厂商标记，暂未添加猜测性 reader。后续取得厂商小型原生示例可扩展。国内成像技术的完整导出格式同样需要逐一确认。

## 自动读取与 AnnData 位置

```python
import spateo as st

result = st.io.read_spatial('/data/sample-output')
print(result.report)             # 所有条目的状态、证据与恢复方案
adata = result.adata             # 仅唯一且完整的输入可用
adata.write_h5ad('/output/sample.h5ad')  # 显式导出
result.write_report('/output/sample_io_report.json')
```

代码没有分值或置信阈值。先在限定深度/文件数内建文件清单，再提出已知格式候选并验证结构；相同矩阵的竞争解释共同决策。唯一可行解释才允许载入。其他有效解释未验证或仍有歧义时保留 `unresolved`，不选第一个文件。无效的国产位置附属文件不会遮蔽同一矩阵的有效 Visium 合约。

完整读取验证全部计数、矩阵轴、ID 与坐标；MTX 原始 COO 记录在合并重复元素**之前**检查，防止负数抵消、分数抵消或整数溢出。浮点原生计数超过精确整数表示能力会被拒绝。可选图像失败不使已经通过的核心计数/坐标丢失。

| AnnData 成分 | 内容 |
|---|---|
| `X` | observation×feature 的 CSR 整数原始计数 |
| `obs` | 以原始 barcode 索引的样本元数据 |
| `var` | 稳定 feature ID、gene_name；可用的 feature_types/genome |
| `obsm['spatial']` | 按 observation ID 对齐的源坐标 |
| `uns['spatial'][library]` | 支持的图像、已提供的有效比例因子、图像路径和资产状态 |
| `uns['spateo_io']` | 实际读取器、证据、参数、完整核心验证、警告、限定范围内文件清单 |
| `uns['native_spatial_export']` | 新增平台的 observation 类型、计数保留策略、导出身份证据的边界 |

输入目录清单表示已检查范围，不等同于所有文件均已消费。边界、mask、转录本表、注册变换及多层图像不因出现在目录中就全部自动导入。图像/scale 是可选资产；IO provenance 是处理记录，两类 `uns` 不互相替代。

四个直接 reader 都接受 `load_images`、`max_memory_bytes`、`return_result`；默认返回唯一成功的 AnnData。多样本或多个 raw/filtered 表示可用 `return_result=True` 检查所有结果。自动入口适用于不知道平台但拥有完整原生目录的情况。

## Lazy 加载的准确含义

```python
result = st.io.read_spatial('/data/collection', lazy=True, load_images=False)
for key, entry in result.items():
    print(key, entry.technology, entry.status, entry.estimated_bytes)

# 从打印的完整 key 中选一个，不默认吞掉其他失败/未加载条目。
key = next(iter(result.keys()))  # 示例；真实多样本任务按所需样本明确选择
adata = result[key].materialize(max_memory_bytes=4 * 1024**3)
print(result.report)            # 其他条目仍然存在
```

`lazy=True` 只执行发现和有限结构探查；第一次访问唯一输入的 `result.adata`，或显式 `entry.materialize()` / `result.load(key)` 时，才完整加载所选 AnnData。报告、迭代、读取 key 不触发加载。这是**延后完整物化**，不是磁盘 backing，也不提供任意格式按行分块取数。`entry.adata` 本身是已载入对象字段，不隐式触发；需要对象时使用 `materialize()`。

`load=False` 仍为纯检查模式，`result.adata` 不隐式读入；`load=False` 与 `lazy=True` 的冲突组合会被拒绝。默认1GiB为估算分配预算而非 OS 强制 RSS 上限；集合已持有的对象计入预算。超预算时保留 `deferred`；增加预算或显式 `retry=True` 后可重试。不自动缩小 bin、不删除 observation/基因来省内存。

成功对象被重用；相同预算下重复访问不反复尝试已延期的加载；失败需显式重试。源文件大小/修改时间改变后必须重新调用 `read_spatial`，避免复用旧解析计划。这不是密码学完整性验证；下载文件仍应核对发布方 checksum。

## 用户问题与恢复

所有条目保留在 `result.report['datasets']`，不能读取的条目不会生成伪 AnnData。`diagnostics[].recovery` 给出动作、说明、相关路径和文档链接；原始 `entry.diagnostics` 与报告中的增强视图有所区别。

| 用户遇到的问题 | 系统处理 | 如何继续 |
|---|---|---|
| 目录错、云文件未下载、卷未挂载 | 路径/权限诊断 | 下载到本地、恢复授权或挂载后重试 |
| 缺矩阵、features/barcodes 或坐标 | `failed`；逐一列出缺失路径 | 从同一 sample/run 的原发布页补齐完整原生导出 |
| ID 不匹配、重复 ID、坏计数/坐标 | `failed`；保留具体原因 | 查明是否混用样本/错误转换，恢复匹配文件；不按行凑合配对 |
| 多种兼容解释/重复文件 | `unresolved`；保留候选 | 选择目标 bundle/矩阵；只有已知平台时才显式指定 technology |
| gzip/H5 损坏或下载截断 | 完整性诊断 | 比较大小/checksum，重新下载对应原文件；改后缀不等于转换 |
| 尚不需加载或内存不足 | `deferred`；lazy/显式按条目加载 | 查看估算值，加载所需样本，或在足够内存环境增加预算 |
| 核心有效但图像/scale 损坏 | 核心保持 ready，给出警告 | 可继续表达/坐标分析；图像分析前补齐匹配资产 |
| 多帧或过大图像 | 保留路径和资产延期状态 | 用适当图像工具按需处理，当前 reader 不展开整幅金字塔 |
| 部分样本失败/扫描范围不完整 | 返回所有已发现结果与 scope 错误 | 已成功条目显式使用；收窄目录或检查后提高扫描范围 |
| 探查后文件变化 | 阻止旧 loader 继续 | 完成下载/导出后重新发现，避免读取变化中的数据 |
| BMK 五列原始芯片索引 | 不伪造 XY | 用 BSTMatrix 聚合输出位置及对应矩阵 |

恢复建议会链接已核实的平台文档，但**不能仅凭本地文件名推断某个私有样本的下载地址**。网页已登录时可用该站下载按钮/官方 CLI；需认证时遵从站点方式。程序不自动安装依赖、不下载/覆盖文件，不自行混配切片。

公开 BMK 示例可用真实 GEO 直链下载；以下仅针对 GSM8816652，不是任意缺失样本的 URL 模板：

```bash
curl --fail --location --retry 3 --continue-at - \
  'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8816nnn/GSM8816652/suppl/GSM8816652_wtirbarcodes_pos.tsv.gz' \
  --output barcodes_pos.tsv.gz
gzip -t barcodes_pos.tsv.gz
```

先检查目标文件并选择新输出目录，避免覆盖其他样本。此示例位置文件 SHA-256 为 `4bbcea19cbb4e19cf56cfcf41f126cc4c1362dda3c2f8143fd1cbc6e81473508`；其余文件 URL/大小/hash 在本次验证记录的下载清单。

## 代码定位与核实来源

2026-09-30 的后续架构重构将每种国产技术拆为独立模块，详见 [独立 reader 与自动调度](domestic_reader_architecture_zh.md)。

- `spateo/io/spatial/_seekspace.py`、`_bmkmanu.py`、`_salus.py`、`_singleron.py`：各平台的发现、字段解析、检查、核心读取与公开接口。
- `spateo/io/spatial/_native_readers.py`：国产平台模块注册；自动层调用模块自身的发现/读取逻辑。
- `spateo/io/spatial/_matrix.py` / `_native_common.py`：MEX/H5 存储验证、共同 ID 连接和计数规则。
- `spateo/io/spatial/auto/_discovery.py` / `_contracts.py` / `_automatic.py`：全平台自动发现、验证调度与唯一解释决策。
- `spateo/io/spatial/_read_engine.py` / `_read_result.py`：两条路线共享的执行、资源预算、lazy 状态和结果集合，不反向依赖自动层。
- `spateo/io/spatial/_recovery.py`：缺失路径、异常类别、恢复动作和来源链接。
- 原 `spatial/_domestic.py` 只保留兼容导出；直接 reader 不再调用自动入口。

主证据为厂商或原论文作者的实际输出代码：

1. [SeekSpaceTools report.py，2434074](https://github.com/seekgene/SeekSpaceTools/blob/2434074a53117ad8e56136566e07d91b09ed1c9c/src/seekspacetools/run/report.py)，以及[厂商教程](https://seeksoul.online/cloudplatform-doc/en/document/General/Notebooks.src/seekspace-spatial-analysis-scanpy.html)。
2. [BMK 输出说明](https://www.biomarker.com.cn/archives/28318)、[官方软件](https://www.bmkgene.com/software-downloads/)中的 `DiffLevelsGeneMatrix.py` / `StdChipModel.py` / `CreateBmkObject.R`；[真实 GSM8816652](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8816652)。
3. [Salus 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12832764/)链接的 [SalusSTS 工作流，6ad825e](https://github.com/xuzaoxu/SalusSTS/tree/6ad825eb88e55c4df87514752b5ec3528bbedad8)，`grid2h5ad.py` 与 `splanegridict.py`。
4. [CeleScope space 指南，851c7b8](https://github.com/singleron-RD/CeleScope/blob/851c7b891a747a31d4271f51571b09d02a64f633/doc/assay/multi_space.md)及同提交的 `celescope/space/analysis.py` / `utils.py`。
5. [DynamicST，d0c651e](https://github.com/DynamicBiosystems/DynamicST/tree/d0c651e653f919fdf9959df089658b99e7f7f2ee)：已调研、未声称完整支持。

## 验证边界

平台格式 fixtures、异常恢复测试与真实样本验证分别统计；测试通过率不等于平台分类准确率、测序准确率或所有厂商版本兼容率。见同目录 `domestic_spatial_io_validation_20260930.json`。本次公开 BMK 全量测试从 MEX/坐标/PNG 开始，独立逐条核对计数、轴 ID、全部坐标、每基因/spot 总量并做 H5AD 往返；其他三个新增平台用核实原生 schema 的小型 fixtures 验证，未冒称已跑真实生产数据。

复现单元/集成测试：`python -m pytest -q tests/io`。Skill 仓库另外提供源代码签名检查、CLI smoke 和真实使用任务测试。

首轮国产平台实现（源码 `d884216`）的完整 IO 回归共 200 项：**199 通过、1 项真实 Visium 环境依赖测试跳过**；其中新增 76 项。公开 BMK 原生输入两次成功运行（一次完整验证、一次可复现脚本验证），均逐条核对 25,239,573 条矩阵记录，总计数 39,824,941。完整读取约4秒、每次读取加全量核对和往返约16秒，依赖本机环境，不是性能承诺。新增代码格式及编译通过；仓库整体 `make check` 遇到已有未修改文件的 isort/Black 格式问题，具体文件及与基线一致的 SHA-256 记录在验证 JSON 中。

```bash
PYTHONPATH=. python scripts/verify_domestic_spatial_reading.py /data/native_GSM8816652 --output /output/new_validation_directory
```
