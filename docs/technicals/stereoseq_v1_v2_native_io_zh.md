# Stereo-seq 原版与 V2：原生矩阵读取说明

本次扩展针对 **SAW 下游原生矩阵**，不是 FASTQ 比对流程。Stereo-seq V2 使用随机引物在 FFPE 中捕获总 RNA；这不等于 GEM/GEF 的文件版本号为 2。原版和 V2 可共享矩阵格式，读取器必须按字段与索引验证，不能按文件名或格式版本猜实验化学版本。

## 依据

- 原文：[Zhao et al., Cell 2025](https://doi.org/10.1016/j.cell.2025.08.008)。
- [作者代码](https://github.com/YoungLi88/Stereo-seq-V2)：`Fig2_sFig2/Fig2_cell_segmentation.ipynb` 实际调用 Spateo `read_bgi_agg` 和 `read_bgi`，读取 `A02384C3/A02384C4.lasso.test_0.gem2.gz`。因此原来的 Spateo 并非完全不能读 V2；主要缺口是自动入口、GEF、字段保留和严格校验。
- [SAW 矩阵说明](https://www.stomics.tech/service/saw_8_1/docs/analysis/outputs/matrices.html)。
- [官方 GEF 文件规范](https://en.stomics.tech/col989/list.html)、[geftools GEM 导出实现](https://github.com/STOmics/geftools/blob/main/geftogem.cpp)。
- [V2 项目 PRJCA025873](https://ngdc.cncb.ac.cn/bioproject/browse/PRJCA025873)：CRA016462（FFPE 脑）、CRA018257（邻接 fresh-frozen 脑）、CRA018250（感染小鼠肺）。人源数据 HRA007387 为受控访问。实验样本条目或 FASTQ 可用不代表对应 GEM/GEF 已能直接下载。

## 推荐用法

```python
import spateo as st

# 路径是唯一必需参数。无平台分值、置信阈值或最高分选择。
result = st.io.read_spatial("sample.gem.gz")
print(result.report)
adata = result.adata

# 已知来源为 V2 时可补充来源标签。它不改变矩阵读取或筛选。
result = st.io.read_spatial("sample.tissue.gef", stereoseq_chemistry="V2")

# 可选科学分辨率设置：GEM 全记录聚合到 bin50；GEF 选择已存 bin50。
# 分箱大小不是自动识别阈值，也不会在内存不足时被静默改变。
adata = st.io.read_stereoseq("sample.gem.gz", bin_size=50,
                           chemistry="V2", max_memory_bytes=4 * 1024**3)
```

`read_stereoseq` 返回一份 AnnData；`read_spatial` 返回含所有条目及状态的 `SpatialReadResult`。读取多切片文件夹时检查 `result.datasets`，只有唯一完整 ready 输入才可取 `result.adata`。

## 格式与输出

| 输入 | 关键字段／路径 | 输出 |
|---|---|---|
| 旧 GEM、GEMv0.1 | geneID, x, y, MIDCount/MIDCounts/UMICount/UMICounts/count/total | X：int64 CSR；obsm['spatial']：存储坐标 |
| GEMv0.2 | 同上，加 geneName | var_names 使用稳定 geneID；var['gene_name'] 保留名称，允许不同 ID 同名 |
| ExonCount | 每条记录的 exon MID 数 | layers['exon']；不自行改称 spliced |
| EXONIC / INTRONIC | 作者或上游提供的计数列 | layers['spliced'] / layers['unspliced']；不由总数自行推算 |
| CellBin GEM | CellID；可含 #BinType=CellBin | 按 CellID 聚合；坐标是每个细胞不同已捕获 DNB 的均值，不是分割多边形质心 |
| square-bin GEF | geneExp/binN/gene、expression；可选 exon | 从 gene offset/count 重建稀疏矩阵；读取存储 x/y |
| cell-bin GEF | cellBin/cell、gene、cellExp；可选 cellExpExon | 按 cell offset/geneCount 重建；使用文件内 cell x/y 与原始 ID，保留 area 等字段 |
| .gem2.gz | V2 作者代码使用的派生 GEM 文件名 | 同一严格 GEM 解析器；不能仅凭该后缀证明实验为 V2 |

默认保持 GEM 的 #BinSize（无该字段的旧 GEM 为 bin1）；GEF 默认选择最小已存 binN。GEM 可显式选择原有 bin 大小的整数倍，不能上采样。GEF 指定 bin 必须实际存在，不能假装完成重分箱。CellBin 不会默默转为方格。

- 保留所有输入基因／RNA 特征，包括非 poly(A)、非编码或微生物条目；不归一化、不筛基因、不猜分类。
- #OffsetX/#OffsetY 原样保存；GEM 同时给出 `obsm['spatial_global'] = stored_xy + offsets`。`obsm['spatial']` 保持存储坐标，便于追溯。
- `uns['stereoseq']` 保存文件格式、分辨率、源元数据及化学版本证据。GEF 中的 version 仅进入 `gef_schema_version`。
- 未提供化学版本时写 `chemistry='unspecified'`；显式 V1/V2 写 `chemistry_evidence='user_declared'`。格式读取成功不证明化学版本。
- `uns['spatial'][library]` 保存可选图片、尺度、图片状态及空间元数据；`uns['spateo_io']` 保存读取器、参数、输入清单、证据和校验结果。
- 只有 GEF 明确提供 resolution 时才记录 pitch_nm；不一律假设所有 Stereo-seq 芯片均为 500 nm。

## 自动读取的实际逻辑

1. inventory：有界遍历文件。
2. discover：找到 GEM/GEF 等平台矩阵候选。
3. probe_stereo：核对表头／HDF5 路径与复合字段；GEF 检查预计稀疏分配规模，GEM 将全量检查交给流式读取。
4. _resolve：只有唯一通过契约检查的读法才继续；没有固定分数或最高分竞争。
5. read_stereo_core：按容器选择 read_gem/read_gef。
6. read_gem：逐块验证所有记录，汇总到 CSR；read_gef：验证所有偏移、长度、基因索引、计数和坐标。
7. read_core：核对 AnnData 的轴、坐标和表达数值，初始化 Spateo 必要字段。
8. _assets：在资源限额内读取可选图像；大图保留来源并标 deferred_resource，不影响核心矩阵 ready。
9. record_spatial_io：记录来源；协调器返回 ready/failed/deferred/unresolved 及原因。

## 校验及边界

拒绝负数、非整数计数、空 ID、冲突 geneName、重复表头、不明确的计数别名、exon 大于总 MID、整型溢出、GEF 索引越界及块重叠／缺口。cell-bin GEF 若提供 expCount，则还核对逐细胞总计。所有合成测试只证明所覆盖契约正确，不能称为所有实际平台数据的识别准确率。

现有 `read_bgi` / `read_bgi_agg` 分割 API 保留；计数改为 int64，避免 uint16 在 65,535 处溢出，并保留 ExonCount（exon）和可用 geneName。

本次不自动处理 FASTQ/BAM、RPI/IPR 图像配准、细胞边界恢复、免疫受体拼接或剪接事件识别。GEF 使用基因/细胞表达核心，不加载 wholeExp 全芯片稠密图。内存预算是保守分配估计，不是操作系统 RSS 硬上限；不会为了通过预算而丢弃数据。

## 连续切片实测示例

选择 [ARTISTA](https://db.cngb.org/stomics/artista/) 的 **10 DPI 蝾螈端脑系列**。作者明确描述沿 rostral-caudal 轴的 serial sections，10 DPI 共 3 张。读取 `10DPI_1.gem.gz`，另外两张作为同系列目录／图像参考。实际切片间距未在下载页明确给出，因此不虚构 z 坐标，也不宣称是没有缺片的等间距完整三维系列。

[原始矩阵和图像下载页](https://db.cngb.org/stomics/artista/download/) 提供 Bin1 GEM 与核酸染色 TIFF。本例不以已整理 H5AD 为输入；新的 H5AD 只是 Spateo 输出。为了得到可展示的方格矩阵，显式按 bin50 聚合所有原始记录，保留全部计数和基因。核酸染色 TIFF 仅作原图参考，未完成与计数坐标的配准，不能把并排展示解释为注册叠加。

PRISTA4D 的 serial-section 元数据和作者说明已核对，但当前 STOmicsDB 文件接口返回 file_is_public=false，不能把其元数据公开误称为原始 GEM 已成功下载。

**真实 V2 GEM/GEF 的生物样本实测仍需独立文件。** 当前 V2 兼容性依据官方格式、作者实际输入列与合成格式用例；不能把本例旧版 ARTISTA 改称 V2 或宣称已实测 V2 临床样本。

## 本次实际结果（2026-09-21）

- 输入：80,168,545 条 GEM 原始记录；全量读取，没有抽样。
- 输出：53,139 × 48,275；55,386,354 非零矩阵项。
- 输入／输出总 MID：116,127,443／116,127,443。
- 真实样本：1 张、1 次完整读取；独立 pandas 分组复算全部记录，逐项比较 CSR 完全一致；H5AD 往返一致。
- 读取耗时：291.27 秒；独立核验与校验和记录耗时：68.73 秒。耗时来自本机本次运行，不是性能排名。
- IO 测试：123 passed，1 skipped（本轮未配置额外 Visium 环境数据）。其中 Stereo-seq 新增 41 项通过。
- 跨平台合成回归：3 轮（种子 20260921-20260923），11 类技术 × 5 情形，共 165 例，每例 2 次＝330 次调用。66 个有效用例均读取及内容一致，99 个无效用例全部正确拒绝，重复运行 165/165 一致。
- 初次回归发现整数值的 `1.0` 坐标文本被过严拒绝；已修正为精确整数解析（不先转 float，保留超过 2^53 的整数），新增对应测试后全部通过。初次失败记录保留在本地 platform_benchmark；最终结果位于 platform_benchmark_final。
- 真实 V2 生物样本实测数量：0；不把合成格式兼容测试当成真实 V2 实测。

### 当前函数定位

| 函数 | 仓库文件及行号 |
|---|---|
| read_spatial | spateo/io/spatial/auto/_automatic.py:240 |
| inventory | spateo/io/spatial/auto/_discovery.py:35 |
| discover | spateo/io/spatial/auto/_discovery.py:99 |
| probe | spateo/io/spatial/auto/_contracts.py:382 |
| _resolve | spateo/io/spatial/auto/_automatic.py:113 |
| read_core | spateo/io/spatial/auto/_contracts.py:453 |
| probe_stereo | spateo/io/spatial/auto/_stereo.py:123 |
| gem_header | spateo/io/spatial/auto/_stereo.py:53 |
| read_gem | spateo/io/spatial/auto/_stereo.py:142 |
| read_gef | spateo/io/spatial/auto/_stereo.py:289 |
| _assets | spateo/io/spatial/auto/_automatic.py:30 |
| record_spatial_io | spateo/io/spatial/_provenance.py:82 |
| read_stereoseq | spateo/io/spatial/_stereoseq.py:273 |
