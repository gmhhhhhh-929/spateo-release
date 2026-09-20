# Slide-seq 原始大矩阵的逐行读取

`Puck_180413_7.tar` 来自 [SCP354](https://singlecell.broadinstitute.org/single_cell/study/SCP354/slide-seq-study)。解包后的 `MappedDGEForR.csv` 约 1.54 GB，方向为 genes × beads；配套 `BeadLocationsForR.csv` 提供 `barcodes,xcoord,ycoord`。

先解包，然后直接读取包含两个原始 CSV 的目录：

```python
import spateo as st

result = st.io.read_spatial("/path/to/Puck_180413_7", load_images=True)
print(result.report)
adata = result.adata
# 本次真实样本：38,666 beads × 19,869 genes
adata.write_h5ad("Puck_180413_7.h5ad")
result.write_report("spatial_read_report.json")
```

## 为什么增加流式适配器

原表格适配器将全部 CSV 解码为字符串 DataFrame，再转稠密数组和稀疏矩阵。在本例中会因内存预算而推迟读取。新增 `_slideseq_counts` 每次只读一行基因计数，将非零条目累计到紧凑缓冲区，最后转为 beads × genes CSR。没有抽样、基因筛选、归一化、缺失值填零或 ID 重命名。

预检查只检查表头、首行和坐标预览，报告的是行处理空间估计，不宣称已经验证全部内容。实际读取逐行核验字段数、所有数值、基因和条形码标识符，并随非零条目增长检查保守的内存预算；超限仍返回 deferred，而不是扩大预算或删减数据。默认预算保持 1 GiB。

执行链：`read_spatial → inventory → discover → probe → _resolve → SpatialDataset.load → _load → read_core → _meta / _table_counts → _slideseq_counts → _assets → record_spatial_io`。其中预检查会先调用 `_meta` 和 `_slideseq_counts(full=False)`，实际读取才调用 `full=True`。坐标按条形码重排并全量核验。

## 图像与切片含义

`BeadImage.tif` 为 6030 × 6030 的二值珠子图像，不能标为 H&E。它的像素数组超过默认 32 MiB 图像预算；多帧 Channel TIFF 也被明确暂缓。路径、资源状态保存在 `uns['spatial']`，核心表达与坐标仍可 ready。独立导出的无损 PNG 不意味着原始图像数组已加载到 AnnData。

`Worker_1…20_LocalBeadImage.tif` 和各 Channel TIFF 的 20 帧不代表连续组织切片。[作者 puck 批次列表](https://github.com/broadchenf/Slideseq/blob/master/PipelineFunctions/Puck_180413X_Pipeline.m#L64)包含 180413_1 至 180413_8，但编号不提供相邻切片证据。

[原论文](https://macoskolab.com/wp-content/uploads/2019/04/slideseq_full.pdf)确认同一背侧小鼠海马的 66 张矢状面切片系列，交替测量 10 μm 切片，采样间隔约 20 μm。`Puck_180413_7` 在[公开研究的数据说明](https://www.biorxiv.org/content/10.1101/2020.10.13.338475v2.full)中被标为冠状面海马，当前不能将它归入该矢状面系列。确认系列成员与 z 顺序需要作者样本映射，不能从目录树推断。

## 验证范围

本轮 IO 回归：82 项通过，1 项需要另行提供 Visium 数据路径的可选测试未执行。补充跨平台合成验证：11 类格式、2 轮、5 种场景、每案例 2 次，共 110 个案例 / 220 次调用，均符合预期。这是构造案例的结果，不代表真实样本总体准确率。

本例通过完整原始 CSV 的独立 NumPy 整数解析，逐基因核对稀疏值与非零位置；所有表达轴、坐标以及 H5AD 写入回读均纳入核验。展示预览仅列出前两行和最后一行，完整数据仍保存在输出对象中。


实测输出：4,442,379 个非零计数，合计 5,152,436 UMI；全部数值及轴、坐标、写入回读核验通过。带函数追踪的读取耗时约 120 秒（单次运行，不作为性能基准）。
