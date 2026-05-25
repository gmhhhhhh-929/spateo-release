# Spateo 代码合并说明文档

## 合并概述

本文档记录了将 `spateo_release/spateo` (代码2) 的修改合并到 `spateo-protocol` (代码1) 的详细过程和结果。

## 合并内容

### 1. IO模块替换

**原结构** (`spateo/io/`):
- 包含多个独立的读取函数文件：`bgi.py`, `nanostring.py`, `slideseq.py`, `tenx.py` 等
- 使用扁平化结构，所有读取函数直接位于 `io/` 目录下

**新结构** (`spateo/io/` - 来自代码2的 `data_io/`):
```
io/
├── __init__.py          # 统一导出接口
├── single/              # 单细胞数据读取
│   ├── __init__.py
│   ├── _formats.py      # 格式定义
│   └── _read.py         # 通用读取逻辑
├── spatial/             # 空间转录组数据读取
│   ├── __init__.py
│   ├── _visium.py
│   ├── _visium_hd.py
│   ├── _xenium.py
│   ├── _slideseq.py
│   ├── _merfish.py
│   ├── _starmap_plus.py
│   ├── _nanostring.py
│   ├── _seqfish.py
│   ├── _stereoseq.py
│   └── _utils.py
└── general/             # 通用I/O工具
    ├── __init__.py
    ├── _tabular.py      # 表格数据读写
    └── _serialization.py # 序列化支持
```

**关键变更**:
- 采用模块化设计，按数据类型（single/spatial/general）组织代码
- 统一使用 `_read()` 函数作为主入口，支持多种格式自动检测
- 新增 `read_csv()`, `save()`, `load()` 等通用函数
- 所有空间转录组平台读取函数统一命名规范：`read_<platform>()`

### 2. Preprocessing模块替换

**原结构** (`spateo/preprocessing/`):
- 分散的函数文件：`normalize.py`, `filter.py`, `transform.py` 等
- 缺乏统一的预处理流程管理

**新结构** (`spateo/preprocessing/`):
```
preprocessing/
├── __init__.py              # 统一API导出
├── preprocessor.py          # 核心预处理类：Preprocessor, SpatialPreprocessor
├── utils.py                 # 标准化工具函数
├── qc.py                    # 质量控制函数
├── normalization.py         # 归一化方法
├── transform.py             # 数据变换
├── feature.py               # 特征选择
├── pca.py                   # PCA降维
├── graph.py                 # 图构建（空间/表达邻域）
└── external/                # 外部工具集成
```

**关键变更**:
- 引入 `SpatialPreprocessor` 类，支持链式预处理流程
- 新增 `preprocess_spatial()` 便捷函数，一键完成标准预处理
- 所有函数采用一致的参数命名和返回规范
- 增强质量控制功能：`calculate_spatial_qc()`, `filter_spots()`

### 3. Configuration模块更新

**主要变更**:
- 新增错误类导入：`LayerKeyError`, `SpatialKeyError`
- 日志模块切换：从 `.logging` 改为 `.spateo_logger`
- 保持 `SpateoConfig` 和 `SpateoAdataKeyManager` 核心功能不变

### 4. 新增核心文件

#### `_settings.py`
```python
class Colors:
    """ANSI color codes for terminal output styling."""
    HEADER = '\033[95m'     # Purple
    BLUE = '\033[94m'       # Blue
    # ... 其他颜色定义
```
- 提供终端输出的颜色样式常量
- 用于统一的日志和警告信息格式化

#### `_registry.py`
- 函数注册系统，支持自然语言查询和语义搜索
- 装饰器 `@register` 用于自动注册函数到全局索引
- 支持函数别名、分类、描述和示例的元数据管理
- 为未来CLI和文档自动生成提供基础

#### `spateo_logger.py` (从代码2复制)
- 统一的日志管理系统
- 支持多级别日志输出和格式化
- 替代原有的 `.logging` 模块

### 5. Errors模块更新

**新增异常类**:
```python
class SpateoError(Exception):
    """Base exception for Spateo errors."""

class SpatialKeyError(PreprocessingError):
    """Raised when spatial coordinates are missing or invalid."""

class LayerKeyError(PreprocessingError):
    """Raised when a requested AnnData layer is missing."""
```
- 建立异常继承层次，便于错误捕获和处理
- 提供更精确的错误定位信息

## Python版本兼容性

### 支持版本
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12

### 兼容性措施
1. **类型注解**: 使用 `from __future__ import annotations` 延迟求值，避免运行时类型检查问题
2. **语法特性**: 避免使用 `match` 语句（Python 3.10+），保持向下兼容
3. **依赖声明**: `setup.py` 中明确声明支持的Python版本范围
4. **标准库**: 仅使用各版本共有的标准库模块

### 注意事项
- 部分第三方依赖（如 `torch`, `tensorflow`）可能有各自的版本要求，请确保环境依赖满足
- 建议在生产环境使用 `requirements.txt` 或 `pyproject.toml` 锁定依赖版本

## 代码验证

### 导入测试
```bash
cd /Users/gaomohan/Desktop/spateo-protocol
python3 -c "import spateo; from spateo import io, preprocessing, configuration; print('✓ All modules imported successfully')"
```

### 功能测试建议
1. **数据读取测试**:
```python
# 测试单细胞数据读取
from spateo.io import read_10x_h5, read_h5ad
# 测试空间数据读取
from spateo.io import read_visium, read_xenium, read_slideseq
```

2. **预处理流程测试**:
```python
from spateo.preprocessing import SpatialPreprocessor, preprocess_spatial

# 使用类方式
processor = SpatialPreprocessor()
# 或使用便捷函数
adata = preprocess_spatial(adata, layers=['counts'])
```

3. **配置测试**:
```python
from spateo.configuration import config
config.logging_level = 'DEBUG'
config.n_threads = 4
```

## 迁移指南

### 对于现有代码的适配

#### IO模块迁移
```python
# 旧代码
from spateo.io import read_10x, read_10x_as_anndata

# 新代码（保持向后兼容）
from spateo.io import read_10x_h5, read_10x_mtx  # 推荐
# 或继续使用旧名称（通过兼容性导出）
from spateo.io import read_10x  # 仍可用
```

#### Preprocessing模块迁移
```python
# 旧代码
from spateo.preprocessing.normalize import normalize_total
from spateo.preprocessing.transform import log1p, scale

# 新代码（更清晰的命名）
from spateo.preprocessing import normalize_total, log1p_layer, scale_layer
# 注意：log1p -> log1p_layer, scale -> scale_layer (命名更明确)
```

#### 配置模块迁移
```python
# 旧代码
from spateo.logging import logger_manager as lm

# 新代码
from spateo import spateo_logger as lm
```

### 推荐的代码更新步骤
1. 备份现有项目代码
2. 更新 `spateo` 包到合并后的版本
3. 运行导入测试，确认无 `ImportError`
4. 逐步替换旧API调用为新API（利用向后兼容性）
5. 运行单元测试验证功能

## 已知问题与建议

### 注意事项
1. **Matplotlib配置**: 首次导入时可能创建字体缓存，建议设置 `MPLCONFIGDIR` 环境变量
2. **依赖冲突**: 确保 `anndata`, `scanpy`, `numpy`, `pandas` 等核心依赖版本兼容
3. **路径问题**: 空间数据读取函数可能需要绝对路径，建议使用 `pathlib.Path` 处理路径

### 改进建议
1. **文档更新**: 建议同步更新 `README.md` 和 API 文档，反映新的模块结构
2. **类型提示**: 考虑为公共函数添加完整的类型注解，提升IDE支持
3. **测试覆盖**: 为新添加的 `_registry` 系统编写单元测试
4. **性能优化**: 大型空间数据读取时可考虑添加进度条和内存优化

## 文件备份

合并过程中创建了以下备份文件，如有问题可恢复：
- `spateo/io_backup/` - 原始io模块
- `spateo/io_old/` - 重命名前的旧io模块
- `spateo/preprocessing_backup/` - 原始preprocessing模块
- `spateo/preprocessing_old/` - 重命名前的旧preprocessing模块
- `spateo/configuration_backup.py` - 原始configuration.py
- `spateo/configuration_old.py` - 重命名前的旧文件

## 推送准备

### Git操作建议
```bash
# 1. 确认变更
cd /Users/gaomohan/Desktop/spateo-protocol
git status

# 2. 添加新文件（排除备份）
git add spateo/io/ spateo/preprocessing/ spateo/configuration.py \
      spateo/_registry.py spateo/_settings.py spateo/spateo_logger.py \
      spateo/errors.py

# 3. 移除备份文件（可选）
rm -rf spateo/io_backup spateo/io_old spateo/preprocessing_backup \
       spateo/preprocessing_old spateo/configuration_backup.py \
       spateo/configuration_old.py

# 4. 提交变更
git commit -m "feat: merge data_io and preprocessing modules from spateo_release

- Replace io/ with modular data_io/ structure (single/spatial/general)
- Update preprocessing/ with SpatialPreprocessor class and unified API
- Add _registry.py for function discovery system
- Add _settings.py for terminal color constants
- Update configuration.py and errors.py for compatibility
- Ensure Python 3.9-3.12 compatibility with future annotations
"

# 5. 推送
git push origin <your-branch>
```

### 版本建议
- 建议在 `setup.py` 中更新版本号（如从 `1.1.1` 到 `1.2.0`）
- 在 `CHANGELOG.md` 中记录本次合并的主要变更

---
*文档生成日期: 2026-05-25*
*合并操作者: Codex CLI*
