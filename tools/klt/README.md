# KLT 原始数据预处理

这是独立、可选的 **GPL-3.0-or-later** 预处理工具，许可证见 [LICENSE](LICENSE)。NavDiag 主 Python 包使用项目根目录的 MIT 许可证，不包含这个目录或上游实现源码。

工具在本地缓存下载、校验并调用 [Fossett Laboratory 的 GPL 特征提取器](https://github.com/fossettlab/gnss-nlos-notebook/blob/b13aa856f723a472bd35d45171017580642054cf/extract_features.py)，统一重算九段 KLT 观测的特征。缓存源码、第三方数据和模型产物都在 Git 忽略目录中。运行记录包含上游提交、文件校验和及每段的时间对齐证据。

标签读取采用上游限制反序列化器，只允许 NumPy 数组和基本容器需要的全局对象。四段已有数据的标签来自固定版本的公开数值表，五段新增标签来自原始相机标注；位置真值和相机图片不进入模型特征。观测与标签逐历元按时间戳核对，不能直接截断到两个数组的较小长度。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[temporal,gnss]"
.\.venv\Scripts\python.exe scripts/fetch_nlos_data.py
.\.venv\Scripts\python.exe scripts/fetch_klt_extended.py
.\.venv\Scripts\python.exe tools/klt/prepare.py
```

原始标签包含图片，所需五段标签和三个日期的观测合计约 3.4 GB；不下载 ROS 包。仅将第三方数据用于本地实验，不将其重新发布到本项目仓库。原作者明确指出 `1116_KLT3_649` 的标签不可靠，因此该段被排除。
