# GitHub 发布

用户指定账号为 **yyyx0402**，仓库名为 **navdiag**，目标为公开开源。当前项目已经准备源码、MIT 许可、复现命令、数据指纹、结果记录、截图、测试与 CI。原始数据和密钥不进入 Git。

2026-10-10 已创建并推送公开仓库：[yyyx0402/navdiag](https://github.com/yyyx0402/navdiag)。使用账号公开 ID 对应的 noreply 邮箱提交，本机授权保存在 Windows 凭据存储。下述登录命令供其他环境复现发布；GitHub CLI 需从 [官方安装入口](https://cli.github.com/) 获取，本次项目缓存中的便携程序不随源码发布。

## 账号与发布

仓库创建和推送需要本机登录 GitHub。用户名本身不能提供写权限；不要在聊天中传 token。可使用 GitHub CLI 浏览器授权：

```powershell
gh auth login --hostname github.com --git-protocol https --web
gh repo create yyyx0402/navdiag --public --source . --remote origin --push
```

发布前应有本地提交，并核对作者署名。使用 GitHub 账户显示的 noreply 邮箱可避免公开个人邮箱；不要猜测邮箱或把别人的身份写入提交。已有远程仓库时先检查其内容，不强制推送覆盖。

## 已完成的验证与发布检查

- 本机 31 项测试通过，包括因果窗口、参考分离、持久化和网页接口。
- 15 行程模拟定位实验、4,575 历元真实处理/训练和 30 题离线诊断记录。
- README 与模型卡明确小样本、未运行真实 API 评测、平面滤波边界。
- 原始文件来源固定到上游 commit，并检查 SHA-256。
- `.gitignore` 排除环境、缓存、第三方数据、模型产物和 `.env`。

推送后还需检查公开地址和 GitHub Actions 实际结果。仓库已经公开、远程 CI 已通过这两点，只能在实际完成后报告。当前公开实验提供可复现工程路径，不能作为完整 GSDC 基准或真实 LLM 答案正确率。
