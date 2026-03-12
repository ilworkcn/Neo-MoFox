# emoji_sender

一个“表情包收藏并发送”的插件：

- 定时从主程序内置的 `data/media_cache/emojis/` 随机抽取表情包
- 调用 VLM 注入主配置人格（`config/core.toml` 的 `personality`）后，让模型决定是否收藏并输出标注（描述 + 情感 tag）
- 若收藏：复制源文件到 `data/emoji_sender/memes/`，并将描述 embedding 写入 `data/emoji_sender/vector_db/`
- 对外暴露：
  - Action：根据“目标描述 + 情感 tag”发送表情包
  - Service：供其他插件以编程方式检索/发送
- 检索阶段支持通过 `config/plugins/emoji_sender/config.toml` 中的 `vector.temperature` 控制采样强度，避免代表性表情反复被固定选中

说明：情感 tag 预设为插件内置常量，不进入配置。用户手动删除 `data/emoji_sender/memes/` 中不想要的表情后，会在下一次入库任务开始时自动清理数据库对应条目。
