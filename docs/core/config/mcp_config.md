# mcp_config 模块

对应源码：src/core/config/mcp_config.py

## 概述

MCPConfig 定义 MCP 功能开关及服务端点配置，供 tool_manager/mcp_manager 初始化连接时使用。

## 配置结构

- mcp.enabled
是否启用 MCP。

- mcp.stdio_servers
字典结构，key 为服务名，value 通常包含 command、args、env，也可附带 instructions、defer_loading。

- mcp.sse_servers
字典结构，key 为服务名，value 可为 URL 字符串，或包含 url、headers、timeout、sse_read_timeout、instructions、defer_loading 的对象。

- mcp.streamable_http_servers
字典结构，key 为服务名，value 可为 URL 字符串，或包含 url、headers、timeout、instructions、defer_loading 的对象。

- defer_loading
单个 MCP 服务级别的工具暴露控制项，默认值为 True。
当为 True 时，该服务工具不会直接暴露给 default chatter 的 actor，只能通过 create_agent 分配给 sub agent 使用。
当为 False 时，该服务工具会像普通工具一样直接暴露给 actor。

## 全局实例管理

- _global_mcp_config: MCPConfig | None
- get_mcp_config: 未初始化时抛 RuntimeError
- init_mcp_config: 自动创建默认文件并加载

## init_mcp_config 行为

1. 配置文件不存在时创建目录并写入默认配置。
2. 使用 MCPConfig.load(config_path) 加载。
3. 返回全局单例。

## 与 MCPManager 协作

MCPManager.initialize 会读取 get_mcp_config：

- enabled=False 时直接跳过。
- stdio_servers 逐项连接。
- sse_servers 逐项连接并发现工具。
- streamable_http_servers 逐项连接并发现工具。
- 已连接服务会缓存 defer_loading 元数据，供 default chatter 决定 actor 是否直接注入对应 MCP 工具。
