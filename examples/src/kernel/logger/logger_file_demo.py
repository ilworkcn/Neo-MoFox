"""
Logger 文件输出演示脚本

展示如何使用日志文件输出功能。
"""

import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kernel.logger import get_logger, COLOR, RotationMode


def main():
    """演示文件输出功能"""

    # 1. 创建带文件输出的 logger（按日期轮转）
    print("\n=== 创建带文件输出的 Logger ===")
    logger = get_logger(
        "app",
        display="应用日志",
        color=COLOR.CYAN,
        enable_file=True,
        file_rotation=RotationMode.DATE,
    )

    logger.info("这是一条会保存到文件的日志")
    logger.warning("警告信息也会保存到文件")
    logger.error("错误信息同样会保存")

    # 2. 使用元数据的文件输出
    print("\n=== 带元数据的日志 ===")
    logger.set_metadata("user_id", "12345")
    logger.set_metadata("session", "abc-def")
    logger.info("用户登录操作", ip="192.168.1.100", action="login")

    # 3. 创建按大小轮转的 logger
    print("\n=== 创建按大小轮转的 Logger ===")
    size_logger = get_logger(
        "size_test",
        display="大小轮转测试",
        color=COLOR.YELLOW,
        enable_file=True,
        file_rotation=RotationMode.SIZE,
        max_file_size=1024,  # 1KB
    )

    size_logger.info("这条日志会写入按大小轮转的文件")
    size_logger.info("当文件超过1KB时会自动创建新文件")

    # 4. 动态启用/禁用文件输出
    print("\n=== 动态启用/禁用文件输出 ===")
    dynamic_logger = get_logger(
        "dynamic",
        display="动态日志",
        color=COLOR.GREEN,
        enable_file=False,  # 初始不启用
    )

    dynamic_logger.info("这条日志只输出到控制台")

    # 动态启用文件输出
    dynamic_logger.enable_file_output(file_rotation=RotationMode.DATE)
    dynamic_logger.info("现在这条日志会同时输出到控制台和文件")

    # 禁用文件输出
    dynamic_logger.disable_file_output()
    dynamic_logger.info("文件输出已禁用，只输出到控制台")

    # 5. 查看生成的日志文件
    print("\n=== 生成的日志文件 ===")
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        print(f"logs 目录中有 {len(log_files)} 个日志文件:")
        for log_file in sorted(log_files):
            size = log_file.stat().st_size
            print(f"  - {log_file.name} ({size} bytes)")
    else:
        print("logs 目录不存在")

    print("\n=== 演示完成 ===")
    print("日志文件已保存到 logs/ 目录中")
    print("您可以使用文本编辑器查看日志文件内容")


if __name__ == "__main__":
    main()
