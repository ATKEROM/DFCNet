# COPIED FROM SELAVI!

import logging
import os
import time
from datetime import timedelta

import pandas as pd


class LogFormatter:
    def __init__(self):

        self.start_time = time.time()

    def format(self, record):

        elapsed_seconds = round(record.created - self.start_time)


        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath):

    log_formatter = LogFormatter()


    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # 创建日志记录器并设置级别为 DEBUG
    logger = logging.getLogger()  # 获取根日志记录器
    logger.handlers = []  # 清除已有的处理器，避免重复记录日志
    logger.setLevel(logging.DEBUG)  # 设置日志记录级别为 DEBUG
    logger.propagate = False  # 禁止日志向上传播，避免重复输出
    if filepath is not None:
        logger.addHandler(file_handler)  # 添加文件处理器
    logger.addHandler(console_handler)  # 添加控制台处理器

    # 重置日志记录器的开始时间
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # 将重置时间函数绑定到日志记录器

    return logger  # 返回配置好的日志记录器


class PD_Stats(object):
    """
    使用 pandas 库记录统计信息
    """

    def __init__(self, path, columns):
        self.path = path  # 统计信息文件的路径

        # 如果路径存在，则重新加载统计信息
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)  # 从文件中加载统计信息

            # 检查列是否与提供的列相同
            assert list(self.stats.columns) == list(columns), "列不匹配，可能的原因是统计信息文件格式不正确。"

        else:
            # 如果文件不存在，则创建一个新的 DataFrame
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        # 更新统计信息，将新的一行添加到 DataFrame
        self.stats.loc[len(self.stats.index)] = row

        # 保存统计信息到文件中
        if save:
            self.stats.to_pickle(self.path)  # 使用 pickle 序列化保存统计信息到文件中
