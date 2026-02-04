import uuid
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json

# ===================== 全局基础配置（按示例定义，可直接修改）=====================
# 广告固定总时长（秒）
AD_TOTAL_DURATION = 60
# 支持的视觉风格枚举及详细定义

# 人物人设固定字段
CHARACTER_FIELDS = ["name", "gender", "age", "appearance", "dress_style", "personality", "signature_action"]
# 场景数量（固定3个）
SCENE_NUM = 3
# 单场景分镜数量（固定4个）
STORYBOARD_PER_SCENE = 4
# 场景时长分配（3个场景，总60s）
SCENE_DURATIONS = [15, 30, 15]
# 人工干预暂停节点（模块名称）
PAUSE_NODES = ["story_builder", "storyboard_designer"]
# API模块重试规则
API_RETRY_TIMES = 2
API_RETRY_INTERVAL = 1  # 秒
# 本地存储配置
BASE_OUTPUT_DIR = "./ad_output"
IMAGE_FORMAT = "PNG"
IMAGE_RESOLUTION = "1080*1920"
VIDEO_FORMAT = "MP4"
VIDEO_RESOLUTION = "1080P"
VIDEO_FPS = 30
# 分镜必须包含的描述项
STORYBOARD_MANDATORY_ITEMS = ["画面内容", "镜头角度", "人物动作/表情", "构图方式", "画面色调/光影"]

# ===================== 抽象模块接口（所有功能模块必须实现）=====================
class AdModule(ABC):
    """广告Agent模块抽象基类"""
    module_name: str  # 模块唯一标识

    @abstractmethod
    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        模块核心执行方法
        :param task_id: 任务唯一ID
        :param context: 全局上下文
        :return: 执行结果（status: success/failed, result: 结果数据, error: 错误信息）
        """
        pass

    @abstractmethod
    def validate_input(self, context: Dict[str, Any]) -> bool:
        """
        输入校验：确保上下文包含模块所需数据
        :param context: 全局上下文
        :return: 校验通过True，否则False
        """
        pass

# ===================== 全局工具函数 =====================
def generate_task_id() -> str:
    """生成8位唯一任务ID"""
    return str(uuid.uuid4())[:8]

def init_task_dir(task_id: str) -> str:
    """初始化任务本地存储目录"""
    task_dir = os.path.join(BASE_OUTPUT_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(os.path.join(task_dir, "grid_images"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "hd_images"), exist_ok=True)
    os.makedirs(os.path.join(task_dir, "videos"), exist_ok=True)
    return task_dir

def save_context(task_id: str, context: Dict[str, Any]) -> None:
    """保存上下文到本地，支持断点续跑"""
    task_dir = os.path.join(BASE_OUTPUT_DIR, task_id)
    with open(os.path.join(task_dir, "ad_context.json"), "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

def load_context(task_id: str) -> Dict[str, Any]:
    """从本地加载上下文"""
    task_dir = os.path.join(BASE_OUTPUT_DIR, task_id)
    with open(os.path.join(task_dir, "ad_context.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def user_confirm(prompt: str) -> bool:
    """用户确认交互：返回True确认，False修改"""
    while True:
        res = input(f"{prompt}（确认请输入y/修改请输入n）：").strip().lower()
        if res in ["y", "n"]:
            return res == "y"
        print("输入无效，请输入y或n！")

def print_separator(title: str) -> None:
    """打印分隔符，优化控制台输出"""
    print(f"\n{'='*20} {title} {'='*20}")