from ad_agent_core import (
    AdModule, generate_task_id, init_task_dir, save_context, load_context,
    user_confirm, print_separator, PAUSE_NODES, BASE_OUTPUT_DIR
)
from ad_modules import (
    DemandInteractModule, StoryBuilderModule, SceneDesignModule,
    StoryboardDesignModule, GridImageGenerateModule, ImageOptimizeModule,
    VideoGenerateModule
)
from typing import Dict, Any, List
import time
import os

class AdMCPControlPlane:
    """产品广告图生视频Agent MCP核心控制平面"""
    def __init__(self):
        # 1. 初始化模块注册表（动态注册所有功能模块）
        self.module_registry: Dict[str, AdModule] = self._register_all_modules()
        # 2. 初始化广告生成固定调度流程
        self.schedule_flow: List[str] = [
            "demand_interact", "story_builder", "scene_designer",
            "storyboard_designer", "grid_image_generator", "image_optimizer",
            "video_generator"
        ]
        # 3. 初始化全局上下文模板
        self.context_template: Dict[str, Any] = {
            "task_id": None,
            "task_status": "init",  # init/processing/finished/failed/paused
            "current_step": None,
            "progress": 0.0,        # 0-100
            "create_time": None,
            "update_time": None,
            "initial_input": {},    # 产品图片、人物人设
            "demand_info": {},      # 需求交互模块输出
            "story_info": {},       # 故事构建模块输出
            "scene_info": {},       # 场景设计模块输出
            "storyboard_info": {},  # 分镜设计模块输出
            "grid_image_info": {},  # 四宫格生图模块输出
            "hd_image_info": {},    # 图像优化模块输出
            "video_info": {},       # 视频生成模块输出
            "module_status": {mod: "idle" for mod in self.schedule_flow},  # idle/busy/success/failed
            "error_log": []
        }

    def _register_all_modules(self) -> Dict[str, AdModule]:
        """动态注册所有7个功能模块（支持热插拔，新增模块仅需在此添加）"""
        modules = [
            DemandInteractModule(), StoryBuilderModule(), SceneDesignModule(),
            StoryboardDesignModule(), GridImageGenerateModule(), ImageOptimizeModule(),
            VideoGenerateModule()
        ]
        module_registry = {mod.module_name: mod for mod in modules}
        print(f"✅ MCP控制平面初始化完成，已注册模块：{list(module_registry.keys())}")
        return module_registry

    def init_ad_task(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """初始化广告任务：生成任务ID、创建目录、初始化上下文"""
        task_id = generate_task_id()
        create_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # 初始化上下文
        context = self.context_template.copy()
        context.update({
            "task_id": task_id,
            "create_time": create_time,
            "update_time": create_time,
            "initial_input": initial_input,
            "task_status": "processing"
        })
        # 初始化任务目录
        init_task_dir(task_id)
        # 保存初始上下文
        save_context(task_id, context)
        print(f"✅ 广告任务初始化成功 | 任务ID：{task_id} | 存储目录：{os.path.join(BASE_OUTPUT_DIR, task_id)}")
        return context

    def _update_context(self, context: Dict[str, Any], update_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新全局上下文，同步保存到本地"""
        context.update(update_data)
        context["update_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        save_context(context["task_id"], context)
        return context

    def _dispatch_module(self, module_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """调度单个模块执行：输入校验→状态更新→执行→结果返回"""
        task_id = context["task_id"]
        module = self.module_registry[module_name]
        
        # 1. 输入校验
        if not module.validate_input(context):
            error_msg = f"模块【{module_name}】输入校验失败，上下文缺失必要数据"
            self._update_context(context, {
                "module_status": {**context["module_status"], module_name: "failed"},
                "error_log": context["error_log"] + [{"step": module_name, "error": error_msg, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]
            })
            return {"status": "failed", "result": {}, "error": error_msg}
        
        # 2. 更新模块状态为busy
        context = self._update_context(context, {
            "current_step": module_name,
            "module_status": {**context["module_status"], module_name: "busy"},
            "progress": self._calculate_progress(module_name)
        })
        
        # 3. 执行模块
        try:
            module_result = module.run(task_id, context)
            # 更新模块状态为success/failed
            new_module_status = "success" if module_result["status"] == "success" else "failed"
            context = self._update_context(context, {
                "module_status": {**context["module_status"], module_name: new_module_status},
                "update_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            })
            # 记录错误日志
            if module_result["status"] == "failed":
                context["error_log"].append({
                    "step": module_name,
                    "error": module_result["error"],
                    "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                })
                save_context(task_id, context)
            return module_result
        except Exception as e:
            error_msg = f"模块【{module_name}】执行异常：{str(e)}"
            context = self._update_context(context, {
                "module_status": {**context["module_status"], module_name: "failed"},
                "error_log": context["error_log"] + [{"step": module_name, "error": error_msg, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]
            })
            return {"status": "failed", "result": {}, "error": error_msg}

    def _calculate_progress(self, current_module: str) -> float:
        """计算任务进度（按调度流程平均分配）"""
        step_idx = self.schedule_flow.index(current_module)
        total_steps = len(self.schedule_flow)
        return round((step_idx + 1) / total_steps * 100, 2)

    def _check_pause_node(self, module_name: str) -> bool:
        """检查是否为人工干预暂停节点"""
        return module_name in PAUSE_NODES

    def run_ad_task(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行广告生成全流程（核心调度方法）
        :param initial_input: 初始输入 → {"product_image": 路径/URL, "character_setting": [人设字典]}
        :return: 最终任务结果
        """
        try:
            print_separator("启动产品广告图生视频Agent（MCP架构）")
            # 1. 初始化任务
            context = self.init_ad_task(initial_input)
            task_id = context["task_id"]
            
            # 2. 按固定流程调度模块
            for module_name in self.schedule_flow:
                print(f"\n📌 开始执行步骤：{module_name} | 当前进度：{context['progress']}%")
                # 调度模块执行
                module_result = self._dispatch_module(module_name, context)
                if module_result["status"] == "failed":
                    # 模块执行失败，终止任务
                    final_context = self._update_context(context, {
                        "task_status": "failed",
                        "error_log": context["error_log"] + [{"step": module_name, "error": module_result["error"], "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}]
                    })
                    return {
                        "code": 500,
                        "task_id": task_id,
                        "status": "failed",
                        "current_step": module_name,
                        "error": module_result["error"],
                        "context": final_context
                    }
                
                # 3. 将模块结果更新至上下文
                context = self._update_context(context, {module_name + "_info": module_result["result"]})
                
                # 4. 人工干预：暂停节点等待用户确认/修改
                if self._check_pause_node(module_name):
                    print_separator(f"【人工干预节点】{module_name}执行完成")
                    if module_name == "story_builder":
                        prompt = "是否确认当前故事线和视觉风格？"
                    else:  # storyboard_designer
                        prompt = "是否确认当前所有分镜设计？"
                    
                    if user_confirm(prompt):
                        print("✅ 用户确认，继续执行下一个步骤...")
                        continue
                    else:
                        # 用户选择修改，接收反馈并重新执行
                        print("🔄 用户选择修改，重新执行当前步骤...")
                        user_feedback = input("请输入具体的修改建议（例如：希望更幽默一些/分镜太少）：").strip()
                        
                        # 将用户反馈临时注入上下文
                        if user_feedback:
                            context["user_feedback"] = user_feedback
                            print(f"📝 已记录修改建议，正在传递给大模型...")
                            
                        module_result = self._dispatch_module(module_name, context)
                        
                        # 清理临时反馈，避免污染后续流程
                        if "user_feedback" in context:
                            del context["user_feedback"]

                        if module_result["status"] == "failed":
                            final_context = self._update_context(context, {"task_status": "failed"})
                            return {
                                "code": 500,
                                "task_id": task_id,
                                "status": "failed",
                                "current_step": module_name,
                                "error": module_result["error"],
                                "context": final_context
                            }
                        # 更新修改后的结果
                        context = self._update_context(context, {module_name + "_info": module_result["result"]})
                        print("✅ 修改完成，继续执行下一个步骤...")
            
            # 3. 全流程执行完成
            final_context = self._update_context(context, {
                "task_status": "finished",
                "progress": 100.0
            })
            return {
                "code": 200,
                "task_id": task_id,
                "status": "success",
                "final_video_path": final_context["video_info"]["final_video_path"],
                "task_dir": os.path.join(BASE_OUTPUT_DIR, task_id),
                "context": final_context
            }
        except Exception as e:
            return {
                "code": 500,
                "status": "failed",
                "error": f"广告任务执行异常：{str(e)}",
                "context": context if 'context' in locals() else {}
            }