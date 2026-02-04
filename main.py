from ad_control_plane import AdMCPControlPlane
from ad_agent_core import print_separator

if __name__ == "__main__":
    # ===================== 配置初始输入（仅需修改此处）=====================
    # 1. 产品图片：本地路径或网络URL
    PRODUCT_IMAGE = "ad_input/product.jpg"  # 替换为你的产品图片路径/URL
    
    # 2. 人设图片：本地路径（系统升级为视觉理解模式，请输入人设参考图路径）
    # 原有的字典格式已废弃，改为直接提供图片路径
    CHARACTER_SETTING = "ad_input/character1.jpg" 
    # 初始化初始输入字典
    INITIAL_INPUT = {
        "product_image": PRODUCT_IMAGE,
        "character_setting": CHARACTER_SETTING
    }

    # ===================== 启动广告生成全流程 =====================
    # 1. 实例化MCP控制平面
    ad_agent = AdMCPControlPlane()
    # 2. 运行广告任务
    result = ad_agent.run_ad_task(INITIAL_INPUT)
    # 3. 打印最终结果
    print_separator("广告任务执行结果")
    if result["code"] == 200:
        print(f"✅ 任务执行成功！")
        print(f"📌 任务ID：{result['task_id']}")
        print(f"🎬 最终视频：{result['final_video_path']}")
        print(f"📁 任务目录：{result['task_dir']}（含四宫格、超分图、最终视频）")
    else:
        print(f"❌ 任务执行失败！")
        print(f"错误信息：{result['error']}")
        if "current_step" in result:
            print(f"失败步骤：{result['current_step']}")