import streamlit as st
import os
import time
import json
from ad_control_plane import AdMCPControlPlane
from ad_agent_core import init_task_dir, BASE_OUTPUT_DIR

# 页面配置
st.set_page_config(
    page_title="AI 广告视频生成 Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
if "control_plane" not in st.session_state:
    try:
        st.session_state.control_plane = AdMCPControlPlane()
    except Exception as e:
        st.error(f"Failed to initialize AdMCPControlPlane: {e}")
if "task_context" not in st.session_state:
    st.session_state.task_context = None
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 0  # 0: Init, 1: Demand, 2: Story, 3: Scene, 4: Storyboard, 5: Grid, 6: Optimize, 7: Video

STAGES = [
    "任务初始化", 
    "需求分析 (Demand Analysis)", 
    "故事构建 (Story Building)", 
    "场景设计 (Scene Design)", 
    "分镜设计 (Storyboard Design)", 
    "四宫格生图 (Visual Generation)", 
    "图像优化 (Image Optimization)", 
    "视频生成 (Video Production)"
]

def save_uploaded_file(uploaded_file, task_id):
    """保存上传的文件到任务目录"""
    if uploaded_file is None:
        return None
    task_dir = os.path.join(BASE_OUTPUT_DIR, task_id)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        
    file_path = os.path.join(task_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ===================== Sidebar: Global Config =====================
with st.sidebar:
    st.title("🎬 控制台")
    
    api_key_input = st.text_input("OpenAI/Gemini API Key", type="password", value=os.getenv("API_KEY", ""))
    if api_key_input:
        os.environ["API_KEY"] = api_key_input
        
    st.divider()
    
    if st.session_state.task_context:
        st.info(f"Task ID: {st.session_state.task_context['task_id']}")
        st.info(f"Status: {st.session_state.task_context['task_status']}")
        progress = st.session_state.task_context.get("progress", 0)
        st.progress(progress / 100)
    
    if st.button("重置任务"):
        st.session_state.task_context = None
        st.session_state.current_stage = 0
        st.rerun()

# ===================== Main Area =====================
st.title("AI 广告视频生成工作台")

# Stage 0: Initialization
if st.session_state.current_stage == 0:
    st.header("1. 上传素材与设置要求")
    
    col1, col2 = st.columns(2)
    with col1:
        product_img = st.file_uploader("上传产品图片", type=["jpg", "png", "jpeg", "webp"])
        if product_img:
            st.image(product_img, caption="产品预览", use_container_width=True)
            
    with col2:
        persona_img = st.file_uploader("上传人设参考图", type=["jpg", "png", "jpeg", "webp"])
        if persona_img:
            st.image(persona_img, caption="人设预览", use_container_width=True)
            
    user_req = st.text_area("广告具体要求（可选）", placeholder="例如：希望是温馨治愈的风格，强调产品的保湿功能，目标受众是年轻妈妈...")
    
    if st.button("开始生成", type="primary"):
        if not api_key_input:
            st.error("请先在左侧输入 API Key")
        elif not product_img or not persona_img:
            st.warning("请上传产品图和人设图")
        else:
            with st.spinner("正在初始化任务..."):
                # 1. Init Task ID via Control Plane helper (or manually)
                # Ensure we have a context
                dummy_input = {"product_image": "", "character_setting": ""}
                # Initialize Context
                ctx = st.session_state.control_plane.init_ad_task(dummy_input)
                task_id = ctx["task_id"]
                
                # 2. Save Images
                p_path = save_uploaded_file(product_img, task_id)
                c_path = save_uploaded_file(persona_img, task_id)
                
                # 3. Update Context
                ctx["initial_input"] = {
                    "product_image": p_path,
                    "character_setting": c_path,
                    "user_requirements": user_req
                }
                st.session_state.task_context = ctx
                st.session_state.current_stage = 1
                st.rerun()

# Generic Step Runner Helper
def run_module_step(module_name, next_stage_idx):
    cp = st.session_state.control_plane
    ctx = st.session_state.task_context
    
    # Check for modifications/feedback
    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = ""
        
    with st.spinner(f"正在执行：{module_name}..."):
        result = cp._dispatch_module(module_name, ctx)
        
    if result["status"] == "success":
        st.session_state.task_context = cp._update_context(ctx, {module_name + "_info": result["result"]})
        st.success(f"✅ {module_name} 完成")
        return True, result["result"]
    else:
        st.error(f"❌ 执行失败: {result['error']}")
        return False, None

# Stage 1: Demand Interact
if st.session_state.current_stage == 1:
    st.header("2. 需求分析与理解")
    
    if "demand_done" not in st.session_state:
        success, res = run_module_step("demand_interact", 2)
        if success:
            st.session_state.demand_done = True
            st.rerun()
            
    if st.session_state.get("demand_done"):
        info = st.session_state.task_context["demand_interact_info"]
        st.json(info)
        
        st.info("确认上述分析结果，准备生成故事线。")
        if st.button("下一步：生成故事线"):
             st.session_state.current_stage = 2
             del st.session_state.demand_done
             st.rerun()

# Stage 2: Story Builder
elif st.session_state.current_stage == 2:
    st.header("3. 故事线构建")
    
    feedback = st.text_input("修改意见（如需修改，请在此输入后点击“重新生成”）", key="story_fb")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("生成/重新生成故事")
    with col2:
        next_btn = st.button("确认定稿，下一步", type="primary")

    if run_btn:
        if feedback:
            st.session_state.task_context["user_feedback"] = feedback
            # Clear feedback from context after use is handled inside module or we handle it here?
            # Module logic handles `context.get("user_feedback")`
            
        success, res = run_module_step("story_builder", 3)
        if success:
             # Remove feedback to prevent sticky behavior
             if "user_feedback" in st.session_state.task_context:
                 del st.session_state.task_context["user_feedback"]
             st.rerun()

    info = st.session_state.task_context.get("story_builder_info")
    if info:
        st.subheader(f"风格：{info['style_detail']['name']}")
        st.markdown(f"**情感基调**：{info['emotional_tone']}")
        
        for part in info['story_line']:
            with st.expander(f"时间段: {part['duration']}s - 核心: {part['core_point']}", expanded=True):
                st.write(part['content'])
                
    if next_btn:
        if info:
            st.session_state.current_stage = 3
            st.rerun()
        else:
            st.warning("请先生成故事线")

# Stage 3: Scene Design
elif st.session_state.current_stage == 3:
    st.header("4. 场景设计")
    
    if "scene_done" not in st.session_state:
        success, res = run_module_step("scene_designer", 4)
        if success:
            st.session_state.scene_done = True
            st.rerun()

    if st.session_state.get("scene_done"):
        info = st.session_state.task_context.get("scene_designer_info")
        if info:
            for scene in info['scenes']:
                st.markdown(f"### {scene['scene_title']} ({scene['duration']}s)")
                st.write(scene['scene_description'])
                st.divider()
                
            if st.button("下一步：分镜设计"):
                st.session_state.current_stage = 4
                del st.session_state.scene_done
                st.rerun()

# Stage 4: Storyboard Design
elif st.session_state.current_stage == 4:
    st.header("5. 分镜设计与Prompt优化")
    
    feedback = st.text_input("分镜修改意见", key="sb_fb")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("生成/重新生成分镜")
    with col2:
        next_btn = st.button("确认分镜，开始生图", type="primary")

    if run_btn:
        if feedback:
            st.session_state.task_context["user_feedback"] = feedback
            
        success, res = run_module_step("storyboard_designer", 5)
        if success:
             if "user_feedback" in st.session_state.task_context:
                 del st.session_state.task_context["user_feedback"]
             st.rerun()

    info = st.session_state.task_context.get("storyboard_designer_info")
    if info:
         for scene_id, s_data in info["storyboards_by_scene"].items():
             with st.expander(f"场景：{s_data['scene_title']}", expanded=True):
                 for sb in s_data["storyboards"]:
                     st.markdown(f"**Shot {sb['storyboard_id']}**: {sb['画面内容']}")
                     st.caption(f"镜头: {sb['镜头角度']} | 构图: {sb['构图方式']}")

    if next_btn:
        if info:
            st.session_state.current_stage = 5
            st.rerun()
        else:
            st.warning("请先生成分镜")

# Stage 5: Visual Generation (Grid)
elif st.session_state.current_stage == 5:
    st.header("6. AI绘画生成 (四宫格)")
    
    if "grid_done" not in st.session_state:
        success, res = run_module_step("grid_image_generator", 6)
        if success:
            st.session_state.grid_done = True
            st.rerun()
            
    if st.session_state.get("grid_done"):
        info = st.session_state.task_context.get("grid_image_generator_info")
        if info:
            cols = st.columns(3)
            idx = 0
            for scene_id, g_detail in info["grid_image_details"].items():
                with cols[idx % 3]:
                    st.image(g_detail["grid_image_path"], caption=g_detail["scene_title"])
                idx += 1
            
            if st.button("下一步：图像高清化"):
                st.session_state.current_stage = 6
                del st.session_state.grid_done
                st.rerun()

# Stage 6: Optimize
elif st.session_state.current_stage == 6:
    st.header("7. 图像优化 (Super Resolution)")
    
    if "opt_done" not in st.session_state:
        success, res = run_module_step("image_optimizer", 7)
        if success:
            st.session_state.opt_done = True
            st.rerun()
            
    if st.session_state.get("opt_done"):
        info = st.session_state.task_context.get("image_optimizer_info")
        if info:
            st.success("高清化处理完成")
            # Usually strict display not needed here to save bandwidth, just proceed
            if st.button("下一步：生成最终视频"):
                st.session_state.current_stage = 7
                del st.session_state.opt_done
                st.rerun()

# Stage 7: Video
elif st.session_state.current_stage == 7:
    st.header("8. 视频合成")
    
    if "video_done" not in st.session_state:
        success, res = run_module_step("video_generator", 8)
        if success:
            st.session_state.video_done = True
            st.rerun()
            
    if st.session_state.get("video_done"):
        info = st.session_state.task_context.get("video_generator_info")
        if info:
            st.video(info["final_video_path"])
            st.success(f"🎉 视频生成任务完成！保存路径: {info['final_video_path']}")
            st.balloons()
            
            with open(info["final_video_path"], "rb") as f:
                st.download_button("下载视频", f, file_name="ad_video.mp4")
