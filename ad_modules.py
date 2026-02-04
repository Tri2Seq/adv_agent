from ad_agent_core import (
    AdModule, SCENE_NUM, SCENE_DURATIONS, STORYBOARD_PER_SCENE,
    STORYBOARD_MANDATORY_ITEMS, API_RETRY_TIMES, API_RETRY_INTERVAL,
    init_task_dir, print_separator,
    IMAGE_FORMAT, IMAGE_RESOLUTION, VIDEO_FORMAT, VIDEO_RESOLUTION, VIDEO_FPS, AD_TOTAL_DURATION
)
from typing import Dict, Any, List
import time
import os
import json
import base64
import requests
from openai import OpenAI

# 初始化OpenAI客户端
# 优先从环境变量获取API Key，如果未设置则请在下方填入
api_key = os.getenv("API_KEY") 
if not api_key:
    # 可以在这里填入默认Key或者在运行时报错提示
    print("⚠️ 警告：未检测到环境变量 'API_KEY'。") 
    api_key = "YOUR_API_KEY_PLACEHOLDER"

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.apiyi.com/v1"
)

# ===================== 1. 需求交互模块：采集产品/广告核心信息 =====================
class DemandInteractModule(AdModule):
    module_name = "demand_interact"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验初始输入：产品图片、人设图片路径"""
        return all(key in context["initial_input"] for key in ["product_image", "character_setting"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始采集广告需求")
            initial_input = context["initial_input"]
            
            # 从输入获取图片路径
            product_image_path = initial_input["product_image"]
            persona_image_path = initial_input["character_setting"]
            
            # 接收用户广告要求
            # 优先从context获取（适配UI/API调用），否则从命令行获取
            if "user_requirements" in context["initial_input"]:
                user_requirements = context["initial_input"]["user_requirements"]
                print(f"收到用户要求（来自上下文）: {user_requirements}")
            else:
                user_requirements = input("请输入您的具体广告要求（例如：风格偏好、重点突出的功能、目标受众等，可留空）：").strip()

            print(f"📸 正在使用多模态模型(gpt-4o)分析图片...")
            print(f"   - 产品图: {product_image_path}")
            print(f"   - 人设图: {persona_image_path}")
            # print(f"   - 用户要求: {user_requirements}") # 可选打印
            
            # 调用多模态模型分析
            analysis_result = self._analyze_images_by_vlm(product_image_path, persona_image_path, user_requirements)
            
            print("\n✅ 图片分析完成！已自动提取以下信息：")
            print(json.dumps(analysis_result, indent=2, ensure_ascii=False))

            # 构造 demand_info
            demand_info = {
                "product_image": product_image_path,
                "character_setting": analysis_result.get("character_setting", [{"name": "AI Generated", "gender": "未知", "age": "未知"}]),
                "product_category": analysis_result.get("product_category", "通用描述"),
                "core_selling_points": analysis_result.get("core_selling_points", ["暂无卖点"]),
                "target_audience": analysis_result.get("target_audience", "通用人群"),
                "ad_core_demand": analysis_result.get("ad_core_demand", "品牌推广"),
                "style_preference": analysis_result.get("suggested_visual_style", "现代简约"),
                "advertising_slogan": analysis_result.get("advertising_slogan", "未生成Slogan")
            }
            
            return {
                "status": "success",
                "result": demand_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"需求采集失败：{str(e)}"
            }

    def _image_to_base64(self, image_path: str) -> str:
        """将本地图片转换为 base64 编码"""
        # 简单去除可能存在的引号
        image_path = image_path.strip('"').strip("'")
        if not os.path.exists(image_path):
            return ""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_images_by_vlm(self, product_path: str, persona_path: str, user_requirements: str = "") -> Dict[str, Any]:
        """模仿用户提供的 requests 方式调用 gemini-2.5-pro"""
        
        b64_product = self._image_to_base64(product_path)
        b64_persona = self._image_to_base64(persona_path)
        
        if not b64_product or not b64_persona:
            print("⚠️ 警告：找不到本地图片，返回模拟数据。")
            return {
                "product_category": "模拟产品",
                "core_selling_points": ["模拟特性1", "模拟特性2"],
                "target_audience": "模拟人群",
                "character_setting": [{"name": "MockUser", "gender": "女", "age": "24"}],
                "suggested_visual_style": "赛博朋克",
                "ad_core_demand": "模拟发布",
                "advertising_slogan": "模拟Slogan"
            }

        url = "https://api.apiyi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # 将用户要求拼接进 prompt
        req_str = ""
        if user_requirements:
             req_str = f"\\n【用户额外要求】：{user_requirements}\\n请根据用户要求并结合图片内容进行分析。"

        prompt = f"""
        请分析这两张图片（图1为产品，图2为目标用户/人设）。{req_str}
        输出一个JSON对象（不要Markdown，纯JSON），包含以下字段：
        - product_category: 产品品类
        - core_selling_points: [最多3个核心卖点]
        - target_audience: 目标人群特征
        - character_setting: [{{"name": "为图2人物起名", "gender": "性别", "age": "年龄"}}] (注意是列表包含字典)
        - ad_core_demand: 推测的广告核心诉求
        - suggested_visual_style: 推荐的视觉风格
        - advertising_slogan: 一句吸引人的广告语
        """

        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_product}"
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_persona}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        # 增加容错
        try:
            # 图片分析可能耗时较长，增加超时时间到300秒
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                print("✅ VLM API 调用成功")
                content = response.json()['choices'][0]['message']['content']
                # 清洗 JSON
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content.strip())
            else:
                print(f"❌ VLM API Error: {response.text}")
                raise Exception("API调用失败")
        except Exception as e:
            print(f"❌ VLM 解析异常: {e}")
            # 返回空结构避免后续Crash
            return {
                "product_category": "解析失败",
                "core_selling_points": [],
                "target_audience": "未知",
                "character_setting": [{"name": "Unknown", "gender": "Unknown", "age": "Unknown"}],
                "suggested_visual_style": "默认",
                "advertising_slogan": ""
            }

# ===================== 2. 故事构建模块：生成1min广告故事线+定版风格 =====================
class StoryBuilderModule(AdModule):
    module_name = "story_builder"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：需求采集结果"""
        return "demand_interact_info" in context

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始构建广告故事线")
            demand_info = context["demand_interact_info"]
            
            # 强制使用大模型解析所有风格，不使用预设
            user_style = demand_info["style_preference"]
            style_key = "custom" # 统—标记为自定义
            
            # 调用大模型解析风格详情
            style_detail = self._expand_style_by_llm(user_style)
            
            core_selling_points = demand_info["core_selling_points"]
            character = demand_info["character_setting"][0]  # 主人物
            
            # 检查是否有用户反馈
            user_feedback = context.get("user_feedback")

            # 调用大模型生成故事线
            story_parts = self._generate_story_by_llm(demand_info, style_detail, user_feedback)
            
            story_info = {
                "final_style": style_key,
                "style_detail": style_detail,
                "emotional_tone": style_detail.get("tone", "积极正面"), # 使用LLM生成的tone
                "story_line": story_parts,
                "total_duration": sum(p["duration"] for p in story_parts),
                "selling_point_placement": {f"part{i+1}": p.get("core_point", "通用") for i, p in enumerate(story_parts)}
            }
            
            # 打印故事线，等待用户确认（人工干预）
            print("\n📖 生成的1min广告故事线：")
            for i, part in enumerate(story_parts, 1):
                print(f"第{i}段（{part['duration']}s）：{part['content']}【核心：{part['core_point']}】")
            print(f"视觉风格：{style_detail['name']} | 情感基调：{story_info['emotional_tone']}")
            
            return {
                "status": "success",
                "result": story_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"故事构建失败：{str(e)}"
            }

    def _generate_story_by_llm(self, demand_info: Dict[str, Any], style_detail: Dict[str, Any], user_feedback: str = None) -> List[Dict[str, Any]]:
        """调用大模型生成广告故事线"""
        print("🎬 正在调用大模型编写广告剧本...")
        
        system_prompt = "你是一位获得奥斯卡奖的广告导演和编剧，擅长创作极具吸引力、视觉感强且能高效转化用户的短视频脚本。"
        
        feedback_str = f"\n【用户修改意见】\n用户对之前的版本提出了修改建议，请务必遵守：{user_feedback}\n" if user_feedback else ""

        user_prompt = f"""
        请根据以下信息，创作一个60秒的广告故事线（Storyboard Script）。
        {feedback_str}
        【产品信息】
        - 品类：{demand_info['product_category']}
        - 核心卖点：{', '.join(demand_info['core_selling_points'])}
        - 目标人群：{demand_info['target_audience']}
        - 广告核心诉求：{demand_info['ad_core_demand']}
        - 广告语（Slogan）：{demand_info['advertising_slogan']}
        
        【视觉与角色】
        - 视觉风格：{style_detail['name']}
        - 风格氛围：{style_detail['atmosphere']}
        - 主角设定：{demand_info['character_setting'][0]['name']} ({demand_info['character_setting'][0]['gender']}, {demand_info['character_setting'][0]['age']})
        
        【创作要求】
        1. 总时长严格控制在60秒左右。
        2. 结构划分为4个部分：
           - Part 1 (开篇, ~10s): 黄金前3秒原则，迅速抓住注意力，引出痛点或产品。
           - Part 2 (发展, ~25s): 场景化展示产品核心卖点，体现使用过程。
           - Part 3 (高潮, ~20s): 情绪升华，展示使用后的惊艳效果，强化核心诉求。
           - Part 4 (结尾, ~5s): 品牌露出，Slogan口播，强力行动号召（Call to Action）。
        3. 剧本内容要画面感强，适合AI视频生成。
        
        【返回格式】
        请直接返回一个JSON数组（List），不要包含Markdown格式，每个元素包含：
        - "duration": (int) 估计时长
        - "content": (str) 详细画面与剧情描述
        - "core_point": (str) 该片段对应解决的卖点或营销目的
        """

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            content = completion.choices[0].message.content.strip()
            # 清理可能的markdown标记
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            
            story_line = json.loads(content.strip())
            return story_line
        except Exception as e:
            print(f"❌ 剧本生成失败，回退到默认模板。错误：{e}")
            # Fallback
            char_name = demand_info['character_setting'][0]['name']
            return [
                {"duration": 10, "content": f"{char_name}出场，身处{style_detail['atmosphere']}场景，首次展示产品", "core_point": "吸引注意"},
                {"duration": 25, "content": f"{char_name}使用{demand_info['product_category']}，详细展示{demand_info['core_selling_points'][0]}", "core_point": demand_info['core_selling_points'][0]},
                {"duration": 20, "content": f"展示效果，情绪饱满，体现{demand_info['core_selling_points'][1]}", "core_point": "效果展示"},
                {"duration": 5, "content": f"产品特写，出现广告语【{demand_info['advertising_slogan']}】", "core_point": "品牌露出"}
            ]

    def _expand_style_by_llm(self, user_style: str) -> Dict[str, Any]:
        """调用大模型API解析用户自定义风格"""
        print(f"⚠️ 正在调用大模型解析视觉风格：“{user_style}”...")
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的广告视觉风格分析师。请根据用户提供的风格名称，生成详细的视觉风格定义。"
                    },
                    {
                        "role": "user",
                        "content": f"请解析视觉风格“{user_style}”，并返回一个JSON对象，包含以下字段：\n"
                                   f"- name: 风格名称（原样返回）\n"
                                   f"- core_ele: 核心元素（3-5个，顿号分隔）\n"
                                   f"- color: 配色方案（详细描述）\n"
                                   f"- lighting: 光影质感（详细描述，如‘柔和自然光，低对比度’或‘霓虹光影，强对比’）\n"
                                   f"- atmosphere: 整体氛围（3个形容词，顿号分隔）\n"
                                   f"- props: 典型道具（3个，顿号分隔）\n"
                                   f"- tone: 情感基调（如‘时尚亲和’、‘高冷科技’等）\n"
                                   f"- expression: 人物典型表情/神态（如‘眼神坚毅，嘴角上扬’）\n\n"
                                   f"请仅返回JSON字符串，不要包含markdown格式或其他文本。"
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            content = completion.choices[0].message.content.strip()
            # 清理可能的markdown标记
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            print(f"API调用失败，使用默认值。错误：{e}")
            return {
                "name": user_style,
                "core_ele": f"与{user_style}相关的通用元素",
                "color": "自然色调",
                "lighting": "自然光",
                "atmosphere": "独特风格",
                "props": "相关道具",
                "tone": "积极正面",
                "expression": "自然微笑"
            }

    # 已移除 _get_emotional_tone 方法，转由LLM生成

# ===================== 3. 场景设计模块：生成3个场景+时长+核心元素 =====================
class SceneDesignModule(AdModule):
    module_name = "scene_designer"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：故事线、需求信息"""
        return all(key in context for key in ["story_builder_info", "demand_interact_info"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始设计广告场景")
            story_info = context["story_builder_info"]
            demand_info = context["demand_interact_info"]
            style_detail = story_info["style_detail"]
            
            # 从 initial_input 或 demand_info 中获取人设图路径
            initial_input = context.get("initial_input", {})
            persona_image_path = initial_input.get("character_setting", "")
            user_requirements = initial_input.get("user_requirements", "")
            # 获取产品图路径 (从 demand_info)
            product_image_path = demand_info.get("product_image", "")
            
            # 调用大模型生成场景设计（不再使用固定规则）
            scenes = self._generate_scenes_by_llm(demand_info, story_info, style_detail, persona_image_path, product_image_path, user_requirements)
            
            # 为每个场景生成广角场景图（Concept Art）
            task_dir = init_task_dir(task_id)
            scene_image_dir = os.path.join(task_dir, "scene_images")
            if not os.path.exists(scene_image_dir):
                os.makedirs(scene_image_dir)
                
            print("\n🖌️ 正在绘制广角场景图（Concept Art）作为拍摄基准...")

            for i, scene in enumerate(scenes):
                print(f"   - 正在生成场景【{scene['scene_title']}】的基准环境图...")
                
                # 构造Prompt：强调广角、环境、无人物或远景
                scene_prompt = (
                    "Wide-angle shot, establishing shot, environment concept art. "
                    f"Scene: {scene['scene_description']}. "
                    f"Style: {style_detail['name']}, {style_detail['atmosphere']}, {style_detail.get('lighting', 'natural light')}. "
                    f"No text, high quality, 8k resolution, cinematic lighting. "
                    "Focus on the background and environment layout."
                )
                
                scene_image_path = os.path.join(scene_image_dir, f"{scene['scene_id']}_concept.png")
                self._call_gen_image_api_for_scene(scene_prompt, scene_image_path, product_image_path, persona_image_path)
                
                # 将场景图路径存入scene信息中
                scene["scene_image_path"] = scene_image_path
                print(f"     ✅ 已保存：{scene_image_path}")

            # 打印生成结果
            for i, scene in enumerate(scenes):
                print(f"✅ {scene['scene_title']}（{scene['duration']}s）：{scene['scene_description']}【核心卖点：{scene['core_selling_point']}】")
            
            scene_info = {
                "scene_num": len(scenes),
                "total_duration": sum(s["duration"] for s in scenes),
                "scenes": scenes,
                "style_consistency": style_detail["name"]
            }
            print("✅ 广告场景设计完成！")
            return {
                "status": "success",
                "result": scene_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"场景设计失败：{str(e)}"
            }

    def _generate_scenes_by_llm(self, demand_info: Dict[str, Any], story_info: Dict[str, Any], style_detail: Dict[str, Any], persona_image_path: str = "", product_image_path: str = "", user_requirements: str = "") -> List[Dict[str, Any]]:
        """调用 VLM 根据剧本、风格和人设参考图设计场景"""
        print("🎬 正在调用大模型设计广告场景（纯静态环境图）...")
        
        system_prompt = "你是一位世界顶级的游戏/电影场景概念设计师（Environment Concept Artist）。你擅长设计纯粹的、无人的、高美感的静态环境图，为后续拍摄提供美术资产。"
        
        story_line_str = json.dumps(story_info['story_line'], ensure_ascii=False, indent=2)
        character = demand_info['character_setting'][0]
        
        req_str = ""
        if user_requirements:
             req_str = f"\\n【用户额外设计要求】：{user_requirements}\\n"

        user_prompt = f"""
        请参考提供的【风格参考图】（图1为人设，图2为产品，如有），根据以下广告剧本，拆解出 {SCENE_NUM} 个核心**拍摄场地（Environment）**。
        
        【重要原则】
        1. **纯净环境**：你描述的是空无一人的静态场景图（Concept Art），绝对**不要**包含任何人物、动作、剧情或镜头语言。
        2. **场景一致性**：这些场景是后续分镜生成的"舞台"，必须稳重、细节丰富且风格统一。
        3. **视觉风格**：必须与参考图（特别是人设图体现的氛围）保持一致。{req_str}
        
        【产品信息】
        - 品类：{demand_info['product_category']}
        - 核心卖点：{', '.join(demand_info['core_selling_points'])}
        
        【视觉风格】
        - 风格名称：{style_detail['name']}
        - 核心元素：{style_detail['core_ele']}
        - 氛围：{style_detail['atmosphere']}
        - 光影：{style_detail.get('lighting', '自然光')}
        
        【广告剧本】
        {story_line_str}
        
        【设计要求】
        1. 提取剧本中涉及的 {SCENE_NUM} 个物理场地域。
        2. "scene_description" 必须是极其细致的环境描写（天气、光线、建筑材质、道具陈设、色彩倾向）。
        3. **再次强调**：场景描述中不能有人！不能有动作！只写环境！
        
        【返回格式】
        请直接返回一个JSON数组，包含 {SCENE_NUM} 个对象，不要包含Markdown格式，每个对象包含：
        - "scene_id": "scene_x"
        - "scene_title": (str) 场景标题（如"洒满阳光的客厅"）
        - "duration": (int) 预计在此场景停留的总时长
        - "core_selling_point": (str) 此环境衬托的卖点
        - "scene_description": (str) 纯环境视觉描述（用于生成Concept Art）
        - "atmosphere": (str) 氛围关键词
        - "props": (str) 场景内的静态道具
        - "character": (str) "无" (强制留空，因为是环境图)
        """

        # 准备 Requests Payload
        url = "https://api.apiyi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json"
        }
        
        content_parts = [{"type": "text", "text": user_prompt}]
        
        if persona_image_path and os.path.exists(persona_image_path):
            try:
                with open(persona_image_path, "rb") as img_f:
                    b64_persona = base64.b64encode(img_f.read()).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_persona}"}
                    })
                print(f"   - 已加载人设参考图：{persona_image_path}")
            except Exception as e:
                print(f"⚠️ 读取人设图失败：{e}")

        if product_image_path and os.path.exists(product_image_path):
            try:
                with open(product_image_path, "rb") as img_f:
                    b64_product = base64.b64encode(img_f.read()).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_product}"}
                    })
                print(f"   - 已加载产品参考图：{product_image_path}")
            except Exception as e:
                print(f"⚠️ 读取产品图失败：{e}")

        payload = {
            "model": "gemini-2.5-pro", # 使用多模态模型
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content'].strip()
                # 清洗 JSON
                if content.startswith("```json"): content = content[7:]
                if content.endswith("```"): content = content[:-3]
                
                return json.loads(content.strip())
            else:
                 print(f"❌ VLM API Error: {response.text}")
                 raise Exception(f"HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ 场景设计失败，回退到默认模板。错误：{e}")
            # Fallback
            scene_titles = ["开篇空镜", "核心展示台", "结尾氛围空间"]
            scenes = []
            for i in range(SCENE_NUM):
                scenes.append({
                    "scene_id": f"scene_{i+1}",
                    "scene_title": scene_titles[i],
                    "duration": SCENE_DURATIONS[i],
                    "character": "无", # 环境图无人物
                    "core_selling_point": demand_info["core_selling_points"][0],
                    "scene_description": f"一个{style_detail['name']}风格的{scene_titles[i]}，空无一人，光影{style_detail['atmosphere']}",
                    "atmosphere": style_detail['atmosphere'],
                    "props": "展示台"
                })
            return scenes

    def _call_gen_image_api_for_scene(self, prompt: str, save_path: str, product_image_path: str = None, persona_image_path: str = None):
        """调用生图API生成单张场景图"""
        # 构造 Image Prompt
        parts = [{"text": prompt}]
        
        # 尝试读取产品图作为参考（可选，确保风格一致）
        if product_image_path and os.path.exists(product_image_path):
            try:
                with open(product_image_path, "rb") as f:
                    img_data = f.read()
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    # 简单判断，默认jpeg
                    mime_type = "image/jpeg" 
                    if product_image_path.lower().endswith(".png"): mime_type = "image/png"
                    parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
            except Exception:
                pass # 失败则仅用文本

        # 尝试读取人设图作为参考（确保风格一致）
        if persona_image_path and os.path.exists(persona_image_path):
            try:
                with open(persona_image_path, "rb") as f:
                    img_data = f.read()
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    # 简单判断，默认jpeg
                    mime_type = "image/jpeg" 
                    if persona_image_path.lower().endswith(".png"): mime_type = "image/png"
                    parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
            except Exception:
                pass # 失败则仅用文本
        
        url = "https://api.apiyi.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        api_key = os.getenv("API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}
            }
        }

        # 简单重试逻辑
        for _ in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300)
                if response.status_code == 200:
                    data = response.json()
                    if "candidates" in data and len(data["candidates"]) > 0:
                        img_b64 = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
                        with open(save_path, "wb") as f:
                            f.write(base64.b64decode(img_b64))
                        return
            except Exception as e:
                time.sleep(2)
        print(f"⚠️ 场景图生成失败：{prompt[:20]}...")

# ===================== 4. 分镜设计模块：每场景4个分镜+精细化描述 =====================
class StoryboardDesignModule(AdModule):
    module_name = "storyboard_designer"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：场景信息、故事线"""
        return all(key in context for key in ["scene_designer_info", "story_builder_info"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始设计广告分镜")
            scene_info = context["scene_designer_info"]
            story_info = context["story_builder_info"]
            style_detail = story_info["style_detail"]
            scenes = scene_info["scenes"]
            
            storyboard_all = {}
            for scene in scenes:
                scene_id = scene["scene_id"]
                scene_duration = scene["duration"]
                # 分镜时长分配（按场景时长平均，最后一个分镜补差值）
                sb_durations = [scene_duration // STORYBOARD_PER_SCENE] * STORYBOARD_PER_SCENE
                sb_durations[-1] += scene_duration % STORYBOARD_PER_SCENE
                
                # 检查是否有用户反馈
                user_feedback = context.get("user_feedback")

                # 调用LLM生成该场景下的所有分镜
                sb_contents = self._generate_storyboards_by_llm(scene, style_detail, sb_durations, user_feedback)
                
                storyboards = []
                for i, content in enumerate(sb_contents):
                    sb = {
                        "storyboard_id": f"{scene_id}_sb{i+1}",
                        "duration": sb_durations[i],
                        **content
                    }
                    storyboards.append(sb)
                    # 打印分镜详情
                    print(f"📸 {scene['scene_title']}-分镜{i+1}（{sb['duration']}s）：{sb['画面内容']}")
                    print(f"     镜头：{sb['镜头角度']} | 构图：{sb['构图方式']} | 色调：{sb['画面色调/光影']}\n")
                
                storyboard_all[scene_id] = {
                    "scene_title": scene["scene_title"],
                    "duration": scene_duration,
                    "storyboards": storyboards
                }
            
            storyboard_info = {
                "storyboard_per_scene": STORYBOARD_PER_SCENE,
                "total_storyboards": SCENE_NUM * STORYBOARD_PER_SCENE,
                "storyboards_by_scene": storyboard_all,
                "style_requirement": style_detail["name"]
            }
            print("✅ 全部分镜设计完成！（3场景×4分镜=12个分镜）")
            return {
                "status": "success",
                "result": storyboard_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"分镜设计失败：{str(e)}"
            }

    def _generate_storyboards_by_llm(self, scene: Dict[str, Any], style_detail: Dict[str, Any], durations: List[int], user_feedback: str = None) -> List[Dict[str, Any]]:
        """调用 VLM (Vision-Language Model) 为单个场景生成一组分镜"""
        print(f"🎨 正在调用大模型绘制场景【{scene['scene_title']}】的分镜...")
        
        system_prompt = "你是一位好莱坞顶级的广告分镜师，擅长将场景拆解为细腻的生图指令（Prompt）。"
        
        num_sbs = len(durations)
        mandatory_items_str = "、".join(STORYBOARD_MANDATORY_ITEMS)
        
        feedback_str = f"\n【用户修改意见】\n用户对之前的分镜提出了修改建议，请严格执行：{user_feedback}\n" if user_feedback else ""

        # 获取场景图路径
        scene_image_path = scene.get("scene_image_path", "")

        user_prompt = f"""
        【重要】参考提供的场景概念图（Concept Art），将该场景拆解为 {num_sbs} 个连续的分镜画面。
        确保所有分镜的背景环境与该概念图保持严格一致！
        {feedback_str}
        
        【场景信息】
        - 场景标题：{scene['scene_title']}
        - 场景描述：{scene['scene_description']}
        - 核心卖点：{scene['core_selling_point']}
        - 出场人物：{scene['character']}
        
        【视觉风格】
        - 风格：{style_detail['name']}
        - 整体氛围：{style_detail['atmosphere']}
        - 光影质感：{style_detail.get('lighting', '自然光')}
        - 典型表情：{style_detail.get('expression', '自然')}
        
        【要求】
        1. 输出 {num_sbs} 个分镜。
        2. 每个分镜必须包含以下字段：{mandatory_items_str}。
        3. "画面内容"必须是极具画面感的详细描述，包含人物状态、背景细节、光影效果，适合作为AI生图的Prompt。
        4. "此分镜的背景描述"必须基于输入的场景概念图。
        5. "人物动作/表情"需体现剧情递进。
        
        【返回格式】
        请直接返回一个JSON数组，包含 {num_sbs} 个对象，无需Markdown格式。每个对象结构如下：
        {{
            "画面内容": "...",
            "镜头角度": "...",
            "人物动作/表情": "...",
            "构图方式": "...",
            "画面色调/光影": "..."
        }}
        """

        # 准备 Requests Payload (类似 DemandInteractModule)
        url = "https://api.apiyi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json"
        }
        
        # 构建消息体（文本+可选图片）
        content_parts = [{"type": "text", "text": user_prompt}]
        
        if scene_image_path and os.path.exists(scene_image_path):
            with open(scene_image_path, "rb") as img_f:
                b64_scene = base64.b64encode(img_f.read()).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_scene}"}
                })
        
        payload = {
            "model": "gemini-2.5-pro", # 使用多模态模型
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_parts}
            ]
        }

        try:
            # 调用API
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                resp_json = response.json()
                content = resp_json['choices'][0]['message']['content'].strip()
                if content.startswith("```json"): content = content[7:]
                if content.endswith("```"): content = content[:-3]
                
                sbs = json.loads(content.strip())
                # 简单校验
                if len(sbs) != num_sbs:
                     if len(sbs) > num_sbs: sbs = sbs[:num_sbs]
                     else: sbs.extend([sbs[-1]] * (num_sbs - len(sbs)))
                return sbs
            else:
                raise Exception(f"HTTP Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"❌ 分镜生成失败，使用降级方案。错误：{e}")
            # Fallback
            return [{
                "画面内容": f"{scene['scene_description']}，分镜{i+1}",
                "镜头角度": "平视",
                "人物动作/表情": "自然展示",
                "构图方式": "中心构图",
                "画面色调/光影": style_detail.get("lighting", "自然光")
            } for i in range(num_sbs)]



# ===================== 5. 四宫格生图模块：调用API生成各场景四宫格【预留API】 =====================
class GridImageGenerateModule(AdModule):
    module_name = "grid_image_generator"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：分镜信息、风格信息、任务ID"""
        return all(key in context for key in ["storyboard_designer_info", "story_builder_info", "task_id"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始生成场景四宫格图片")
            storyboard_info = context["storyboard_designer_info"]
            story_info = context["story_builder_info"]
            style_detail = story_info["style_detail"]
            storyboards_by_scene = storyboard_info["storyboards_by_scene"]
            
            # 获取产品图路径 (从 demand_info)
            product_image_path = context.get("demand_interact_info", {}).get("product_image", "")
            
            task_dir = init_task_dir(task_id)
            grid_image_dir = os.path.join(task_dir, "grid_images")
            if not os.path.exists(grid_image_dir):
                os.makedirs(grid_image_dir)
            
            grid_image_result = {}
            # 遍历每个场景，生成四宫格（4个分镜→1张四宫格）
            for scene_id, scene_sb in storyboards_by_scene.items():
                scene_title = scene_sb["scene_title"]
                storyboards = scene_sb["storyboards"]
                # 提取4个分镜的生图描述（prompt）
                sb_prompts = [self._generate_sb_prompt(sb, style_detail) for sb in storyboards]
                print(f"正在生成【{scene_title}】四宫格...")
                
                # 调用生图API
                grid_image_path = os.path.join(grid_image_dir, f"{scene_id}_grid.png") # 强制保存为png
                self._call_gen_image_api_with_retry(sb_prompts, style_detail, grid_image_path, product_image_path)
                
                grid_image_result[scene_id] = {
                    "scene_title": scene_title,
                    "grid_image_path": grid_image_path,
                    "storyboard_mapping": {sb["storyboard_id"]: sb["画面内容"] for sb in storyboards},
                    "style": style_detail["name"],
                    "resolution": "2K"
                }
                print(f"✅ {scene_title}四宫格生成完成：{grid_image_path}")
            
            grid_image_info = {
                "grid_image_num": SCENE_NUM,
                "grid_image_dir": grid_image_dir,
                "grid_image_details": grid_image_result,
                "gen_image_style": style_detail["name"]
            }
            print("✅ 所有场景四宫格生成完成！（3张四宫格，共12个分镜）")
            return {
                "status": "success",
                "result": grid_image_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"四宫格生图失败：{str(e)}"
            }

    def _generate_sb_prompt(self, sb: Dict[str, Any], style_detail: Dict[str, Any]) -> str:
        """生成分镜生图prompt（贴合API格式，关键词前置）"""
        prompt = f"{sb['画面内容']}，{sb['镜头角度']}，{sb['构图方式']}，{sb['人物动作/表情']}，{sb['画面色调/光影']}"
        return prompt

    def _call_gen_image_api_with_retry(self, prompts: List[str], style_detail: Dict[str, Any], save_path: str, product_image_path: str = None):
        """带重试的生图API调用"""
        
        # 构造四宫格组合 Prompt
        combined_prompt = (
            "You are an expert storyboard artist. "
            f"Please generate a single 2x2 grid image (Four-panel storyboard, 16:9 aspect ratio total) based on the following 4 shot descriptions. "
            f"Style: {style_detail['name']}, {style_detail['atmosphere']}. "
            "Maintain strict character and product consistency across all 4 panels.\n\n"
            f"Panel 1 (Top-Left): {prompts[0]}\n"
            f"Panel 2 (Top-Right): {prompts[1]}\n"
            f"Panel 3 (Bottom-Left): {prompts[2]}\n"
            f"Panel 4 (Bottom-Right): {prompts[3]}"
        )
        
        # 读取产品图并转base64
        parts = [{"text": combined_prompt}]
        if product_image_path and os.path.exists(product_image_path):
            try:
                with open(product_image_path, "rb") as f:
                    img_data = f.read()
                    b64_data = base64.b64encode(img_data).decode('utf-8')
                    # 判断图片类型
                    mime_type = "image/jpeg"
                    if product_image_path.lower().endswith(".png"):
                        mime_type = "image/png"
                    elif product_image_path.lower().endswith(".webp"):
                        mime_type = "image/webp"

                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_data
                        }
                    })
            except Exception as read_err:
                print(f"⚠️ 读取产品图失败，将仅使用纯文本生成: {read_err}")
        else:
            print("⚠️ 未找到产品图或路径为空，使用纯文本生成。")

        # API配置
        url = "https://api.apiyi.com/v1beta/models/gemini-3-pro-image-preview:generateContent"
        api_key = os.getenv("API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"}
            }
        }

        for retry in range(API_RETRY_TIMES + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=300) # 5分钟超时
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

                resp_json = response.json()
                if "error" in resp_json:
                     raise Exception(f"API Error: {resp_json['error']}")

                # 提取图片
                if "candidates" in resp_json and len(resp_json["candidates"]) > 0:
                    cand = resp_json["candidates"][0]
                    if "content" in cand and "parts" in cand["content"]:
                         img_b64_res = cand["content"]["parts"][0]["inlineData"]["data"]
                         with open(save_path, "wb") as f:
                             f.write(base64.b64decode(img_b64_res))
                         return # 成功退出
                
                raise Exception("Response format unexpected or no image returned")

            except Exception as e:
                error_msg = str(e)
                if retry == API_RETRY_TIMES:
                    raise Exception(f"生图API调用失败（重试{API_RETRY_TIMES}次）：{error_msg}")
                print(f"生图API调用失败，{API_RETRY_INTERVAL}秒后重试（{retry+1}/{API_RETRY_TIMES}）... Error: {error_msg}")
                time.sleep(API_RETRY_INTERVAL)

# ===================== 6. 图像优化模块：调用API超分重绘【预留API】 =====================
class ImageOptimizeModule(AdModule):
    module_name = "image_optimizer"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：四宫格图片信息、任务ID"""
        return all(key in context for key in ["grid_image_generator_info", "task_id"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始优化四宫格图片（超分重绘）")
            grid_image_info = context["grid_image_generator_info"]
            grid_image_details = grid_image_info["grid_image_details"]
            task_dir = init_task_dir(task_id)
            hd_image_dir = os.path.join(task_dir, "hd_images")
            
            hd_image_result = {}
            # 遍历四宫格，超分优化
            for scene_id, grid_info in grid_image_details.items():
                grid_image_path = grid_info["grid_image_path"]
                hd_image_path = os.path.join(hd_image_dir, f"{scene_id}_hd.{IMAGE_FORMAT.lower()}")
                print(f"正在优化【{grid_info['scene_title']}】四宫格...")
                
                # 调用超分API（预留方法，自动重试）
                self._call_super_res_api_with_retry(grid_image_path, hd_image_path)
                
                hd_image_result[scene_id] = {
                    "original_grid_path": grid_image_path,
                    "hd_grid_path": hd_image_path,
                    "optimize_type": "超分重绘",
                    "resolution": IMAGE_RESOLUTION,
                    "style_consistency": grid_info["style"]
                }
                print(f"✅ {grid_info['scene_title']}四宫格超分完成：{hd_image_path}")
            
            hd_image_info = {
                "hd_image_dir": hd_image_dir,
                "hd_image_details": hd_image_result,
                "optimize_status": "全部完成"
            }
            print("✅ 所有四宫格图片超分优化完成！")
            return {
                "status": "success",
                "result": hd_image_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"图片优化失败：{str(e)}"
            }

    def _call_super_res_api_with_retry(self, original_path: str, save_path: str):
        """带重试的超分API调用（预留方法，替换为你的API代码即可）"""
        for retry in range(API_RETRY_TIMES + 1):
            try:
                # ========== 此处替换为你的超分API代码 ==========
                # 示例：模拟API调用，生成空文件（实际开发替换为真实API）
                open(save_path, "w").close()
                # ==============================================
                return
            except Exception as e:
                if retry == API_RETRY_TIMES:
                    raise Exception(f"超分API调用失败（重试{API_RETRY_TIMES}次）：{str(e)}")
                print(f"超分API调用失败，{API_RETRY_INTERVAL}秒后重试（{retry+1}/{API_RETRY_TIMES}）...")
                time.sleep(API_RETRY_INTERVAL)

# ===================== 7. 视频生成模块：调用API生成1min广告视频【预留API】 =====================
class VideoGenerateModule(AdModule):
    module_name = "video_generator"

    def validate_input(self, context: Dict[str, Any]) -> bool:
        """校验输入：超分图片信息、场景时长、任务ID"""
        return all(key in context for key in ["image_optimizer_info", "scene_designer_info", "task_id"])

    def run(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print_separator("开始生成1min广告视频")
            # hd_image_info = context["hd_image_info"] 
            hd_image_info = context["image_optimizer_info"]
            scene_info = context["scene_designer_info"]
            story_info = context["story_builder_info"] # 视频生成可能也需要故事风格信息
            hd_image_details = hd_image_info["hd_image_details"]
            scenes = scene_info["scenes"]
            task_dir = init_task_dir(task_id)
            video_dir = os.path.join(task_dir, "videos")
            video_path = os.path.join(video_dir, f"ad_final_{task_id}.{VIDEO_FORMAT.lower()}")
            
            # 提取视频生成所需参数：超分四宫格、场景时长、风格、总时长
            video_gen_params = {
                "hd_grid_images": [info["hd_grid_path"] for info in hd_image_details.values()],
                "scene_durations": [scene["duration"] for scene in scenes],
                "total_duration": AD_TOTAL_DURATION,
                "style": story_info["style_detail"]["name"],
                "resolution": VIDEO_RESOLUTION,
                "fps": VIDEO_FPS,
                "save_path": video_path
            }
            print(f"视频生成参数：{AD_TOTAL_DURATION}s | {VIDEO_RESOLUTION} | {VIDEO_FPS}帧 | {video_gen_params['style']}")
            
            # 调用视频生成API（预留方法，自动重试）
            self._call_gen_video_api_with_retry(video_gen_params)
            
            video_info = {
                "final_video_path": video_path,
                "video_format": VIDEO_FORMAT,
                "resolution": VIDEO_RESOLUTION,
                "fps": VIDEO_FPS,
                "total_duration": AD_TOTAL_DURATION,
                "scene_num": SCENE_NUM,
                "source_hd_images": [info["hd_grid_path"] for info in hd_image_details.values()],
                "generate_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            print(f"✅ 1min广告视频生成完成！最终文件：{video_path}")
            print_separator("广告生成全流程完成！")
            return {
                "status": "success",
                "result": video_info,
                "error": None
            }
        except Exception as e:
            return {
                "status": "failed",
                "result": {},
                "error": f"视频生成失败：{str(e)}"
            }

    def _call_gen_video_api_with_retry(self, video_params: Dict[str, Any]):
        """带重试的视频生成API调用（预留方法，替换为你的API代码即可）"""
        for retry in range(API_RETRY_TIMES + 1):
            try:
                # ========== 此处替换为你的视频生成API代码 ==========
                # 示例：模拟API调用，生成空文件（实际开发替换为真实API）
                open(video_params["save_path"], "w").close()
                # ==============================================
                return
            except Exception as e:
                if retry == API_RETRY_TIMES:
                    raise Exception(f"视频生成API调用失败（重试{API_RETRY_TIMES}次）：{str(e)}")
                print(f"视频生成API调用失败，{API_RETRY_INTERVAL}秒后重试（{retry+1}/{API_RETRY_TIMES}）...")
                time.sleep(API_RETRY_INTERVAL)