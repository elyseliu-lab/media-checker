import streamlit as st
import pandas as pd
import os
import io
import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
import traceback
from openai import OpenAI

st.set_page_config(page_title="媒体身份核查平台", page_icon="📰", layout="wide")

# ==================== AI 配置侧边栏 ====================
st.sidebar.header("⚙️ AI 自动核查配置")
st.sidebar.markdown("由于深度核查需要从全网搜索信息并进行语义分析，建议配置兼容 OpenAI 格式的大模型 API (如 DeepSeek, 智谱, 阿里千问等)。")

ai_api_key = st.sidebar.text_input("API Key (必填)", type="password", placeholder="sk-...")
ai_api_base = st.sidebar.text_input("API Base URL", value="https://api.deepseek.com/v1")
ai_model_name = st.sidebar.text_input("Model Name", value="deepseek-chat")

# ==================== 爬虫辅助模块 ====================
@st.cache_data(ttl=3600)
def fetch_search_snippets(query):
    snippets = []
    
    # Baidu Search
    try:
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Connection': 'keep-alive',
        }
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = soup.find_all('div', class_=lambda x: x and 'result' in x and 'c-container' in x)
        for r in results[:5]:
            title = r.find('h3')
            title_text = title.get_text(strip=True) if title else ""
            
            content = r.find('span', class_=lambda x: x and 'content-right' in x)
            if not content:
                content = r.find('div', class_=lambda x: x and 'c-abstract' in x)
            if not content:
                content = r.find('div', class_=lambda x: x and 'c-row' in x)
            content_text = content.get_text(strip=True) if content else r.get_text(strip=True)
            
            if title_text and content_text:
                snippets.append(f"【百度】标题：{title_text}\n内容：{content_text}")
    except Exception as e:
        pass

    # Sogou Search
    try:
        url = f"https://www.sogou.com/web?query={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        results = soup.find_all('div', class_='vrwrap') or soup.find_all('div', class_='rb')
        for r in results[:5]:
            title = r.find('h3')
            title_text = title.get_text(strip=True) if title else ""
            content = r.find('div', class_='str_info') or r.find('div', class_='ft')
            content_text = content.get_text(strip=True) if content else r.get_text(strip=True)
            if title_text and content_text:
                snippets.append(f"【搜狗】标题：{title_text}\n内容：{content_text}")
    except Exception:
        pass
        
    return "\n---\n".join(snippets[:8])  # 返回最多8条

def call_ai_analysis(media_name, snippets):
    import httpx
    # 配置更长的超时时间
    http_client = httpx.Client(timeout=60.0)
    
    # 自动补全 http/https 协议，防止用户输入时不小心漏掉导致 httpx 报错 UnsupportedProtocol
    base_url_clean = ai_api_base.strip()
    if base_url_clean and not base_url_clean.startswith(('http://', 'https://')):
        base_url_clean = 'https://' + base_url_clean

    client = OpenAI(
        api_key=ai_api_key, 
        base_url=base_url_clean,
        http_client=http_client
    )
    system_prompt = "你是一个专业的媒体资质审核助手。请根据提供的搜索结果，分析该媒体的【主管主办单位】、【机构性质】和【许可证类型】。"
    user_prompt = f"""
    请根据以下搜索引擎提取的片段，分析媒体“{media_name}”的以下三个属性。如果片段中没有明确信息，请结合你本身的知识库进行合理推测（重点识别党媒/官媒属性），但如果完全未知请填“未知”。
    
    搜索片段：
    {snippets}
    
    你需要提取以下信息并严格按 JSON 格式返回，不要输出多余解释内容：
    {{
        "sponsor": "主管或主办单位名称（如：中共xx市委宣传部/天津日报社/腾讯科技等。如果没有找到则填'未知'）",
        "shareholder_type": "必须且只能在以下三个选项中选择一个：['机关/事业单位/100%国有全资', '国有控股', '民营/外资/混合所有制']。（判定规则补充：如果是多个国有股东叠加出资，仍然算作 100% 国有全资）",
        "license_type": "必须且只能在以下四个选项中选择一个：['采编发布', '转载服务', '平台传播', '无/不确定']",
        "reasoning": "你的分析过程（简短说明你的判断依据和找到的信息来源）"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=ai_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        
        # 更强壮的 JSON 提取逻辑（剔除大模型啰嗦的前后缀文本）
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            clean_content = content[start:end+1]
        else:
            clean_content = content
            
        return json.loads(clean_content)
        
    except json.JSONDecodeError as je:
        raise Exception(f"模型返回的不是有效的 JSON 格式: {je}\n原始返回内容: {content}")
    except Exception as e:
        raise Exception(f"接口请求失败或拒绝访问 ({type(e).__name__}): {str(e)}")

# 全局变量
CENTRAL_CSV = "central_media.csv"
OFFICIAL_CSV = "official_media_whitelist_cleaned.csv"

@st.cache_data(ttl=60)
def load_data():
    central_df, official_df = pd.DataFrame(), pd.DataFrame()
    
    if os.path.exists(CENTRAL_CSV):
        central_df = pd.read_csv(CENTRAL_CSV).fillna("")
    else:
        st.warning(f"⚠️ 未找到 {CENTRAL_CSV}")

    if os.path.exists(OFFICIAL_CSV):
        official_df = pd.read_csv(OFFICIAL_CSV).fillna("")
    else:
        st.warning(f"⚠️ 未找到 {OFFICIAL_CSV}")
        
    return central_df, official_df

central_df, official_df = load_data()

# ==================== 辅助判断逻辑 ====================

def check_advanced_criteria(name, sponsor, shareholder_type, license_type):
    """
    根据用户提供的高级规则进行判定
    
    参数:
    name: 媒体名称
    sponsor: 主管主办单位 (如: 中共北京市委宣传部, 清华大学, 腾讯科技)
    shareholder_type: 股东/机构性质 (选项: 机关/事业单位/100%国有全资, 国有控股, 民营/外资/个人)
    license_type: 许可证类型 (选项: 采编发布, 转载服务, 平台传播, 无/不确定)
    """
    
    reasons = []
    is_official = False
    
    # 规则1: 主管单位级别判定
    valid_sponsors = ['中共', '委员会', '宣传部', '网信办', '广播电视总台', '人民政府', '报业集团', '融媒体中心', '广播电视台']
    invalid_sponsors = ['大学', '学院', '学校', '公司', '企业', '协会', '学会', '联合会', '商会']
    
    sponsor_valid = False
    if sponsor:
        if any(v in sponsor for v in valid_sponsors):
            sponsor_valid = True
            reasons.append(f"✅ 主管单位[{sponsor}]属于党政/宣传/党媒体系")
        elif any(i in sponsor for i in invalid_sponsors):
             # 特殊豁免: 教育部/党媒主办的教育类
            if "教育" in name and ("教育部" in sponsor or "党报" in sponsor):
                sponsor_valid = True
                reasons.append(f"✅ 特殊豁免: 教育部/党媒主管教育类媒体[{sponsor}]")
            else:
                reasons.append(f"❌ 主管单位[{sponsor}]属于高校/企业/社团，非党政机关")
        else:
            reasons.append(f"⚠️ 主管单位[{sponsor}]性质需人工复核")
    
    # 规则2: 许可证判定
    license_valid = False
    if license_type == "采编发布":
        license_valid = True
        reasons.append("✅ 持有《互联网新闻信息采编发布服务许可证》")
    elif license_type in ["转载服务", "平台传播"]:
        reasons.append(f"❌ 持有[{license_type}]许可证，无采编资质")
    else:
        reasons.append("⚠️ 未明确标注采编发布资质")
        
    # 规则3: 机构性质判定
    nature_valid = False
    if shareholder_type == "机关/事业单位/100%国有全资":
        nature_valid = True
        reasons.append("✅ 机构性质为机关/事业单位或100%国有全资 (多个国有股东叠加视为100%国有)")
    elif shareholder_type == "国有控股":
        reasons.append("❌ 机构性质为国有控股(非100%全资)，属于市场化媒体")
    else:
        reasons.append("❌ 机构性质为民营/外资/混合所有制")

    # 综合判定公式
    # 官媒 = (主管单位合格) + (采编发布许可) + (事业单位 OR 100%国有全资)
    if sponsor_valid and license_valid and nature_valid:
        is_official = True
        final_result = "✅ 符合【非名单内官媒】认定标准"
    else:
        final_result = "❌ 不符合【非名单内官媒】认定标准"
        
    return is_official, final_result, {
        "sponsor_check": sponsor_valid,
        "sponsor_msg": reasons[0] if len(reasons) > 0 else "未知",
        "license_check": license_valid,
        "license_msg": reasons[1] if len(reasons) > 1 else "未知",
        "nature_check": nature_valid,
        "nature_msg": reasons[2] if len(reasons) > 2 else "未知"
    }

# ==================== 核心核查逻辑 ====================
def check_media(query):
    result = {
        "查询名称": query,
        "核查结果": "未命中白名单",
        "媒体层级": "未知层级",
        "匹配依据": "",
        "详细信息": "",
        "是否被系统拦截": False,
        "拦截理由": ""
    }
    
    if not query:
        return result
        
    global central_df, official_df
    central_df, official_df = load_data()

    # 1. 查央媒
    if not central_df.empty:
        mask = central_df.apply(lambda x: x.astype(str).str.contains(query, case=False, na=False)).any(axis=1)
        res = central_df[mask]
        if not res.empty:
            result["核查结果"] = "✅ 命中"
            result["媒体层级"] = "🇨🇳 央媒"
            result["匹配依据"] = "《中央新闻单位及相关网站名单》"
            details = []
            for _, row in res.iterrows():
                details.append(f"[{row.get('媒体名称', '未知')}] 主管: {row.get('主管单位', '未知')}")
            result["详细信息"] = " | ".join(details)
            return result

    # 2. 查地方官媒
    if not official_df.empty:
        mask = official_df['媒体名称'].astype(str).str.contains(query, case=False, na=False)
        res = official_df[mask]
        
        # 优先检查是否有完全精确匹配的名称，如果有则以精确匹配为准
        if not res.empty:
            exact_mask = res['媒体名称'].astype(str).str.lower() == query.lower()
            if exact_mask.any():
                res = res[exact_mask]
                
        if not res.empty:
            first_row = res.iloc[0]
            is_official = str(first_row.get("是否官媒", "是")).strip()
            
            if is_official == "否":
                result["核查结果"] = "❌ 已拦截"
                result["媒体层级"] = "非官方媒体"
                result["匹配依据"] = "白名单深度核查不合格"
                result["拦截理由"] = str(first_row.get("判断理由", "不符合官方标准"))
                result["是否被系统拦截"] = True
                return result
            else:
                result["核查结果"] = "✅ 命中"
                result["媒体层级"] = "🏛️ 官方媒体"
                result["匹配依据"] = str(first_row.get("备注", "网信办官媒白名单"))
                details = []
                for _, row in res.iterrows():
                    table = row.get("所属表名", "")
                    col = row.get("所属列名", "")
                    details.append(f"[{row['媒体名称']}] 归属: {table} -> {col}")
                result["详细信息"] = " | ".join(details)
                return result

    return result

# ==================== 页面布局 ====================
st.title("📰 媒体身份核查系统 (智能版)")
st.markdown("基于最新版《互联网新闻信息稿源单位名单》及央媒数据的自动化核查平台，支持 AI 深度研判。")

tab1, tab2 = st.tabs(["🔍 智能身份核查", "📁 批量文件核查"])

# ----------------- Tab 1: 智能核查 (合并原单条+高级) -----------------
with tab1:
    st.markdown("### 🤖 全能核查：优先匹配白名单，未命中则自动进行 AI 深度资质审计")
    
    with st.container(border=True):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1:
            query_name = st.text_input("媒体名称 (必填)", placeholder="例如：津云、新华网、XX日报...", key="smart_query_name")
        with col_in2:
            query_sponsor = st.text_input("主管单位 (选填)", placeholder="辅助 AI 搜索与判定", key="smart_query_sponsor")
            
        btn_check = st.button("🚀 开始核查", type="primary", use_container_width=True)

    if btn_check:
        if not query_name:
            st.warning("⚠️ 请先输入要核查的媒体名称！")
        else:
            query_name = query_name.strip()
            # Phase 1: 白名单快速核查
            with st.spinner("正在检索国家互联网新闻信息稿源白名单..."):
                whitelist_res = check_media(query_name)
            
            if whitelist_res["核查结果"] == "✅ 命中":
                # 命中白名单，直接显示结果
                st.success(f"## ✅ 权威收录：'{query_name}' 在册")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("媒体层级", whitelist_res["媒体层级"])
                m2.metric("匹配依据", whitelist_res["匹配依据"].split("（")[0])
                m3.metric("核查结果", whitelist_res["核查结果"])
                
                st.info(f"📝 **具体收录详情：**\n\n{whitelist_res['详细信息']}")
                st.balloons()
            
            elif whitelist_res.get("是否被系统拦截", False):
                st.error(f"## ❌ 拦截：'{query_name}' 虽在原始名单中，但经过深度清洗已确认为非官媒")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("媒体层级", "非官方媒体")
                m2.metric("拦截依据", "白名单深度核查不合格")
                m3.metric("核查结果", "❌ 已拦截")
                
                st.error(f"**拦截理由**：\n\n{whitelist_res['拦截理由']}")
            
            else:
                # 未命中白名单，进入 Phase 2
                st.warning(f"⚠️ 提示：在《互联网新闻信息稿源单位名单》中未找到 '{query_name}'。")
                
                if not ai_api_key:
                    st.error("❌ 无法进行深度研判：请在左侧边栏配置 AI API Key 以启用【非名单内官媒】的高级核查功能。")
                    st.caption("提示：许多官方新媒体（如地方党媒下属账号）可能不在白名单主表中，但符合官媒定义。配置 AI 后可自动识别。")
                else:
                    st.markdown("---")
                    st.markdown("#### 🔄 启动 AI 深度资质核查流程")
                    
                    # 使用 st.status 显示进度条和步骤
                    with st.status("🚀 正在进行 AI 智能研判...", expanded=True) as status:
                        
                        progress_bar = st.progress(0, text="初始化...")
                        
                        # Step 1: Search
                        progress_bar.progress(10, text="🌐 步骤 1/3: 正在全网检索媒体背景与资质信息...")
                        st.write("正在搜索相关公开信息...")
                        search_term = f"{query_name} 主管单位 机构性质 互联网新闻信息服务许可证"
                        if query_sponsor:
                            search_term += f" {query_sponsor}"
                            
                        snippets = fetch_search_snippets(search_term)
                        if not snippets:
                            st.write("⚠️ 搜索引擎未返回有效内容，将依赖模型内部知识...")
                        else:
                            st.write(f"✅ 已抓取 {len(snippets.split('---'))} 条相关公开信息。")
                        
                        # Step 2: AI Analysis
                        progress_bar.progress(50, text="🧠 步骤 2/3: 大模型正在分析实体特征 (主管单位/股权/牌照)...")
                        st.write("请求大模型进行语义解析...")
                        try:
                            ai_res = call_ai_analysis(query_name, snippets)
                            
                            # 提取字段
                            r_sponsor = ai_res.get("sponsor", "未知")
                            r_shareholder = ai_res.get("shareholder_type", "未知")
                            r_license = ai_res.get("license_type", "无/不确定")
                            r_reasoning = ai_res.get("reasoning", "暂无推理详情")

                            # 如果 AI 没查到主管单位但用户填了，优先使用用户填写的（或者做个提示）
                            if r_sponsor in ["未知", "无", ""] and query_sponsor:
                                r_sponsor = query_sponsor
                                st.write(f"ℹ️ 使用用户提供的主管单位: {query_sponsor}")
                            
                            st.write("✅ 特征提取完成。")
                            
                            # Step 3: Rule Check & Detailed Report
                            progress_bar.progress(90, text="⚖️ 步骤 3/3: 生成深度核查报告...")
                            st.write("正在生成逐项判定报告...")
                            
                            is_off, final_res, audit_dict = check_advanced_criteria(query_name, r_sponsor, r_shareholder, r_license)
                            
                            progress_bar.progress(100, text="✅ 核查完成")
                            status.update(label="✅ 核查完成", state="complete", expanded=True)
                            
                            st.divider()
                            st.markdown("### 📊 深度核查结论报告")
                            
                            if is_off:
                                st.success(f"## {final_res}")
                                st.balloons()
                            else:
                                st.error(f"## {final_res}")

                            st.info(f"💡 **AI 综合研判分析：**\n\n{r_reasoning}")

                            st.markdown("#### 🛠️ 逐项判定细节 (基于判定规则)")
                            
                            # 1. 主管单位核查
                            st.markdown("##### 1️⃣ 主管/主办单位性质核查")
                            if audit_dict["sponsor_check"]:
                                st.success(f"**通过**：{audit_dict['sponsor_msg']}")
                            else:
                                st.error(f"**未通过**：{audit_dict['sponsor_msg']}")

                            # 2. 股权/资金来源核查
                            st.markdown("##### 2️⃣ 资本属性与股权结构 (是否国资100%/事业单位)")
                            if audit_dict["nature_check"]:
                                st.success(f"**通过**：{audit_dict['nature_msg']}")
                            else:
                                st.error(f"**未通过**：{audit_dict['nature_msg']}")

                            # 3. 许可证核查
                            st.markdown("##### 3️⃣ 互联网新闻信息服务许可 (采编发布资质)")
                            if audit_dict["license_check"]:
                                st.success(f"**通过**：{audit_dict['license_msg']}")
                            else:
                                st.error(f"**未通过**：{audit_dict['license_msg']}")

                            st.caption("注：以上结论基于公开互联网信息检索 + AI 语义分析生成，仅供参考。建议通过国家企业信用信息公示系统或国家网信办官网进行最终人工确认。")

                        except Exception as e:
                            status.update(label="❌ 核查过程中断", state="error")
                            st.error(f"AI 服务调用失败: {e}")
                            with st.expander("👉 点击展开查看详细错误日志"):
                                st.code(traceback.format_exc())

# ----------------- Tab 2: 批量核查 -----------------
with tab2:
    st.markdown("### 📁 批量文件核查与清洗")
    
    st.info("💡 提示：此处使用您配置的 AI API Key 进行批量处理。处理 1000+ 条数据可能需要较长时间及相应的 Token 消耗。")
    
    # Mode Selection
    batch_mode = st.radio("选择操作模式:", ["📂 上传新文件核查", "🔄 现有白名单再清洗 (Official Whitelist Re-check)"])
    
    if batch_mode == "📂 上传新文件核查":
        uploaded_file = st.file_uploader("请上传 Excel 或 CSV 文件", type=['xlsx', 'xls', 'csv'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                    
                st.success(f"成功加载文件：{uploaded_file.name}，共 {len(df_upload)} 行。")
                media_col = st.selectbox("👉 请选择包含【媒体名称】的列:", df_upload.columns)
                
                if st.button("🚀 开始批量核查", type="primary"):
                    if not ai_api_key:
                         st.error("❌ 请先配置 AI API Key")
                    else:
                        with st.status("正在进行批量 AI 核查...", expanded=True) as status:
                            results = []
                            progress_bar = st.progress(0)
                            total = len(df_upload)
                            
                            for i, name in enumerate(df_upload[media_col]):
                                # Update progress
                                progress_bar.progress((i + 1) / total, text=f"正在核查 ({i+1}/{total}): {name}")
                                
                                media_name = str(name).strip() if pd.notna(name) else ""
                                
                                # Search & Analyze
                                search_term = f"{media_name} 主管单位 机构性质"
                                snippets = fetch_search_snippets(search_term)
                                try:
                                    ai_res = call_ai_analysis(media_name, snippets)
                                    # Check rules
                                    is_off, final_res, audit = check_advanced_criteria(
                                        media_name, 
                                        ai_res.get("sponsor", ""), 
                                        ai_res.get("shareholder_type", ""), 
                                        ai_res.get("license_type", "")
                                    )
                                    
                                    results.append({
                                        "媒体名称": media_name,
                                        "核查结果": "✅ 官媒" if is_off else "❌ 非官媒",
                                        "判定理由": final_res,
                                        "主管单位": ai_res.get("sponsor", ""),
                                        "机构性质": ai_res.get("shareholder_type", ""),
                                        "AI分析": ai_res.get("reasoning", "")
                                    })
                                except Exception as e:
                                    results.append({
                                        "媒体名称": media_name,
                                        "核查结果": "⚠️ 出错",
                                        "判定理由": f"AI 服务异常: {str(e)}"
                                    })
                            
                            status.update(label="✅ 批量核查完成", state="complete")
                            
                            res_df = pd.DataFrame(results)
                            final_df = pd.concat([df_upload.reset_index(drop=True), res_df], axis=1)
                            st.dataframe(final_df)
                            
                            # Download
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                final_df.to_excel(writer, index=False)
                            st.download_button("⬇️ 下载核查结果", output.getvalue(), "batch_result.xlsx")
                            
            except Exception as e:
                st.error(f"文件处理失败: {e}")

    elif batch_mode == "🔄 现有白名单再清洗 (Official Whitelist Re-check)":
        st.markdown(f"当前系统内置白名单文件：`{OFFICIAL_CSV}`")
        if not os.path.exists(OFFICIAL_CSV):
            st.error("❌ 未找到白名单文件！")
        else:
            df_official = pd.read_csv(OFFICIAL_CSV)
            st.write(f"共加载 {len(df_official)} 条数据。")
            
            col_limit, col_start = st.columns(2)
            check_start = col_start.number_input("起始行 (从第几行开始)", min_value=0, value=0, step=1)
            check_limit = col_limit.number_input("核查数量 (0 表示全部)", min_value=0, value=5, step=1)
            
            if st.button("🚀 开始白名单深度清洗", type="primary"):
                if not ai_api_key:
                    st.error("❌ 请先配置 AI API Key")
                else:
                    # Slice dataframe
                    if check_limit == 0:
                        target_df = df_official.iloc[check_start:]
                    else:
                        target_df = df_official.iloc[check_start : check_start + check_limit]
                    
                    st.info(f"本次将核查 {len(target_df)} 条数据 (行 {check_start} 到 {check_start + len(target_df)})")
                    
                    results = []
                    progress_bar = st.progress(0, text="初始化批量任务...")
                    status_text = st.empty()
                    
                    total = len(target_df)
                    
                    for i, (idx, row) in enumerate(target_df.iterrows()):
                        name = str(row['媒体名称']).strip()
                        pct = (i + 1) / total
                        progress_bar.progress(pct, text=f"({i+1}/{total}) 正在深度审计: {name}")
                        
                        try:
                            # 1. Search
                            snippets = fetch_search_snippets(f"{name} 主管单位 机构性质 互联网新闻信息服务许可证")
                            
                            # 2. AI Analysis
                            ai_res = call_ai_analysis(name, snippets)
                            r_sponsor = ai_res.get("sponsor", "未知")
                            r_shareholder = ai_res.get("shareholder_type", "未知")
                            r_license = ai_res.get("license_type", "无/不确定")
                            r_reasoning = ai_res.get("reasoning", "")
                            
                            # 3. Rule Check
                            is_off, final_res, audit = check_advanced_criteria(name, r_sponsor, r_shareholder, r_license)
                            
                            res_item = {
                                "媒体名称": name,
                                "是否官媒": "是" if is_off else "否",
                                "判断理由": final_res,
                                "主管单位": r_sponsor,
                                "机构性质": r_shareholder,
                                "许可证": r_license,
                                "AI分析过程": r_reasoning[:200] + "..." if len(r_reasoning) > 200 else r_reasoning
                            }
                            results.append(res_item)
                            
                        except Exception as e:
                            results.append({
                                "媒体名称": name,
                                "是否官媒": "错误",
                                "判断理由": f"API调用失败: {str(e)}"
                            })
                            
                    st.success("✅ 批量清洗完成！")
                    
                    # Merge results
                    res_df = pd.DataFrame(results)
                    # Align with original DF structure for display
                    final_display_df = res_df
                    
                    st.dataframe(final_display_df)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        final_display_df.to_excel(writer, index=False, sheet_name='清洗结果')
                    
                    st.download_button(
                        label="⬇️ 下载清洗后的白名单结果",
                        data=output.getvalue(),
                        file_name=f"whitelist_recheck_result_{check_start}_{check_start+len(target_df)}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

st.divider()
st.caption(f"当前系统数据源：央媒记录 ({len(central_df)} 条) | 官媒记录 ({len(official_df)} 条)")