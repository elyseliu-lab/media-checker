# 媒体身份自动核查与资质审计系统

这是一个基于 Streamlit 和大模型的自动化媒体资质核查系统。能够快速比对国家互联网信息办公室的“稿源白名单”，并对未收录媒体进行 AI 深度资质核验。

## 本地运行

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 启动应用：
```bash
streamlit run app.py
```

## 核心文件说明
- `app.py`: 主程序代码。
- `central_media.csv`: 央媒名录数据。
- `official_media_whitelist_cleaned.csv`: 深度清洗后的地方官媒/新闻源名录。
- `requirements.txt`: 云端部署依赖文件。