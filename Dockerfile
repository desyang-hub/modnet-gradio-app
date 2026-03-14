# 使用 python:3.9.12 作为基础镜像
FROM python:3.9.12

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive


# 设置工作目录
WORKDIR /app

# 复制源码和 CMakeLists.txt
COPY app.py .
COPY requirements.txt .
COPY modnet/ ./modnet/

# 安装环境
RUN python -m pip install 
# -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 运行项目
CMD ["python3", "-u", "app.py"]