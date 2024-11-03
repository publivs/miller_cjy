# chrome_driver下载地址
http://chromedriver.storage.googleapis.com/index.html

# C++17 / C++20 配置
https://blog.csdn.net/qq_45859188/article/details/120678430  

#  mingw 安装
https://zhuanlan.zhihu.com/p/367010946

#  vscode c++ 环境配置
https://blog.csdn.net/weixin_44996090/article/details/104432593

# ---------------------------------- 分割线 -------------------------------------- # 
wsl_2 -- nvidia 环境配置
https://zhuanlan.zhihu.com/p/621142457

# cuda|cudnn
https://blog.csdn.net/anmin8888/article/details/127910084

# cuda 下载地址
https://developer.nvidia.com/cuda-toolkit-archive

# wsl_ubantu的地址
# installer_type 选择runfile(local)
https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0


# ---------------------------------- 分割线 ------------------------------------- # 
模型部署 vicuna-13b 

# 知乎老哥 1
https://zhuanlan.zhihu.com/p/624286959

# 知乎老哥 2
https://zhuanlan.zhihu.com/p/622020107

可以互相当参考,下载的那部分要替换LFS有坑,填上之后问题就不大了


# 模型合成
python -m fastchat.model.apply_delta --base  ./vicuna_model/pyllama_data/output/7B  --delta ./vicuna_model/vicuna-7b-delta-v1.1 --target ./vicuna_model/vicuna-7b-v1.1


# fine-tuning 相关的方案
https://zhuanlan.zhihu.com/p/630287397



# ----------------------- 分割线 ------------------------- #
# 本地client 直接联通测试
python -m fastchat.serve.cli --model-path vicuna_model/vicuna-7b-v1.1

# web部署流程
python3 -m fastchat.serve.controller 
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path vicuna_model/vicuna-7b-v1.1
python3 -m fastchat.serve.gradio_web_server --host localhost --port 8000

# API流式部署

# 某知乎老哥的git_hub地址,看下能不能调通
https://github.com/little51/FastChat 

# terminal 测试发送
curl http://localhost:8000/v1/chat/completions/stream  -H "Content-Type: application/json" -d '{"model": "vicuna-7b-v1.1","messages": [{"role": "user", "content": "请写一篇100字的日记"}]}'   

# Api_流式部署
pkill -9 -f fastchat  #  这里是清掉所有fastchat进程
python -u -m fastchat.serve.controller 
python -u -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path vicuna_model/vicuna-7b-v1.1 
python -u -m fastchat.serve.api_stream --host 0.0.0.0 --port 8000 

# 设置WSL2上的服务微调
# 代理配置
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=172.24.110.131 protocol=tcp

# 代理删除
netsh interface portproxy delete v4tov4 listenport=80 listenaddress=0.0.0.0



# ---------------------- 华丽的分割线 ----------------------- # 
# 微调
https://zhuanlan.zhihu.com/p/633469921