## 1. 环境构建

- Clone项目后，首先创建一些文件夹

  ```shell
  mkdir data
  mkdir result
  mkdir model
  ```

- 创建虚拟环境

   ```shell
    python -m venv ./.venv
   ```

- 激活虚拟环境

  ```shell
  source ./.venv/bin/activate
  ```

- 安装依赖

  ```shell
  pip install -r requirements.txt
  ```

## 2. 下载基础模型

- 使用`download_tools`中的脚本下载`sentence-bert`模型

  ```shell
  python ./hf_download.py --model sentence-transformers/all-MiniLM-L6-v1 --save_dir ../model/
  ```

