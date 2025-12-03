import websocket
import uuid
import json
import urllib.request
import urllib.parse
import time
import os

# 配置
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())
workflow_file = "workflow_api.json"

def queue_prompt(prompt_workflow):
    """将工作流提交到队列"""
    try:
        p = {"prompt": prompt_workflow, "client_id": client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{server_address}/prompt", data=data)
        response = urllib.request.urlopen(req)
        print("任务已成功提交！")
        return json.loads(response.read())
    except Exception as e:
        print(f"提交任务失败: {e}")
        return None

def get_history(prompt_id):
    """获取指定ID的历史记录"""
    with urllib.request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())

def main():
    """主执行函数"""
    print("正在加载工作流...")
    if not os.path.exists(workflow_file):
        print(f"错误: 工作流文件 '{workflow_file}' 不存在。")
        return

    with open(workflow_file, 'r') as f:
        prompt = json.load(f)["prompt"]

    # 提交工作流
    prompt_id = queue_prompt(prompt).get('prompt_id')
    if not prompt_id:
        return

    print(f"已获取任务ID: {prompt_id}")

    # 等待执行完成
    ws = websocket.WebSocket()
    try:
        ws.connect(f"ws://{server_address}/ws?clientId={client_id}")
        print("WebSocket 连接成功，等待执行结果...")

        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        print("\n任务执行完毕！")
                        break # 执行完成
                    else:
                        print(f"正在执行节点: {data.get('node', 'N/A')}", end='\r')
            time.sleep(0.1)
    except Exception as e:
        print(f"WebSocket 出现错误: {e}")
    finally:
        ws.close()
        print("WebSocket 连接已关闭。")

if __name__ == "__main__":
    main()