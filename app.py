# app.py — Qwen3 医学问答 Demo
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= 配置 =========
MODEL_PATH      = "./output/Qwen3-1.7B/checkpoint-1000"   # 权重目录
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS  = 1024
SYSTEM_MESSAGE  = "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。"

# ========= 加载模型 =========
print("[INFO] Loading model …")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto" if DEVICE == "cuda" else None,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
).eval()
print("[INFO] Model ready.")

# ========= 推理回调 =========
@torch.inference_mode()
def generate(user_msg: str, history: list[tuple[str, str]]):
    """
    Parameters
    ----------
    user_msg : 用户本次输入
    history  : [('上一条 user','上一条 assistant'), ...]  — ChatInterface 自动维护
    Returns
    -------
    str      : 最新助手回复（只需这一条）
    """
    # 1. 组装完整消息序列
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for u, a in history:
        messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_msg})

    # 2. 构造 prompt & 推理
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(DEVICE)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    reply = tokenizer.decode(
        gen_ids[0, inputs.input_ids.shape[1]:],  # 只取新增 token
        skip_special_tokens=True,
    ).strip()

    # 3. 仅返回助手回复
    return reply

# ========= UI =========
EXAMPLES = [
    "医生，我最近被诊断为糖尿病，应该怎样管理饮食中的碳水化合物？",
    "总是腰痛，怀疑肾结石，需要做哪些检查？",
    "儿童发烧 39 °C 以上，家长应如何处理？",
]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🩺 Qwen3 医学问答演示  
        基于 SFT 的 **Qwen3-1.7B**，支持带 Chain-of-Thought 的专业医学回答。  
        <span style="color:#d9534f"><strong>免责声明</strong>：仅供技术交流与学术研究，不可替代医疗诊断。</span>
        """,
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                placeholder="<div style='text-align:center;'><strong>请输入医学问题…</strong></div>"
            )
            gr.ChatInterface(
                fn=generate,
                chatbot=chatbot,
                examples=EXAMPLES,
                retry_btn="🔄 重新生成",
                undo_btn="↩️ 撤回上一条",
                clear_btn="🗑️ 清空对话",
            )
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## 使用说明")
            gr.Markdown(
                """
                1. 在左侧输入医学相关问题。\n
                2. 模型会进行内部思考并返回结构化答复。\n
                3. 点击「清空对话」可重新开始。\n
                若需修改推理参数，可直接编辑 `app.py`。
                """
            )

    gr.Markdown("--- © 2025 xuxufei12 — Powered by Gradio 4 & Transformers")

if __name__ == "__main__":
    demo.launch(server_port=6006, share=True)
