# 🩺 Qwen3-Medical-SFT
📖 English version available: [README_EN.md](./README_EN.md)

基于 [Qwen3-1.7B](https://modelscope.cn/models/Qwen/Qwen3-1.7B) 大模型的医学对话全参数微调项目，支持 **带推理链条（Chain-of-Thought）** 的专业医疗问答能力，辅以 [SwanLab](https://swanlab.cn) 实现训练过程可视化追踪。

> ⚠️ 本项目仅供技术交流与学术研究，**不作为医疗建议或诊断依据**。

---

## 🔧 项目结构

| 文件/目录         | 功能描述                                                                 |
|------------------|--------------------------------------------------------------------------|
| `train.py`       | 使用 Modelscope + Transformers 对 Qwen3-1.7B 模型进行全参数微调训练        |
| `inference.py`   | 脚本式调用模型进行单轮推理测试                                            |
| `app.py`         | 基于 Gradio 搭建的 Web 医学问答交互界面                                   |
| `train_format.jsonl` | 模型训练用数据（question + think + answer 转换格式）                          |

---

## 🧪 使用方法

### 1. 安装依赖

```bash
pip install swanlab modelscope==1.22.0 "transformers>=4.50.0" datasets==3.2.0 accelerate pandas addict gradio==4.44.1
```

环境要求：
- Python ≥ 3.8
- GPU 显存建议 ≥ 32GB（若用全参微调）

---

### 2. 数据准备

本项目使用 [delicate_medical_r1_data](https://modelscope.cn/datasets/krisfu/delicate_medical_r1_data) 数据集，我已完成下载和划分，可直接使用。每条数据包含 question、think、answer 字段。

使用 `train.py` 中的 `dataset_jsonl_transfer` 方法自动转为微调格式：

```json
{
  "instruction": "你是一个医学专家...",
  "input": "儿童发烧 39°C 以上，家长应如何处理？",
  "output": "<think>...</think>\n答案内容..."
}
```

---

### 3. 开始训练

```bash
python train.py
```

- 模型自动下载至 `Qwen/Qwen3-1.7B/`
- 权重保存在 `./output/Qwen3-1.7B/checkpoint-*`
- 训练过程通过 SwanLab 可视化

---

### 4. 推理测试

```bash
python inference.py
```

---

### 5. 启动 Web Demo

```bash
python app.py
```

默认在浏览器打开 http://localhost:6006，支持用户输入医学问题进行问答。

---

## 📊 可视化结果

训练过程中 loss 曲线如下：

![图片说明](./train_loss_curve.png)


> 💡 本人实验中发现，loss 呈现**阶梯式下降趋势**，表现出**明显的过拟合现象**。由于数据集较小，全参数微调仅适合进行 **1 个 epoch 的训练**，否则会导致性能下降。

---

## 🤖 模型信息

- 模型名称：`Qwen/Qwen3-1.7B`
- 下载地址：https://modelscope.cn/models/Qwen/Qwen3-1.7B
- 微调方式：全参数微调
- 推理框架：Transformers + Gradio

---

## 📎 参考项目

- [Qwen3 模型](https://modelscope.cn/models/Qwen)
- [SwanLab 可视化工具](https://swanlab.cn)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

## 📄 License

MIT License