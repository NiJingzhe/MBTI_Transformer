# %%
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import tensorboard as tb

# %%
check_list = {"E": 0, "I": 1, "N": 0, "S": 1, "F": 0, "T": 1, "J": 0, "P": 1}


class MBTIDataset(Dataset):
    def __init__(self, folder_path):
        print("开始读取数据文件...")
        # 读取文件夹下所有csv文件并合并
        import os

        csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
        data_frames = []
        for i, csv_file in enumerate(csv_files):
            print(f"正在读取第{i+1}/{len(csv_files)}个文件: {csv_file}")
            csv_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(csv_path)
            data_frames.append(df)
        data_sheet = pd.concat(data_frames, ignore_index=True)
        print(f"成功读取{len(csv_files)}个文件")

        print("开始处理帖子数据...")
        self.posts = []
        total = len(data_sheet["posts"])
        for i, posts_50 in enumerate(data_sheet["posts"]):
            if i % 10000 == 0:
                print(f"处理进度: {i}/{total}")
            posts_50_list = posts_50.split("|||")
            if len(posts_50_list) < 50:
                posts_50_list += [""] * (50 - len(posts_50_list))
            if len(posts_50_list) > 50:
                posts_50_list = posts_50_list[:50]
            self.posts.append(posts_50_list)
        print("帖子数据处理完成")

        print("开始处理标签数据...")
        self.mbti_labels = []
        self.current_dim = "EI"  # 默认使用EI维度
        for i, mbti_type in enumerate(data_sheet["type"]):
            if i % 10000 == 0:
                print(f"标签处理进度: {i}/{len(data_sheet['type'])}")
            # 存储所有四个维度的标签
            labels = {
                "EI": check_list[mbti_type[0]],
                "NS": check_list[mbti_type[1]], 
                "FT": check_list[mbti_type[2]],
                "JP": check_list[mbti_type[3]]
            }
            self.mbti_labels.append(labels)
        print("标签处理完成")
        print(f"数据集初始化完成,共{len(self.posts)}条数据")

    def set_dim(self, dim):
        """设置当前使用的MBTI维度"""
        assert dim in ["EI", "NS", "FT", "JP"]
        self.current_dim = dim
        print(f"当前使用维度: {dim}")

    def __len__(self):
        return len(self.posts)

    def __getitem__(self, idx):
        label = torch.tensor(self.mbti_labels[idx][self.current_dim], dtype=torch.long)
        return {"posts": self.posts[idx], "type": label}


# %% [markdown]
# #### 创建嵌入模型和分类模型

# %%
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class MBTIClassifier(nn.Module):
    def __init__(
        self,
        device=torch.device("cuda"),
        model_path="../model/models--sentence-transformers--all-MiniLM-L6-v2",
    ):
        super(MBTIClassifier, self).__init__()

        self.device = device
        self.embedding_dim = 384
        self.num_classes = 2  # 修改为二分类

        self.embedding_model = SentenceTransformer(model_path, device=self.device)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=768, dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=4, norm=self.layer_norm  
        )
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim, nhead=8, dim_feedforward=768, dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.transformer_decoder_layer, num_layers=4, norm=self.layer_norm
        )
        self.ffn = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.num_classes),
        )

        self.to(self.device)

    def forward(self, batch_sentences_list):
        # 预处理每个句子
        preprocessed_batch = []
        for sentences_list in batch_sentences_list:
            preprocessed_list = []
            for sentence in sentences_list:
                sentence = sentence.lower()
                sentence = sentence.replace("  ", " ")
                preprocessed_list.append(sentence)
            preprocessed_batch.append(preprocessed_list)

        # 扁平化以适应SentenceTransformer
        flattened_sentences = [
            sentence
            for sentences_list in preprocessed_batch
            for sentence in sentences_list
        ]

        # 编码所有句子
        embeddings = self.embedding_model.encode(
            flattened_sentences, convert_to_tensor=True, device=self.device
        )

        # 重塑维度为 (batch_size, sequence_length, embedding_dim)
        batch_size = len(preprocessed_batch)
        sequence_length = len(preprocessed_batch[0])  # 假设所有列表长度相同
        embeddings = embeddings.view(batch_size, sequence_length, -1)
        embeddings = embeddings.permute(
            1, 0, 2
        )  # (sequence_length, batch_size, embedding_dim)

        # 
        embeddings = self.layer_norm(embeddings)

        # Transformer Encoder
        transformer_encoder_output = self.transformer_encoder(embeddings)

        # 创建并应用Decoder输入
        mbti_tensor = torch.ones(1, batch_size, self.embedding_dim).to(
            self.device
        )  # 单个token

        # Transformer Decoder
        transformer_output = self.transformer_decoder(
            mbti_tensor, transformer_encoder_output
        )
        transformer_output = transformer_output.squeeze(
            0
        )  # (batch_size, embedding_dim)

        # 前馈网络输出
        output = self.ffn(transformer_output)  # (batch_size, num_classes)
        return output

    def train(self, mode=True):
        """
        Sets the module in training mode.
        """
        super(MBTIClassifier, self).train(mode)
        self.embedding_model.train()
        self.layer_norm.train()
        self.transformer_encoder.train()
        self.transformer_decoder.train()
        self.ffn.train()

    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        super(MBTIClassifier, self).eval()
        self.embedding_model.eval()
        self.layer_norm.eval()
        self.transformer_encoder.eval()
        self.transformer_decoder.eval()
        self.ffn.eval()

    def model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

# %%
import itertools
import shutil
import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )


def train_epoch(model, dataloader, criterion, optimizer, device, writer, global_step):
    model.train()
    epoch_loss = 0.0
    for i, batch in enumerate(dataloader):

        # posts is a list of 50 strings for each of the 16 samples
        # We need to convert it to (16, 50) to fit the model
        posts = batch["posts"]  # Shape: (50, 16)
        real_batch_size = len(posts[0])
        posts_shape_fix = [[] for _ in range(real_batch_size)]
        for i_, j_ in itertools.product(range(real_batch_size), range(50)):
            try:
                posts_shape_fix[i_].append(posts[j_][i_])
            except Exception as e:
                print(f"Error: {e}, current i is {i_}, current j is {j_}")
                print(f"current posts[j] is {posts[j_]}")
                exit(0)
        posts = posts_shape_fix

        labels = batch["type"].to(device)

        optimizer.zero_grad()
        outputs = model(posts)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar("Loss/train", loss.item(), global_step)
        global_step += 1

        if i % 10 == 0:
            print(f"Iter {i}, train loss: {loss.item()}")

    avg_loss = epoch_loss / len(dataloader)
    return avg_loss, global_step


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for batch in dataloader:
            # posts is a list of 50 strings for each of the 16 samples
            # We need to convert it to (16, 50) to fit the model
            posts = batch["posts"]  # Shape: (50, 16)
            real_batch_size = len(posts[0])
            posts_shape_fix = [[] for _ in range(real_batch_size)]
            for i, j in itertools.product(range(real_batch_size), range(50)):
                try:
                    posts_shape_fix[i].append(posts[j][i])
                except Exception as e:
                    print(f"Error: {e}, current i is {i}, current j is {j}")
                    print(f"current posts[j] is {posts[j]}")
                    exit(0)
            posts = posts_shape_fix
            labels = batch["type"].to(device)

            outputs = model(posts)
            loss = criterion(outputs, labels)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
            val_loss += loss.item()
            val_acc += accuracy.item()

    return val_loss / len(dataloader), val_acc / len(dataloader)


def train(
    dataset,
    model,
    device,
    batch_size=32,
    epochs=10,
    lr=2e-5,
    patience=3,
    scheduler_type="linear",
    model_save_name="best_model.pth",
):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_dataset, val_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.01)

    if scheduler_type == "linear":
        scheduler = LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
        )
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 清空之前的日志目录
    log_dir = "runs/mbti_classification"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    global_step = 0
    best_val_acc = 0.0
    epochs_no_improve = 0

    # 创建DataFrame来存储训练过程中的指标
    metrics_df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_acc"])

    for epoch in range(epochs):
        train_loss, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device, writer, global_step
        )
        print("validating...")
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # 将当前epoch的指标添加到DataFrame
        metrics_df = pd.concat(
            [
                metrics_df,
                pd.DataFrame(
                    {
                        "epoch": [epoch],
                        "train_loss": [train_loss],
                        "val_loss": [val_loss],
                        "val_acc": [val_acc],
                    }
                ),
            ],
            ignore_index=True,
        )

        print(
            f"Epoch {epoch}, Average train loss: {train_loss}, Average val loss: {val_loss}, Average val accuracy: {val_acc}"
        )

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"../result/{model_save_name}")
            print("Model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break

    writer.close()

    # 保存指标到csv文件
    metrics_df.to_csv(
        f'../result/training_metrics_{model_save_name.split(".")[0]}.csv', index=False
    )

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    sns.lineplot(data=metrics_df, x="epoch", y="train_loss", label="Train Loss")
    sns.lineplot(data=metrics_df, x="epoch", y="val_loss", label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 准确率曲线
    plt.subplot(1, 2, 2)
    sns.lineplot(data=metrics_df, x="epoch", y="val_acc", label="Val Accuracy")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(f'../result/training_curves_{model_save_name.split(".")[0]}.png')
    plt.close()


# %%
def evaluate(model, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # posts is a list of 50 strings for each of the 16 samples
            # We need to convert it to (16, 50) to fit the model
            posts = batch["posts"]  # Shape: (50, 16)
            real_batch_size = len(posts[0])
            posts_shape_fix = [[] for _ in range(real_batch_size)]
            for i, j in itertools.product(range(real_batch_size), range(50)):
                try:
                    posts_shape_fix[i].append(posts[j][i])
                except Exception as e:
                    print(f"Error: {e}, current i is {i}, current j is {j}")
                    print(f"current posts[j] is {posts[j]}")
                    exit(0)
            posts = posts_shape_fix
            labels = batch["type"].to(device)

            outputs = model(posts)
            accuracy = (outputs.argmax(dim=1) == labels).float().sum().item()
            total_acc += accuracy
            total_samples += labels.size(0)

    return total_acc / total_samples


# 评估所有模型并将结果写入Excel文件
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
dimensions = ['EI', 'NS', 'FT', 'JP']


# 创建数据集
dataset = MBTIDataset("../data/")
for i in range(len(dataset)):
    if len(dataset[i]["posts"]) != 50:
        print(f"{i} error, length is {len(dataset[i]['posts'])}")
        
print(dataset[0])

for dim in dimensions:
    print(f"\n========================== 开始处理{dim}维度 =========================")

    # 启动tensorboard
    os.system("nohup ./start_tb.sh > tb_log.txt 2>&1 &") 
    
    dataset.set_dim(dim)
    
    # 创建并训练模型
    model = MBTIClassifier()
    
    print(f"当前模型大小: {model.model_size()}")
    
    # 检查是否有此前保存的模型
    model_path = f"../result/best_model_{dim}.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    train(
        dataset,
        model,
        torch.device("cuda"),
        scheduler_type="linear", 
        epochs=200,
        batch_size=768,
        lr=1e-4,
        patience=3,
        model_save_name=f"best_model_{dim}.pth",
    )

    # 评估模型
    accuracy = evaluate(model, dataset, device)
    results.append({
        '维度': dim,
        '准确率': f"{accuracy * 100:.2f}%"
    })

    print(f"{dim}维度处理完成")

# 使用pandas创建DataFrame并保存为Excel
df = pd.DataFrame(results)
output_file = '../result/mbti_accuracy.xlsx'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_excel(output_file, index=False)

print(f"结果已保存到: {output_file}")
