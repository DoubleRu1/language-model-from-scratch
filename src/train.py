import argparse
import logging
from einops import rearrange
import torch
from torch import Tensor
from dataloader import TokenDataset
from structure import TransformerLm
from tokenizer import BPE_Tokenizer
from criterion import CrossEntropy
from optimizer import AdamwCls
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

class Trainer():
    def __init__(self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        epoches: int,
        ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dataloader = dataloader
        self.epoches = epoches
        self.total_steps = len(self.dataloader) * self.epoches

    def train(self):
        self.model.train()
        for epoch in range(self.epoches):
            for i, (inputs, targets) in enumerate(self.dataloader):
                inputs = inputs
                targets = targets
                outputs = self.model(inputs)
                # print(outputs.shape)
                # print(targets.shape)
                outputs = rearrange(outputs, 'b t c -> (b t) c')
                targets = rearrange(targets, 'b t -> (b t)')
                loss = self.criterion(outputs, targets)
                logging.info(f"Epoch {epoch} Loss: {loss.item()} ({i}/{self.total_steps})")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--vocab_size", type = int, default = 10000, help= "词表大小")
    args.add_argument("--context_length", type = int, default = 1024, help="上下文长度")
    args.add_argument("--d_model", type = int, default = 256, help = "隐藏层大小")
    args.add_argument("--num_layers", type = int, default = 12, help = "层数")
    args.add_argument("--num_heads", type = int, default = 8, help = "注意力头数")
    args.add_argument("--d_ff", type = int, default = 512, help = "前馈神经网络大小")
    args.add_argument("--rope_theta", type = float, default = 1e6, help = "位置编码超参数")
    args.add_argument("--weights", type = dict[str, Tensor], default = None, help = "预训练权重")
    args.add_argument("--epoches", type = int, default = 3, help = "训练轮数")
    args = args.parse_args()

    vocab_size = args.vocab_size
    context_length = args.context_length
    d_model = args.d_model
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = args.rope_theta
    epoches = args.epoches

    data_path = "data/TinyStories-valid.txt"
    vocab_path = "data/vocab.json"
    merges_path = "data/merges.json"  # 改为.json扩展名

    # special_tokens = ["<|endoftext|>"]
    # logging.info("Training BPE tokenizer...")
    # vocab, merges = run_train_bpe(data_path, vocab_size, special_tokens)
    
    # with open(vocab_path, "w") as f:
    #     # 将bytes转换为字符串以便JSON序列化
    #     serializable_vocab = {str(k): v.decode('latin1', errors='replace') for k, v in vocab.items()}
    #     json.dump(serializable_vocab, f, ensure_ascii=False, indent=4)
    
    # # 将merges转换为可序列化的格式
    # serializable_merges = []
    # for first, second in merges:
    #     # 将bytes转换为其整数表示，这样可以正确处理所有字节，包括不可打印字符
    #     first_bytes = [b for b in first]
    #     second_bytes = [b for b in second]
    #     serializable_merges.append([first_bytes, second_bytes])
    
    # with open(merges_path, "w") as f:
    #     json.dump(serializable_merges, f)
    # logging.info("BPE tokenizer trained and saved to %s and %s", vocab_path, merges_path)

    tokenizer = BPE_Tokenizer.from_files(vocab_path, merges_path)
    # print(tokenizer.vocab)
    # logging.info("Encoding text...")
    # with open(data_path, "r") as f:
    #     text = f.read()
    # token_ids = tokenizer.encode(text)  # List[int]

    # token_tensor = torch.tensor(token_ids, dtype=torch.int32)  # vocab 小于 65536 用 int32 就够
    # torch.save({"tokens": token_tensor, "block_size": context_length}, "data/tinystories_tokens.pt")


    ckpt = torch.load("data/tinystories_tokens.pt", map_location="cpu")
    token_tensor = ckpt["tokens"]
    block_size = ckpt["block_size"]
    dataset = TokenDataset(token_tensor.tolist(), block_size)

    # dataset = TokenDataset(token_ids, context_length)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = TransformerLm(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)

    criterion = CrossEntropy()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    
    optimizer = AdamwCls(model.parameters(), lr = 1e-3)

    trainer = Trainer(model, optimizer, criterion, device, dataloader,epoches)


    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Total parameters: {total / 1e6:.2f}M")
    logging.info(f"Trainable parameters: {trainable / 1e6:.2f}M")

    trainer.train()