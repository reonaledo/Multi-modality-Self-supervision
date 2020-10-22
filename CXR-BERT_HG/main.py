import argparse

from torch.utils.data import DataLoader

from data.helper import get_transforms
from data.dataset import CXRDataset
from models.cxrbert import CXRBERT, CXRBertEncoder
from models.train import CXRBERT_Trainer

from transformers import BertTokenizer

def train(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

    transforms = get_transforms()

    # TODO: erase or something to do...
    vocab = None  # actually not in used

    # print("Loading Vocab", args.vocab_path)
    # vocab = WordVocab.load_vocab(args.vocab_path)
    # print("Vocab Size: ", len(vocab))

    print("Load Train dataset", args.train_dataset)
    train_dataset = CXRDataset(args.train_dataset, tokenizer, transforms, vocab, args)

    print("Load Test dataset", args.test_dataset)
    test_dataset = CXRDataset(args.test_dataset, tokenizer, transforms, vocab, args) \
        if args.test_dataset is not None else None

    print("Create DataLoader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building CXRBERT model")
    # TODO: CXRBERT or CXRBertEncoder... ?
    cxrbert = CXRBertEncoder(args)

    print("Creating BERT Trainer")
    trainer = CXRBERT_Trainer(args, cxrbert, train_data_loader=train_data_loader, test_dataloader=test_data_loader,
                              lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
                              weight_decay=args.adam_weight_decay, warmup_steps=10000,
                              with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)


    """
    def __init__(self, args, bert: CXRBertEncoder, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
    """

    print("Training Start!")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TODO: train_dataset & test_dataset seems similar with '--data_path'
    parser.add_argument("--train_dataset", required=True, type=str, help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default=None, help='test dataset for evaluate train set')
    # TODO: check
    parser.add_argument("--vocab_path", required=True, type=str, help="build vocab model path with bert-vocab")
    parser.add_argument("--output_path", required=True, type=str, help="ex)output/bert.model")

    # TODO: bert-base, parameter, check
    # parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int, default=8, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence len")

    parser.add_argument("--batch_size", type=int, default=64, help="number of batch size")
    parser.add_argument("--epochs", type=int, default=10, help='number of epochs')
    parser.add_argument("--num_workers", type=int, default=5, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    # TODO: check
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    # parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    # TODO: img-SGD, txt-AdamW
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")

    # add, according to the mmbt
    # TODO: loading BlueBERT
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased',
                        choices=["bert-base-uncased", "BlueBERT"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--save_dir", type=str, default="/path/to/save_dir/")  # seems like --output_path

    # TODO: embed_sz...??
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--num_image_embeds", type=int, default=10)  # TODO: num_image_embeds must be discussed ...!!!

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # TODO: ...!
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)

    # TODO: ???
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--weight_classes", type=int, default=1)

    args = parser.parse_args()

    train(args)
