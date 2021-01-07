import torch
import torch.nn as nn

"""
super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        # self.linear.weight = CXRBertEncoder(args).txt_embeddings.word_embeddings.weight
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: torch.Size([8, 512, 768])
        # return: torch.Size([8, 512, 30522])
        # return self.softmax(self.linear(x))
        return self.linear(x)

# print('mlm_output.size:', mlm_output.size())  # [bsz, seq_len, vocab_sz]
# print('txt_labels', txt_labels.size()) # torch.Size([16, 512])

# itm prediction accuracy
correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
total_correct += correct
total_element += is_aligned.nelement()

# mlm accuracy
mlm_correct = mlm_output.argmax(dim=-1).eq(txt_labels).sum().item()
total_mlm_correct += mlm_correct
total_mlm_element += txt_labels.nelement()
"""
linear = nn.Linear(768, 30522)
softmax = nn.LogSoftmax(dim=-1)

# x = torch.rand(8, 512, 768, dtype=torch.LongTensor)
x = torch.rand(8, 512, 768)
txt_labels = torch.rand(8, 512)
output = linear(x)
print(output.size())  # torch.Size([8, 512, 30522])
print(output.argmax(dim=-1))

correct = output.argmax(dim=-1).eq(txt_labels).sum().item()
print(correct)

test_labels = torch.Tensor([[0, 5, 3, 0], [1, 3, 0, 0]])
print('test_labels:', test_labels)
print('test_labels_size:', test_labels.size())

test_output = torch.Tensor(
    [[[0.1, 0.5, 0.2, 0.2, 0.0], [0.1, 0.1, 0.1, 0.1, 0.6], [0.1, 0.1, 0.6, 0.1, 0.1], [0.5, 0.1, 0.1, 0.1, 0.2]],
     [[0.7, 10, 0.2, 0.2, 0.0], [0.1, 0.9, 0.1, 10, 0.6], [9, 0.1, 0.6, 0.1, 0.1], [5, 0.1, 0.1, 0.1, 0.7]]])
print('test_output:', test_output)
print('test_output_size:', test_output.size())
print('test_output_argmax:', test_output.argmax(dim=-1))
print('test_output_eq:', test_output.argmax(dim=-1).eq(test_labels))