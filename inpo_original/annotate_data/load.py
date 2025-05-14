from datasets import load_dataset, load_from_disk

# dataset_name = "RLHFlow/iterative-prompt-v1-iter1-20K"
# ds = load_dataset(dataset_name, split="train")

# data_keys = []
# for sample in ds:
#     context = sample["context_messages"]
#     n = len(context)
#     for i in range(n):
#         if i % 2 == 0:
#             assert context[i]['role'] == 'user'
#         else:
#             assert context[i]['role'] == 'assistant'


dataset_name = "/apdcephfs_us/share_300814644/user/yuhenyzhang/pref_datasets/ipopref2_0.005_iter1/data_pref_prob"
ds = load_from_disk(dataset_name)
print(len(ds))


