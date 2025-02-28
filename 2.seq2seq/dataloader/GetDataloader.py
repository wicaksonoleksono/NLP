import os
import pickle
from torch.utils.data import DataLoader
from ._collate_fn import collate_fn
from ._dataset import TranslationDataset
def get_dataloader(
    pth, 
    batch_size=32, 
    preprocessed_file="preprocessed_data.pkl"
):
    with open(os.path.join(pth, preprocessed_file), "rb") as f:
        data_dict = pickle.load(f)
    train_pairs = data_dict["train"]
    val_pairs   = data_dict["validation"]
    test_pairs  = data_dict["test"]

    train_dataset = TranslationDataset(train_pairs)
    val_dataset   = TranslationDataset(val_pairs)
    test_dataset  = TranslationDataset(test_pairs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader
