import pickle
from torch.utils.data import DataLoader


def load_dataset(batch_size, file_path):
    with open(file_path, "rb") as dataset_file:
        dataset = pickle.load(dataset_file)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True)
    
    return dataloader

def stat(file_path):
    data_loader = load_dataset(BATCH_SIZE, file_path)
    print("记录数:", len(data_loader.dataset))

if __name__ == '__main__':
    BATCH_SIZE = 32
    train_dataset_path = "../model/ch7/chat/train_dataset.pkl"
    stat(train_dataset_path)

    valid_dataset_path = "../model/ch7/chat/valid_dataset.pkl"
    stat(valid_dataset_path)

    test_dataset_path = "../model/ch7/chat/test_dataset.pkl"
    stat(test_dataset_path)
