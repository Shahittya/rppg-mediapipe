from models.deepphys.dataset_loader import RPPGDataset

dataset = RPPGDataset("data/videos")

print("Total samples:", len(dataset))

motion, signal = dataset[0]

print("Motion shape:", motion.shape)
print("Signal shape:", signal.shape)