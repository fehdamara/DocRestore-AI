"""
test_dataset.py

Small test script to verify that the dataset loads clean images,
generates damaged images, and returns tensors correctly.
"""

# Importiamo la funzione per creare il dataloader
from src.dataset import create_dataloader


# Creiamo il dataloader usando immagini pulite
dataloader = create_dataloader(
    clean_dir="data/clean",
    image_size=256,
    batch_size=2,
    shuffle=True,
    generate_damage=True
)

# Prendiamo un batch dal dataloader
batch = next(iter(dataloader))

# Stampiamo le dimensioni dei tensori
print("Damaged batch shape:", batch["damaged"].shape)
print("Clean batch shape:", batch["clean"].shape)
print("Filenames:", batch["filename"])