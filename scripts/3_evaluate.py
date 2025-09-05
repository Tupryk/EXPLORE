import torch
from omegaconf import DictConfig
import hydra
from utils.logger import get_logger
from models.my_model import MyModel
from datasets.my_dataset import MyDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

obs, _ = env.reset()
    env.render()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger = get_logger(cfg)
    logger.info("Starting evaluation...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    model = MyModel(cfg.model).to(device)
    model.load_state_dict(torch.load(cfg.eval.checkpoint_path, map_location=device))
    model.eval()

    # Load test data
    test_dataset = MyDataset(cfg.data.test_path)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers
    )

    # Run evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
