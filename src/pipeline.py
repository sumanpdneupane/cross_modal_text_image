import torch
from tqdm import tqdm
from src.utils.save import save_model
from src.utils.save import save_training_n_validation_loss
from src.utils.save import save_retrieval_metrics
from src.utils.utils import clean_memory


def prepare_dataset_tensors(full_dataset, desc="Preparing texts & metadata"):
    dataset_texts = []
    dataset_ingredients_instructions = []
    dataset_metadata = []
    dataset_images = []

    for i in tqdm(range(len(full_dataset)), desc=desc):
        sample = full_dataset[i]
        dataset_texts.append(sample["input_text"])  # title
        dataset_ingredients_instructions.append(sample["target_text"])  # ingredients, instructions
        dataset_images.append(sample["image"])
        dataset_metadata.append(sample["metadata"])
    # Stack images into a single tensor [N, C, H, W]
    dataset_images = torch.stack(dataset_images)
    return dataset_texts, dataset_ingredients_instructions, dataset_metadata, dataset_images


def encode_texts_in_batches(model, texts, batch_size=32, desc="Building titles embeddings:", device="cpu"):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i + batch_size]
            embeds = model.forward_text(batch_texts)
            all_embeddings.append(embeds)

            del embeds
            clean_memory(device=device)

    return torch.cat(all_embeddings, dim=0)


def encode_images_in_batches(model, dataset_images, device, batch_size=32):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset_images), batch_size), desc="Building image embeddings:"):
            batch = dataset_images[i:i + batch_size].to(device)
            embeds = model.forward_image(batch)
            all_embeds.append(embeds)

            del embeds, batch
            clean_memory(device=device)

    return torch.cat(all_embeds, dim=0)


def retrieval_metrics(sim_matrix, gt_indices, device="cpu"):
    sorted_idx = torch.argsort(sim_matrix, dim=1, descending=True)
    matches = (sorted_idx == gt_indices.unsqueeze(1))
    ranks = matches.float().argmax(dim=1)

    r1 = (ranks < 1).float().mean().item()
    r5 = (ranks < 5).float().mean().item()
    r10 = (ranks < 10).float().mean().item()

    del sorted_idx, matches, ranks
    clean_memory(device=device)

    return {"R@1": r1, "R@5": r5, "R@10": r10}


def calculate_contrastive_loss(emb_text, emb_long_text, emb_images, loss_fun):
    loss_t2i = loss_fun(emb_text, emb_images)
    loss_i2t = loss_fun(emb_images, emb_text)

    loss_lt2i = loss_fun(emb_long_text, emb_images)
    loss_i2lt = loss_fun(emb_images, emb_long_text)

    loss_t2lt = loss_fun(emb_text, emb_long_text)
    loss_lt2t = loss_fun(emb_long_text, emb_text)

    loss = (
                   loss_t2i + loss_i2t +
                   loss_lt2i + loss_i2lt +
                   loss_t2lt + loss_lt2t
           ) / 6

    return loss


def validate_one_epoch(model, loader, loss_fun, device):
    model.eval()
    text_embeds_list, long_text_embeds_list, image_embeds_list = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            texts = batch["input_text"]
            long_texts = batch["target_text"]
            images = batch["image"].to(device)

            emb_text = model.forward_text(texts).to(device)
            emb_long_text = model.forward_long_text(long_texts).to(device)
            emb_images = model.forward_image(images).to(device)

            batch_loss = calculate_contrastive_loss(emb_text, emb_long_text, emb_images, loss_fun)
            running_loss += batch_loss.item()

            text_embeds_list.append(emb_text)
            long_text_embeds_list.append(emb_long_text)
            image_embeds_list.append(emb_images)

            del emb_text, emb_long_text, emb_images, batch_loss
            clean_memory(device=device)

    avg_loss = running_loss / len(loader)

    text_embeds = torch.cat(text_embeds_list, dim=0)
    long_text_embeds = torch.cat(long_text_embeds_list, dim=0)
    image_embeds = torch.cat(image_embeds_list, dim=0)

    # Clear memory of intermediate lists
    del text_embeds_list, long_text_embeds_list, image_embeds_list

    # Retrieval
    sim_t2i = torch.matmul(text_embeds, image_embeds.T)
    sim_i2t = torch.matmul(image_embeds, text_embeds.T)
    sim_t2lt = torch.matmul(text_embeds, long_text_embeds.T)
    sim_i2lt = torch.matmul(image_embeds, long_text_embeds.T)

    gt = torch.arange(len(loader.dataset), device=device)  # ground-truth indices

    results = {
        "val_loss": avg_loss,
        "Text->Image": retrieval_metrics(sim_t2i, gt, device),
        "Text->LongText": retrieval_metrics(sim_t2lt, gt, device),
        "Image->Text": retrieval_metrics(sim_i2t, gt, device),
        "Image->LongText": retrieval_metrics(sim_i2lt, gt, device),
    }

    # Clear large tensors
    del text_embeds, long_text_embeds, image_embeds
    del sim_t2i, sim_i2t, sim_t2lt, sim_i2lt, gt
    clean_memory(device=device)

    return results


def train_one_epoch(model, loader, optimizer, loss_fun, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        texts = batch["input_text"]
        long_texts = batch["target_text"]
        images = batch["image"].to(device)

        optimizer.zero_grad()

        emb_text = model.forward_text(texts).to(device)
        emb_long_text = model.forward_long_text(long_texts).to(device)
        emb_images = model.forward_image(images).to(device)

        loss = calculate_contrastive_loss(emb_text, emb_long_text, emb_images, loss_fun)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        del emb_text, emb_long_text, emb_images, loss
        clean_memory(device=device)

    avg_loss = running_loss / len(loader)
    return avg_loss


def train_and_validate_model(model, train_loader, val_loader,
                             optimizer, loss_fun, device,
                             start_epoch=1, epochs=10,
                             save_path="fusion_model.pth"):
    total_epoch = start_epoch + epochs
    print(f'Start epoch: {start_epoch}, Total epochs: {total_epoch - 1}')

    for epoch in range(start_epoch, total_epoch):
        # Training
        print(f"Epoch [{epoch}]")
        training_loss = train_one_epoch(model, train_loader, optimizer, loss_fun, device)
        print(f"Training Loss: {training_loss:.4f}")

        # Validation
        val_results = validate_one_epoch(model, val_loader, loss_fun, device)
        validation_loss = val_results["val_loss"]
        print(f"Validation Loss: {validation_loss:.4f}")
        for key, metrics in val_results.items():
            if key != "val_loss":
                print(f"{key}: "
                      f"R@1 {metrics['R@1']:.3f}, R@5 {metrics['R@5']:.3f}, R@10 {metrics['R@10']:.3f}")

        # Save
        save_model(model, optimizer, epoch=epoch, path=save_path)
        save_training_n_validation_loss(epoch, training_loss, validation_loss)
        save_retrieval_metrics(epoch, val_results)

        # Clean GPU after each epoch
        clean_memory(device=device)
