import torch
from tqdm import tqdm
from src.utils.save import save_model, save_training_n_validation_loss, save_retrieval_metrics

def encode_texts_in_batches(model, texts, batch_size=32, desc="Building titles embeddings:", device="cpu"):
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch_texts = texts[i:i + batch_size]
            embeds = model.forward_text(batch_texts)  # [B, D]
            # embeds = embeds.cpu()                     # MOVE OFF GPU
            all_embeddings.append(embeds)

            del embeds
            torch.mps.empty_cache()  # important on Apple Silicon

    return torch.cat(all_embeddings, dim=0)

def build_images_in_batches(model, dataset_images, device, batch_size=32):
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset_images), batch_size), desc="Building image embeddings:"):
            batch = dataset_images[i:i + batch_size].to(device)  # already tensor
            embeds = model.forward_image(batch)  # [B, D]
            all_embeds.append(embeds)

            del embeds
            torch.mps.empty_cache()  # important on Apple Silicon

    return torch.cat(all_embeds, dim=0)

# ---------------- Retrieval Metrics ----------------
def retrieval_metrics(sim_matrix, gt_indices):
    # Calculate Recall@1,5,10, Median Rank, and MRR.

    ranks = []
    for i in range(sim_matrix.size(0)):
        sims = sim_matrix[i]
        sorted_idx = torch.argsort(sims, descending=True)
        rank = (sorted_idx == gt_indices[i]).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    ranks = torch.tensor(ranks, dtype=torch.float)
    r1 = (ranks < 1).float().mean().item()
    r5 = (ranks < 5).float().mean().item()
    r10 = (ranks < 10).float().mean().item()
    # medr = ranks.median().item()
    medr = ranks.median().item() + 1
    mrr = (1.0 / (ranks + 1)).mean().item()

    return {"R@1": r1, "R@5": r5, "R@10": r10, "MedR": medr, "MRR": mrr}


def calculate_contrastive_loss(emb_text, emb_long_text, emb_images, loss_fun):
    loss_text_image = loss_fun(emb_text, emb_images)
    loss_long_text_image = loss_fun(emb_long_text, emb_images)
    loss_title_long = loss_fun(emb_text, emb_long_text)

    # loss = 0.4 * loss_title_long + 1.0 * loss_text_image + 1.0 * loss_long_text_image
    loss = 0.3 * loss_title_long + 1.1 * loss_text_image + 1.1 * loss_long_text_image

    return loss


# ---------------- Validation Loop ----------------
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

            # Store embeddings for retrieval metrics
            text_embeds_list.append(emb_text)
            long_text_embeds_list.append(emb_long_text)
            image_embeds_list.append(emb_images)

    avg_loss = running_loss / len(loader)

    text_embeds = torch.cat(text_embeds_list, dim=0)
    long_text_embeds = torch.cat(long_text_embeds_list, dim=0)
    image_embeds = torch.cat(image_embeds_list, dim=0)

    # ---------------- Retrieval ----------------
    sim_t2i = torch.matmul(text_embeds, image_embeds.T)
    sim_i2t = torch.matmul(image_embeds, text_embeds.T)
    sim_t2lt = torch.matmul(text_embeds, long_text_embeds.T)
    sim_i2lt = torch.matmul(image_embeds, long_text_embeds.T)

    gt = torch.arange(len(loader.dataset))  # ground-truth indices

    results = {
        "val_loss": avg_loss,
        "Text->Image": retrieval_metrics(sim_t2i, gt),
        "Text->LongText": retrieval_metrics(sim_t2lt, gt),
        "Image->Text": retrieval_metrics(sim_i2t, gt),
        "Image->LongText": retrieval_metrics(sim_i2lt, gt),
    }

    return results


# ---------------- Training Loop ----------------
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

    avg_loss = running_loss / len(loader)
    return avg_loss


# ---------------- Full Training Function ----------------
def train_and_validate_model(model, train_loader, val_loader,
                             optimizer, loss_fun, device,
                             start_epoch=1, epochs=10,
                             save_path="fusion_model.pth"):
    total_epoch = start_epoch + epochs
    print(f'Total epochs: {total_epoch - 1}')

    for epoch in range(start_epoch, total_epoch):
        # ---------------- Training ----------------
        print(f"Epoch [{epoch}]")
        training_loss = train_one_epoch(model, train_loader, optimizer, loss_fun, device)
        print(f"Training Loss: {training_loss:.4f}")

        # ---------------- Validation ----------------
        val_results = validate_one_epoch(model, val_loader, loss_fun, device)
        validation_loss = val_results["val_loss"]
        print(f"Validation Loss: {validation_loss:.4f}")
        for key, metrics in val_results.items():
            if key != "val_loss":
                print(f"{key}: "
                      f"R@1 {metrics['R@1']:.3f}, R@5 {metrics['R@5']:.3f}, R@10 {metrics['R@10']:.3f}, "
                      f"MedR {float(metrics['MedR']):.3f}, MRR {metrics['MRR']:.3f}")

        # Save
        save_model(model, optimizer, epoch=epoch, path=save_path)
        save_training_n_validation_loss(epoch, training_loss, validation_loss)
        save_retrieval_metrics(epoch, val_results)
