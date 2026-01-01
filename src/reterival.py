import torch

def retrieve(
        model,
        query_texts=None,
        query_images=None,
        dataset_title_embeds=None,
        dataset_ingredients_instructions_embeds=None,
        dataset_image_embeds=None,
        top_k=5,
        device="cpu"
):
    # Unified retrieval:
    #     - Text -> Text
    #     - Text -> Image
    #     - Image -> Text
    #     - Image -> Image
    # Returns a dictionary with scores and indices for each mode.
    model.eval()
    results = {}

    with torch.no_grad():
        # Text Queries
        if query_texts is not None:
            query_text_embeds = model.forward_text(query_texts).to(device)  # [Q, D]

            # Text -> Text
            if dataset_title_embeds is not None:
                sim_tt = torch.matmul(query_text_embeds, dataset_title_embeds.T)
                scores_tt, indices_tt = sim_tt.topk(top_k, dim=1)
                results["text->text"] = (indices_tt, scores_tt)

            # Text -> Ingredients Instructions
            if dataset_ingredients_instructions_embeds is not None:
                sim_tt = torch.matmul(query_text_embeds, dataset_ingredients_instructions_embeds.T)
                scores_tt, indices_tt = sim_tt.topk(top_k, dim=1)
                results["text->ingredients_instructions"] = (indices_tt, scores_tt)

            # Text -> Image
            if dataset_image_embeds is not None:
                sim_ti = torch.matmul(query_text_embeds, dataset_image_embeds.T)
                scores_ti, indices_ti = sim_ti.topk(top_k, dim=1)
                results["text->image"] = (indices_ti, scores_ti)

            # Free memory
            del query_text_embeds
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # Image Queries
        if query_images is not None:
            query_images = query_images.to(device)
            query_img_embeds = model.forward_image(query_images)  # [Q, D]

            # Image -> Text (title)
            if dataset_title_embeds is not None:
                sim_it = torch.matmul(query_img_embeds, dataset_title_embeds.T)
                scores_it, indices_it = sim_it.topk(top_k, dim=1)
                results["image->text"] = (indices_it, scores_it)

            # Image -> Text (target: ingredients + instructions)
            if dataset_ingredients_instructions_embeds is not None:
                sim_itarget = torch.matmul(query_img_embeds, dataset_ingredients_instructions_embeds.T)
                scores_itarget, indices_itarget = sim_itarget.topk(top_k, dim=1)
                results["image->ingredients_instructions"] = (indices_itarget, scores_itarget)

            # Image -> Image
            if dataset_image_embeds is not None:
                sim_ii = torch.matmul(query_img_embeds, dataset_image_embeds.T)

                # Remove self-matching ONLY if same pool
                if query_img_embeds.shape[0] == dataset_image_embeds.shape[0]:
                    sim_ii.fill_diagonal_(-1e9)

                scores_ii, indices_ii = sim_ii.topk(top_k, dim=1)
                results["image->image"] = (indices_ii, scores_ii)

            # Free memory
            del query_img_embeds
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    return results