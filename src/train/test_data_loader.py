import asyncio

from train_dataloader import getEffecientPileV1

from transformers import AutoTokenizer


async def test_lazy_data_loader():
    print("Initializing lazy data loader...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-moe-16b-base")
    dataset_generator = getEffecientPileV1(tokenizer)  # Shuffle set to True to test the shuffling mechanism as well

    print("Fetching a few batches from the dataset...")
    batch_count = 0
    max_batches_to_fetch = 5  # Limit the number of batches to fetch for this test

    async for batch in dataset_generator():
        batch_count += 1
        print(f"\nBatch {batch_count} fetched.")
        print(f"Batch size: {len(batch['input_ids'])}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")

        if batch_count >= max_batches_to_fetch:
            break

    print("\nFinished fetching batches.")


if __name__ == "__main__":
    asyncio.run(test_lazy_data_loader())
