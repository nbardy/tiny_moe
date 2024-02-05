import asyncio
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
import datasets

import asyncio
import pyarrow
import pandas as pd
import json

import torch

import random

MAX_CHUNK_SIZE = 512


def tokenize_dataset(item, tokenizer):
    text = item["text"]
    tokenized_text = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Directly return the tokenized text without chunking
    return {
        "input_ids": tokenized_text["input_ids"],
        "attention_mask": tokenized_text["attention_mask"],
        "labels": tokenized_text["input_ids"].clone(),
    }


import logging


# Define formatter as a top-level function
def formatter(item, header_columns, prefix="", chunk_size=512, base_key="markdown"):
    try:
        if base_key not in item or not item[base_key]:
            logging.warning(f"Missing or empty base_text for item: {item}")
            return []

        base_text = str(item[base_key])
        markdown_chunks = [" ".join(base_text.split()[i : i + chunk_size]) for i in range(0, len(base_text.split()), chunk_size)] if base_text else []

        all_chunks = []

        for chunk in markdown_chunks:
            header_text = " ".join(f"{key}: {item.get(key, 'N/A')}" for key in header_columns)
            formatted_text = f"{prefix} {header_text} Excerpt: {chunk}"
            all_chunks.append({"text": formatted_text})

        if prefix != "" and markdown_chunks:
            random_chunk = random.choice(markdown_chunks)
            formatted_text = f"{prefix} {random_chunk}"
            all_chunks.append({"text": formatted_text})

        import pyarrow
        import pandas as pd

        table = pyarrow.Table.from_pandas(pd.DataFrame(all_chunks))

        return table

    except Exception as e:
        logging.exception(f"Exception occurred while formatting item: {item}")
        return []


# Modify format_item_fn to return a partial function of formatter with preset arguments
from functools import partial


def format_item_fn(header_columns, prefix="", chunk_size=512, base_key="markdown"):
    # Return a partial function with preset arguments
    return partial(formatter, header_columns=header_columns, prefix=prefix, chunk_size=chunk_size, base_key=base_key)


def format_chat(item, column_name="conversations", from_key="from", value_key="value", prefix="", chunk_size=MAX_CHUNK_SIZE):
    """
    Formats chat logs into a string or text chat log format based on a random mode.
    Args:
        item (dict): The item containing the chat logs.
        column_name (str): The key in the item dict that contains the chat logs.
        from_key (str): The key in each chat log dict that identifies the sender.
        value_key (str): The key in each chat log dict that contains the message text.
    Returns:
        pyarrow.Table: A table containing the formatted chat logs.
    """

    # Ensure the column exists in the item
    if column_name not in item or not item[column_name]:
        logging.warning(f"Missing or empty {column_name} for item: {item}")
        return []

    chat_logs = item[column_name]
    mode = random.choice(["string", "chat_log"])

    all_chunks = []

    if mode == "string":
        # Dump json
        json_str = json.dumps(chat_logs)
        # Split into chunk size chunks and label with "{prefix} ""
        chunk_size = 512  # Define chunk size for splitting the json string
        chat_string = ""
        for i in range(0, len(json_str), chunk_size):
            chunk = json_str[i : i + chunk_size]
            if prefix:
                chunk = f"{prefix} {chunk}"
            chat_string += chunk + " " if i + chunk_size < len(json_str) else chunk
        all_chunks.append({"text": chat_string})
    else:
        # Format each chat message into a 'from: message' format
        chat_formatted = "\n".join([f"{log[from_key]}: {log[value_key]}" for log in chat_logs if from_key in log and value_key in log])
        all_chunks.append({"text": chat_formatted})

    # Convert to pyarrow table
    table = pyarrow.Table.from_pandas(pd.DataFrame(all_chunks))

    return table


format_science_papers = format_item_fn(["absract", "article"], prefix="Science paper", base_key="article")
format_wiki = format_item_fn(["title", "categories"], prefix="Wikipedia", base_key="markdown")
format_mathpile_chunks = format_item_fn([], prefix="Math Pile", base_key="text")
format_pile_chunks = format_item_fn([], prefix="Web Text", base_key="text")
format_grounded_textbooks = format_item_fn(["topic", "model", "concepts", "outline"], prefix="Grounded Synthetic Textbook")
format_synthetic_textbooks = format_item_fn(["model", "topic", "concepts", "outline"], prefix="Synthetic Textbooks")
format_fake_textbooks = format_item_fn(["model", "topic", "outline", "field", "subfield"], prefix="Fake Textbooks(Some Grounded)")
format_orca_textbooks = format_item_fn([], base_key="textbook", prefix="Tiny Orca Textbooks")
format_strange_textbooks = format_item_fn([], prefix="Tiny Strange Textbooks", base_key="text")
format_math_textbooks = format_item_fn([], prefix="Tiny Math Textbooks", base_key="text")
format_code_textbooks = format_item_fn([], prefix="Tiny Code Textbooks", base_key="text")
format_tiny_textbooks = format_item_fn([], prefix="Tiny Textbooks", base_key="text")
format_muse_textbooks = format_item_fn([], prefix="Muse Textbooks", base_key="text")
format_all_you_need_books = format_item_fn([], prefix="Synthetic Code Textbooks", base_key="text")
format_anthropic_chat = format_item_fn([], prefix="Safe Chat(Chosen)", base_key="chosen")
format_anthropic_chat_rejected = format_item_fn([], prefix="Safe Chat(Rejected)", base_key="rejected")

format_auto_math_text = format_item_fn([], prefix="Math Papers", base_key="text")
format_hq_math_text = format_item_fn([], prefix="Quality Math Papers", base_key="text")
format_auto_math_code = format_item_fn([], prefix="Math Code", base_key="text")

format_open_hermes_2_5 = format_item_fn([], prefix="Open Hermes Conversation", base_key="text")

# huggingface datasets lazy stuff isn't working so we do it ourselves


def _getLazyPool(n_proc=16):
    """
    Creates a shared ProcessPoolExecutor and returns a closure function to utilize this shared pool
    for executing tasks in parallel.

    Args:
        n_proc (int): The number of processes to use.

    Returns:
        A closure function that takes a dataset and a function (f) as arguments and applies f to each
        item of the dataset in parallel using the shared ProcessPoolExecutor.
    """
    pool = ProcessPoolExecutor(max_workers=n_proc)

    async def dataLazyMap(dataset, f):
        """
        An asynchronous generator that applies a function to each item of a dataset in parallel,
        managing a buffer of futures to yield results as they are completed. It ensures that the
        buffer size does not exceed 2 * n_proc, allowing for efficient parallel processing using a
        shared ProcessPoolExecutor.

        Args:
            dataset: An iterable dataset.
            f: A function to apply to each item of the dataset.

        Yields:
            The result of applying f to each item of the dataset, as they are completed.
        """
        buffer_size = 2 * n_proc
        futures = []

        while True:
            if len(futures) < buffer_size:
                datum = await dataset.__anext__()  # [Change] Use __anext__() to fetch next item from async generator

                if datum is None:
                    break
                # future = pool.submit(f, datum)
                loop = asyncio.get_event_loop()
                future = asyncio.ensure_future(loop.run_in_executor(pool, f, datum))
                futures.append(future)

            if len(futures) >= buffer_size:
                await asyncio.sleep(0.2)  # Async sleep to yield control and manage buffer size

            done, _ = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED, timeout=0)

            for future in done:
                try:
                    yield future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    import traceback

                    traceback.print_exc()

                futures.remove(future)

        # cleanup all pending queue items

        while futures:
            done, _ = await asyncio.wait(futures, return_when=asyncio.FIRST_COMPLETED)
            for future in done:
                try:
                    yield future.result()
                except Exception as exc:
                    print(f"Generated an exception: {exc}")
                    import traceback

                    traceback.print_exc()
                futures.remove(future)

    async def dataLazyFlatMap(dataset, f):
        """
        An asynchronous generator that applies a function to each item of a dataset in parallel,
        flattening the result if the applied function returns an iterable, using dataLazyMap for parallel processing.

        Args:
            dataset: An iterable dataset.
            f: A function to apply to each item of the dataset, which may return an iterable.

        Yields:
            The result of applying f to each item of the dataset, as they are completed, potentially flattened if the result is an iterable.
        """

        async for item in dataLazyMap(dataset, f):
            if type(item) == list:
                for sub_item in item:
                    yield sub_item
            else:
                yield item

    return dataLazyFlatMap, dataLazyMap


dataLazyFlatMap, dataLazyMap = _getLazyPool()


async def dataLazyInterleave(datasets, probabilities, length=1000000):  # [Change] Added length parameter with default value of 1,000,000
    # norm probabilities to total of 1
    probabilities = [p / sum(probabilities) for p in probabilities]  # Normalize probabilities to sum to 1
    count = 0
    while datasets and count < length:  # [Change] Added condition to stop after yielding length items
        chosen_dataset = random.choices(datasets, weights=probabilities, k=1)[0]
        try:
            item = await chosen_dataset.__anext__()  # [Change] Use async generator's __anext__ method to fetch next item
            yield item
            count += 1  # Increment count after each yield
        except StopAsyncIteration:  # [Change] Catch StopAsyncIteration for async generators
            datasets.remove(chosen_dataset)


def load_data(dataset_name, split=None, f=None, subset_name=None, shuffle=False):
    """
    Loads a dataset with streaming enabled and applies a transform function in batched mode.

    Args:
        dataset_name (str): The name of the dataset to load.
        subset_name (str): The subset name of the dataset.
        transform_function (function): The transform function to apply to the dataset.

    Returns:
        Dataset: The loaded and transformed dataset.
    """
    # assert transform_function is set
    if f is None:
        raise ValueError("f(transform_function) must be set")

    print(f"Loading {dataset_name}, {subset_name}, {split}")
    dataset = load_dataset(dataset_name, name=subset_name, split=split, streaming=True)
    if shuffle:
        dataset = dataset.shuffle()
    # transformed_dataset = dataset.map(f, batched=True)

    # Convert the IterableDataset to __anext__
    async def to_async_iterable(dataset):
        ds = iter(dataset)

        while True:
            try:
                yield next(ds)
            except StopIteration:
                yield None
                break

    async_dataset = to_async_iterable(dataset)
    transformed_dataset = dataLazyFlatMap(async_dataset, f)

    return transformed_dataset


def collate_data(batch):
    input_ids = torch.stack([item["input_ids"] for sublist in batch for item in sublist], dim=0)
    labels = torch.stack([item["labels"] for sublist in batch for item in sublist], dim=0)
    attention_masks = torch.stack([item["attention_mask"] for sublist in batch for item in sublist], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


# replace torch dataloader
def dataLazyBatch(lazy_dataset_generator, collate_fn, batch_size):
    async def generator():
        batch = []
        async for item in lazy_dataset_generator:
            batch.append(item)
            if len(batch) == batch_size:
                yield collate_fn(batch)
                batch = []
        if batch:  # Handle the last batch if it's smaller than batch_size
            yield collate_fn(batch)

    return generator


# Modifying the collate function to handle chunks
def collate_data(batch):
    input_ids = torch.stack([item["input_ids"] for sublist in batch for item in sublist], dim=0)
    labels = torch.stack([item["labels"] for sublist in batch for item in sublist], dim=0)
    attention_masks = torch.stack([item["attention_mask"] for sublist in batch for item in sublist], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


##
# imports a giant merge dataset cut into chunks.dataset
#
# This dataset aims to provide a slim number for token in the 1-2B range
# for effeciently training models with synthetic data and curated web data
#
#  We a few typses of web data:
#    - scientific papers
#    - Slim web crawls
# Synthetic data is of the instruct, chat, and textbook variety
#
# Hypothesis:
# Can we train on small context window 512 chuks to train a model that can serve as a effecient foundation model training run
# Can we finetune that model on more complex downstream tasks?


# Small dataset for training effecient text models(Goal sub $2k taining runs)
#
# ~1 million rows of effecient web(General)
# ~1 million rows of science paper(Reasoning)
# ~1 million rows of instruct synthetic
# ~1 million rows of synthetic textbooks
# small amount of grounded textbooks
def getEffecientPileV1(tokenizer):
    all_datasets = [
        # 30% web text
        (0.15, load_data("JeanKaddour/minipile", "train", f=format_pile_chunks)),
        # 34% papers on xaxriv
        # (0.14, load_data("GAIR/MathPile_Commercial", "train", f=format_mathpile_chunks)),
        (0.12, load_data("scientific_papers", f=format_science_papers, subset_name="arxiv", split="train")),
        # (0.02, load_data("math-ai/AutoMathText", "code-python-0.80-to-1.00", f=format_auto_math_text)),
        # (0.04, load_data("math-ai/AutoMathText", "web-0.80-to-1.00", f=format_hq_math_text)),
        # (0.04, load_data("math-ai/AutoMathText", "arxiv-0.80-to-1.00", f=format_hq_math_text)),
        # 15% synthetic instruct/chat
        (0.15, load_data("teknium/OpenHermes-2.5", "train", f=format_chat)),
        # 5% wiki
        # (0.05, load_data("euirim/goodwiki", "train", f=format_wiki)),
        # Rest is synthetic textbooks
        (0.05, load_data("SciPhi/textbooks-are-all-you-need-lite", "train", f=format_all_you_need_books)),
        (0.05, load_data("open-phi/textbooks_grounded", "train", f=format_grounded_textbooks)),
        (0.002, load_data("Nbardy/art-theory-textbooks", "train", f=format_all_you_need_books)),
        (0.002, load_data("Nbardy/science-theory-textbooks", "train", f=format_all_you_need_books)),
        # (0.04, load_data("Anthropic/hh-rlhf", "train", f=format_anthropic_chat)),
        # (0.04, load_data("Anthropic/hh-rlhf", "train", f=format_anthropic_chat_rejected)),
        # Additional datasets commented out for potential future restoration
        # (0.01, load_data("open-phi/wile-e", "train", format_synthetic_textbooks)),
        # (0.02, load_data("open-phi/textbooks", "train", format_fake_textbooks)),
        # (0.05, load_data("nampdn-ai/tiny-orca-textbooks", "train", format_orca_textbooks)),
        # (0.005, load_data("nampdn-ai/tiny-strange-textbooks", "train", format_strange_textbooks)),
        # (0.02, load_data("nampdn-ai/tiny-textbooks", "train", format_tiny_textbooks)),
        # (0.04, load_data("TanvirOnHF/muse_textbooks", "train", format_muse_textbooks)),
        # (0.01, load_data("ndurkee/muse_textbooks", "train", format_muse_textbooks)),
        # (0.02, load_data("SkySyrup/muse_textbooks", "train", format_muse_textbooks)),
        # (0.02, load_data("Ba2han/muse_textbooks", "train", format_muse_textbooks)),
        # (0.01, load_data("amongglue/muse_textbooks", "train", format_muse_textbooks)),
        # (0.06, load_data("nampdn-ai/tiny-math-textbooks", "train", format_math_textbooks)),
        # (0.04, load_data("nampdn-ai/tiny-code-textbooks", "train", format_code_textbooks)),
        # (0.1, load_data("cognitivecomputations/dolphin", "train", instruct_model)),
    ]

    dataset_list = list(map(lambda x: x[1], all_datasets))

    dataset = dataLazyInterleave(
        datasets=dataset_list,
        probabilities=list(map(lambda x: x[0], all_datasets)),
    )
    dataset = dataLazyMap(dataset, partial(tokenize_dataset, tokenizer=tokenizer))
    dataset = dataLazyBatch(dataset, collate_data, batch_size=108)

    return dataset
