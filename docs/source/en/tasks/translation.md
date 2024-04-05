<!--
# docs/source/en/tasks/translation.md
# 
# git pull from huggingface/transformers by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Mar 22, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Apr 5, 2024
# 
# 该文档介绍了如何进行翻译。
-->

# Translation

**Translation converts a sequence of text from one language to another.** It is one of several tasks you can formulate as a sequence-to-sequence problem, a powerful framework for returning some output from an input, like translation or summarization. Translation systems are commonly used for translation between different language texts, but it can also be used for speech or some combination in between like text-to-speech or speech-to-text.

This guide will show you how to:

1. **Finetune [T5](https://huggingface.co/google-t5/t5-small) on the English-French subset of the [OPUS Books](https://huggingface.co/datasets/opus_books) dataset to translate English text to French.**
2. **Use your finetuned model for inference.**

<Tip>
The task illustrated in this tutorial is supported by the following model architectures:

[BART](../model_doc/bart), [BigBird-Pegasus](../model_doc/bigbird_pegasus), [Blenderbot](../model_doc/blenderbot), [BlenderbotSmall](../model_doc/blenderbot-small), [Encoder decoder](../model_doc/encoder-decoder), [FairSeq Machine-Translation](../model_doc/fsmt), [GPTSAN-japanese](../model_doc/gptsan-japanese), [LED](../model_doc/led), [LongT5](../model_doc/longt5), [M2M100](../model_doc/m2m_100), [Marian](../model_doc/marian), [mBART](../model_doc/mbart), [MT5](../model_doc/mt5), [MVP](../model_doc/mvp), [NLLB](../model_doc/nllb), [NLLB-MOE](../model_doc/nllb-moe), [Pegasus](../model_doc/pegasus), [PEGASUS-X](../model_doc/pegasus_x), [PLBart](../model_doc/plbart), [ProphetNet](../model_doc/prophetnet), [SeamlessM4T](../model_doc/seamless_m4t), [SeamlessM4Tv2](../model_doc/seamless_m4t_v2), [SwitchTransformers](../model_doc/switch_transformers), [T5](../model_doc/t5), [UMT5](../model_doc/umt5), [XLM-ProphetNet](../model_doc/xlm-prophetnet)

</Tip>

Before you begin, make sure you have all the necessary libraries installed:

```bash
pip install transformers datasets evaluate sacrebleu
```

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Load OPUS Books dataset

Start by loading **the English-French subset** of the [OPUS Books](https://huggingface.co/datasets/opus_books) dataset from the 🤗 Datasets library:

```py
>>> from datasets import load_dataset

>>> books = load_dataset("opus_books", "en-fr")
```

Split the dataset into a train and test set with the [`~datasets.Dataset.train_test_split`] method:

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

Then take a look at an example:

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`: an English and French translation of the text.

## Preprocess

The next step is to load **a T5 tokenizer** to process the English-French language pairs:

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

The preprocessing function you want to create needs to:

1. **Prefix the input with a prompt so T5 knows this is a translation task.** Some models capable of multiple NLP tasks require prompting for specific tasks.
2. **Tokenize the input (English) and target (French) separately because you can't tokenize French text with a tokenizer pretrained on an English vocabulary.**
3. **Truncate sequences to be no longer than the maximum length set by the `max_length` parameter.**

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

To apply the preprocessing function over the entire dataset, use 🤗 Datasets [`~datasets.Dataset.map`] method. You can speed up the `map` function by setting `batched=True` to process multiple elements of the dataset at once:

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

Now create a batch of examples using [`DataCollatorForSeq2Seq`]. It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## Evaluate

Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

**Then create a function that passes your predictions and labels to [`~evaluate.EvaluationModule.compute`] to calculate the SacreBLEU score:**

```py
>>> import numpy as np


>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.

## Train

You're ready to start training your model now! Load **T5** with [`AutoModelForSeq2SeqLM`]:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

At this point, only three steps remain:

1. Define your training hyperparameters in [`Seq2SeqTrainingArguments`]. **The only required parameter is `output_dir` which specifies where to save your model.** You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). **At the end of each epoch, the [`Trainer`] will evaluate the SacreBLEU metric and save the training checkpoint.**
2. Pass the training arguments to [`Seq2SeqTrainer`] along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
3. Call [`~Trainer.train`] to finetune your model.

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     evaluation_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True,
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     tokenizer=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

Once training is completed, share your model to the Hub with the [`~transformers.Trainer.push_to_hub`] method so everyone can use your model:

```py
>>> trainer.push_to_hub()
```

<Tip>

For a more in-depth example of how to finetune a model for translation, take a look at the corresponding
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb)
or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation-tf.ipynb).

</Tip>

## Inference

Great, now that you've finetuned a model, you can use it for inference!

Come up with some text you'd like to translate to another language. **For T5, you need to prefix your input depending on the task you're working on.** For translation from English to French, you should prefix your input as shown below:

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

The simplest way to try out your finetuned model for inference is to use it in a [`pipeline`]. Instantiate a `pipeline` for **translation** with your model, and pass your text to it:

```py
>>> from transformers import pipeline

>>> translator = pipeline("translation", model="my_awesome_opus_books_model")
>>> translator(text)
[{'translation_text': 'Legumes partagent des ressources avec des bactéries azotantes.'}]
```

You can also manually replicate the results of the `pipeline` if you'd like:

Tokenize the text and return the `input_ids` as PyTorch tensors:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

Use the [`~transformers.generation_utils.GenerationMixin.generate`] method to create the translation. For more details about the different text generation strategies and parameters for controlling generation, check out the [Text Generation](../main_classes/text_generation) API.

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

Decode the generated token ids back into text:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```