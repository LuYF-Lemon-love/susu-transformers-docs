<!--
# docs/source/en/accelerate.md
# 
# git pull from huggingface/transformers by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Mar 22, 2024
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Mar 24, 2024
# 
# 该文档介绍了🤗 加速分布式训练。
-->

# Distributed training with 🤗 Accelerate

In this tutorial, learn **how to customize your native PyTorch training loop to enable training in a distributed environment**.

## Setup

Get started by installing 🤗 **Accelerate**:

```bash
pip install accelerate
```

Then import and create an [`~accelerate.Accelerator`] object. The [`~accelerate.Accelerator`] will automatically detect your type of distributed setup and initialize all the necessary components for training. **You don't need to explicitly place your model on a device.**

```py
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
```

## Prepare to accelerate

The next step is to pass all the relevant training objects to the [`~accelerate.Accelerator.prepare`] method. This includes your **training and evaluation DataLoaders**, **a model** and **an optimizer**:

```py
>>> train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
...     train_dataloader, eval_dataloader, model, optimizer
... )
```

## Backward

The last addition is to replace the typical `loss.backward()` in your training loop with 🤗 Accelerate's [`~accelerate.Accelerator.backward`]method:

```py
>>> for epoch in range(num_epochs):
...     for batch in train_dataloader:
...         outputs = model(**batch)
...         loss = outputs.loss
...         accelerator.backward(loss)

...         optimizer.step()
...         lr_scheduler.step()
...         optimizer.zero_grad()
...         progress_bar.update(1)
```

As you can see in the following code, you only need to **add four additional lines of code** to your training loop to enable distributed training!

```diff
+ from accelerate import Accelerator
  from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```

## Train

Once you've added the relevant lines of code, launch your training in a script or a notebook like Colaboratory.

### Train with a script

If you are running your training from a script, run the following command to **create and save a configuration file**:

```bash
accelerate config
```

Then launch your training with:

```bash
accelerate launch train.py
```

### Train with a notebook

🤗 Accelerate can also run in a notebook if you're planning on using Colaboratory's TPUs. **Wrap all the code responsible for training in a function**, and pass it to [`~accelerate.notebook_launcher`]:

```py
>>> from accelerate import notebook_launcher

>>> notebook_launcher(training_function)
```

For more information about 🤗 Accelerate and its rich features, refer to the [documentation](https://huggingface.co/docs/accelerate).
