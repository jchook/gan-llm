Hacker News Data
================

Hacker News has a reasonably good signal-to-noise ratio compared to other
internet-based, human-written text sources.

It's also ridiculously easy to scrape (in 2024) using the firebase API.

Everything is an "item" (polls, stories, comments, etc). Items appear to have
sequential IDs, and are all accessible from the same endpoint. So it becomes
possible to simply iterate through each item and save it out to a JSONL file.

Example API URL:
https://hacker-news.firebaseio.com/v0/item/4388729.json?print=pretty

See [fetch.sh](./fetch.sh).


Data Timeframe
--------------

We are only interested in pre-2022 data (around ID 30000000). GPT etc became
widely used and available after that time. See [Wordfreq
SUNSET.md](https://github.com/rspeer/wordfreq/blob/master/SUNSET.md) for an
expert opinion on this topic.

So which range of posts should we scrape?

We could use a post in August 2012 as the start ID (4380000). That's when
I signed-up for Hacker News and started reading it on a daily basis, and
I imagine it started to get a lot more popular around that time. However, this
large time range (2012-2022) yields 25,620,000 "items" to fetch.

If we estimate roughly 5% of the items will be usable, that would yield over
1 million usable examples. That's far more than we need. Also, the HN comments
of 2012 have a certain "exclusive hacker" vibe that I think softens over the
years, making later comments more broad in topic and flavor.

Targeting roughly 50k usable posts, we will need a window of 1 million items.
Since 2020-2022 are innundated with arguments about COVID, I would prefer to
steer clear of that timeframe entirely.

Item ID 20000000 to 22000000 gives us May 2019 to January 2020, and doubles
the total items we estimate will be needed. That timeframe seems good for this
task.


Observations
------------

HN comments do not close `<p>` tags. They simply separate paragraphs with `<p>`
as if it were `<br><br>`.

A surprising number of commenters add a double-space between sentences.

At some point some of the HTML entity encodings switched from numeric to common
names. For example, `&#62;` became `&gt;`.

The HTML encoding seems clean enough that one can confidently strip HTML tags
with a simple `re.sub('<[^<]+?>', '', text)` after replacing `<p>` with `\n\n`.


Other Sources
-------------

Other folks have already scraped HN and provided datasets.

- https://huggingface.co/datasets/OpenPipe/hacker-news - full data until 2023, 10GB
- https://huggingface.co/datasets/OpenPipe/best-hn-comment-pairs-v2 - 36k rows of selected comments
- https://github.com/sytelus/HackerNewsData - old, only to 2014
- https://huggingface.co/datasets/jkeisling/hacker-news-corpus-2007-2022

Quasi-related but not useful for this task:

- https://huggingface.co/datasets/julien040/hacker-news-posts - might help you predict whether a post would be successful


Rewriting with AI
-----------------

After a few iterations, it seems that cycling through numerous different
prompts could produce a wider variety of machine-generated output. For example,
prompts should include/exclude various features such as:

- Double-space between sentences
- Include a couple innocuous spelling mistakes to make it seem more natural
- Use all lowercase
- Capture the writing style of the original author

All prompts should include some minimum guards against common AI output junk:

- Do not include a preamble or post-text. Only output the rewritten [comment|essay|whatever].
- Do not write "In conclusion"
- Do not plagiarize the content

Balancing the dataset between human and machine written examples makes a lot of
common problems with training a detection AI more apparent, so a 50/50 split
tends to be a good target.

Ideally the process will use a variety of LLMs to generate the
machine-generated examples, including cutting-edge commercial models like
ChatGPT 4o, Claude, and Grok.

Here are some example prompts I am considering for instructing Llama 3.2 or
ChatGPT 4o to rewrite the human-written comments.

> Rewrite this Hacker News comment in your own words. Try to sound as human as
> possible and capture all of the essence and style of the comment without
> plagiarizing it. Do not write "In conclusion". Do not add a preamble or post
> text; only output the rewritten comment.

> Rewrite this Hacker News comment in your own words. Try to preserve the
> author's writing style and essence without plagiarizing it or keeping the
> exact same word choice. Try to mimic any of the author's typing quirks such
> as innocuous misspellings, spacing, punctuation preferences, etc. Only output
> the rewritten comment.

---

Here we can see some as a system prompt rather than a user prompt:

> Assume the role of Hacker News comment rewriter. Rewrite Hacker News comments
> in your own words. Try to preserve the author's writing style and essence
> without plagiarizing it or keeping the same word choice. Try to occasionally
> mimic the author's typing quirks such as innocuous misspellings, spacing,
> punctuation preferences, etc., but do not copy the comment structure exactly.
> Only output the rewritten comment.

---

The next one emits output that kinda freakily sounds like a human. However, it
relies heavily on real human input. It highlights (lol) two key but different
use cases of this overall project: detecting AI-generated works vs AI-processed
works.

In the case of AI-generated works, human content may not be directly available
to guide the AI into writing like a human. In the AI-processed case, it will be
available. However, it seems most AI-processed work would like to benefit from
a stronger rewrite of the original data to make it sound more correct,
professional, succinct, accurate, well-written, etc.

> Assume the role of Hacker News comment rewriter. Completely rewrite Hacker News comments
> in your own words. Try to preserve the author's writing style and essence
> without plagiarizing it or keeping the same word choice. Try to mimic any of
> the author's typing quirks such as innocuous misspellings, spacing,
> punctuation preferences, etc. Only output the rewritten comment.

Here is a slightly imroved version of this:

> Assume the role of Hacker News comment rewriter. Rewrite this Hacker News
> comment in your own words. Try to preserve the author's writing style and
> essence without plagiarizing it or keeping the exact same word choice. Try to
> capture the author's writing quirks such as innocuous misspellings, spacing,
> punctuation preferences, sentence structure, mood, formality/informality,
> attitude, etc. Only output the rewritten comment. Here is the comment to
> rewrite:

---

Here is one that ChatGPT came up with. It doesn't seem to perform as well,
especially since it causes the output to have typical AI-like punctuation.

> Rewrite the following text, preserving its unique feel and personality without using the exact words or phrasing. Focus on capturing the writer’s *voice* and *expressiveness* so that the rewritten text still feels like it could have been written by them.
>
> Pay close attention to:
> 1. **Cadence and Rhythm**: Keep the pacing and flow of sentences similar, reflecting whether the writing has a quick, choppy style or a more measured, flowing one.
> 2. **Syntax Patterns**: Emulate the arrangement of phrases and clauses; if the writer tends toward complex structures, reflect that complexity, or if they prefer simplicity, echo that instead.
> 3. **Register and Diction Choices**: Match the level of formality or informality, from conversational to technical. Capture any preference for certain types of words, like straightforward vocabulary or more nuanced, descriptive language.
> 4. **Colloquialism and Phrasing Style**: Use conversational language or idioms where they appear, and incorporate similar expressions or descriptors while avoiding direct copies.
> 5. **Structural Emphasis**: Keep any unique use of punctuation—such as ellipses, em-dashes, or italics—that creates emphasis and rhythm, and try to use similar patterns to preserve the emotional weight of each point.
> 6. **Conciseness or Verbosity**: Match how much detail the writer includes, whether they’re concise or enjoy tangents and asides.
> 7. **Nuance and Implication**: Emulate how the writer hints at meaning, whether they are understated or direct, humorous or serious, playful or detached.
>
> Your goal is to recreate the *essence*, *tone*, and *mood* of the original while generating fresh language. Use these aspects to craft a new version that is legally distinct but carries the same personality and intention.


---

And combining them:

> Assume the role of Hacker News comment rewriter. Rewrite this Hacker News
> comment. Try to preserve the author's writing style and essence without
> plagiarizing it or keeping the exact same word choice. Try to capture the
> author's writing quirks such as innocuous misspellings, spacing, punctuation
> preferences, sentence structure, mood, formality/informality, attitude,
> cadence, rhythm, register, nuance/implication, verbosity (or lack thereof),
> etc. Only output the rewritten comment. Here is the comment to rewrite:


---

This one creates very AI-esque output, presumably because it doesn't have any
real human text to work from. This might be a good strategy for the GAN arch.

> Assume the role of a Hacker News discussion user. Reply intelligently to the
> provided comment in the style of a Hacker News comment section. Do not copy
> the provided comment at all, buttry to sound as real and human as possible.
> Reply using approximately as many words as the provided comment, with an
> original thought that directly addresses the provided comment. Only output
> the reply to the comment.


Resources
---------

- [Entity Conversion](https://www.crummy.com/software/BeautifulSoup/bs3/documentation.html#Entity%20Conversion)

