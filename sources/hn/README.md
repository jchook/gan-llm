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

HN comments do not close their <p> tags. They simply separate paragraphs with
`<p>` as if it were an HTML5 `<br><br>`

A surprising number of users double-space between sentences. Should this be
removed to avoid teaching the machine that double-spacing = human?

At some point some of the HTML entity encodings switched from numeric to common
names. For example, `&#62;` became `&gt;`. Use `html.unescape()`.

The HTML encoding seems clean enough that I believe we can strip tags with
a simple `re.sub('<[^<]+?>', '', text)` after replacing `<p>` with `\n\n`.


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

Here are some example prompts I am considering for instructing Llama 3.2 or
ChatGPT 4o to rewrite the human-written comments.

> Rewrite this Hacker News comment in your own words. Try to sound as human as
> possible and capture all of the essence and style of the comment without
> plagiarizing it. Do not write "In conclusion". Do not add a preamble or post
> text; only output the rewritten comment.


Resources
---------

- [Entity Conversion](https://www.crummy.com/software/BeautifulSoup/bs3/documentation.html#Entity%20Conversion)

