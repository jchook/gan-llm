Hacker News Data
================

Hacker News has a reasonably good signal-to-noise ratio compared to other
internet-based, human-written text sources.

It's also ridiculously easy to scrape (in 2024) using the firebase API.

Everything is an "item" (polls, stories, comments, etc). Items appear to have
sequential IDs, and are all accessible from the same endpoint. So it becomes
possible to simply iterate through each item and save it out to a JSONL file.

See [fetch.sh](./fetch.sh).


Data Cut-Off
------------

We are only interested in pre-2022 data (around ID 30000000).

GPT etc became widely used and available after that time. See [Wordfreq
SUNSET.md](https://github.com/rspeer/wordfreq/blob/master/SUNSET.md) for an
expert opinion on this topic.

We'll arbitrarily use a post in August 2012 as the start ID (4380000). That's
when I signed-up for Hacker News and started reading it on a daily basis, and
I imagine it started to get a lot more popular around that time.

Between these "goal posts", we have 25,620,000 items to fetch.


Other Sources
-------------

Other folks have already scraped HN and provided datasets.

- https://huggingface.co/datasets/OpenPipe/hacker-news - full data until 2023, 10GB
- https://huggingface.co/datasets/OpenPipe/best-hn-comment-pairs-v2 - 36k rows of selected comments
- https://github.com/sytelus/HackerNewsData - old, only to 2014
- https://huggingface.co/datasets/jkeisling/hacker-news-corpus-2007-2022


Resources
---------

- [Entity Conversion](https://www.crummy.com/software/BeautifulSoup/bs3/documentation.html#Entity%20Conversion)

