import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from markdownify import markdownify as md

book = epub.read_epub('example.epub')

for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        print(md(item.get_content()))

