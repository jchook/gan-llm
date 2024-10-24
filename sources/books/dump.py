import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

book = epub.read_epub('example.epub')

for item in book.get_items():
    if item.get_type() == ebooklib.ITEM_DOCUMENT:
        print('==================================')
        print('NAME : ', item.get_name())
        print('----------------------------------')
        print(item.get_content())
        print('==================================')
