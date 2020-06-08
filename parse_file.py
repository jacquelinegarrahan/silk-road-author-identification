from html.parser import HTMLParser
from bs4 import BeautifulSoup
import re
import os
import csv


WORD_REPLACEMENTS = [
    ".",
    "a ",
    " as ",
    " about ",
    " above ",
    " am ",
    " again ",
    " after ",
    " all ",
    " an ",
    " at ",
    " any ",
    " are ",
    " be ",
    " before ",
    " been ",
    " because ",
    " being ",
    " below ",
    " between ",
    " both ",
    " but ",
    " by ",
    " could ",
    " did ",
    " do ",
    " does ",
    "don't",
    " doing ",
    " down ",
    " during ",
    " each ",
    " few",
    " for ",
    " from ",
    " had",
    " has ",
    " have ",
    " having ",
    " he ",
    " he'd ",
    " he'll ",
    " he's",
    " her",
    " here",
    " here's",
    " hers",
    " after",
    " herself",
    " him ",
    " himself ",
    " his ",
    " how ",
    " how's ",
    "i ",
    " i'd ",
    "i'm ",
    "i'll ",
    "i've ",
    " if ",
    " in ",
    " into ",
    " is ",
    " it ",
    "it's ",
    "its ",
    " itself ",
    " let's ",
    " me ",
    " more ",
    " most ",
    " myself ",
    " my ",
    " nor ",
    " of ",
    " on ",
    " once ",
    " only ",
    " or ",
    " other ",
    " our ",
    " ours ",
    " ourselves ",
    " out ",
    " over ",
    " own ",
    " same ",
    " she ",
    " she'd ",
    " she'll ",
    " she's ",
    " should ",
    " so ",
    " some ",
    " such ",
    " than",
    " that ",
    " that's ",
    "the ",
    " their ",
    " theirs ",
    " them ",
    " themseves ",
    " then ",
    " there ",
    " there's ",
    " these ",
    " they ",
    " they'd ",
    " they'll ",
    " they're ",
    " they've ",
    "this ",
    " those ",
    " through ",
    " to ",
    " too ",
    " under ",
    " until ",
    " up ",
    " very ",
    " was ",
    " we ",
    " we'd ",
    " we'll ",
    " we're ",
    " we've ",
    " were ",
    " what ",
    " what ",
    " what's ",
    " when ",
    " when's ",
    " where ",
    " where's ",
    " which ",
    " while ",
    " who ",
    " who's ",
    " whom ",
    " why ",
    " why's ",
    " with ",
    " would ",
    " you ",
    " you'd ",
    " you'll ",
    " you're ",
    " you've ",
    " your ",
    " yours ",
    " yourself ",
    " yourselves ",
    " can ",
    " not ",
    " and ",
    " if ",
    "''",
    "   ",
    "\t",
    "\r",
    "\n",
]



TWITTER_GLOVE = [25, 50, 100, 200]
WIKIPEDIA_GLOVE = [50, 100, 200, 300]

def parse_forum_html(file_name: str) -> dict:
    """
    Extracts all users from a silk road forum page, and their posts
    
    Paramters
    ---------
    file_name: str
        File name to parse
    
    """
    file = open(file_name, "r")
    soup = BeautifulSoup(file, "html.parser")

    post_headers = soup.find_all("dt")  # titles contain: title, post by, and time
    for div in soup.find_all("a"):
        div.decompose()

    post_bodys = soup.find_all("dd")

    for i, post in enumerate(post_headers):
        text = post.get_text()
        info = text[text.find("Post by:") :]
        date_idx = info.find("on")
        author = info[len("Post by: ") : date_idx - 1]
        author = author.strip()

        body = post_bodys[i]
        for div in body.find_all("blockquote", {"class": "bbc_standard_quote"}):
            div.decompose()
        body_text = body.get_text().strip().lower()
        body_text = re.sub("[^' a-zA-Z]", " ", body_text)
        for replacement in WORD_REPLACEMENTS:
            body_text = body_text.replace(replacement, " ")

        posts.append({"author": author, "body_text": body_text})

    return posts



if __name__ == "__main__":
    dirpath = "test_files"
    file_list = os.listdir(dirpath)
    posts = []
    with open("posts.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["author", "post"])
        for f in range(len(file_list)):
            print(f"{f} of {len(file_list)}")
            posts = parse_forum_html(dirpath + "/" + file_list[f])
            for post in posts:
                writer.writerow([post["author"], post["body_text"]])


    #df.to_pickle(file_name)