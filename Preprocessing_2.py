#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score as precision
from sklearn.metrics import accuracy_score as accuracy
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import csv
import io
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import spacy 
from spacy_langdetect import LanguageDetector
import time


# In[4]:


def alphanumeric():
    alpha = string.ascii_lowercase
    numeric = string.digits
    
    alphanumeric_list = []
    for char in alpha:
        alphanumeric_list.append(char)
    
    for char in numeric:
        alphanumeric_list.append(char)
        
    alphanumeric_list.append("'")
    alphanumeric_list.append("-")
    
    return alphanumeric_list


# In[5]:


def generate_stopwords():
    EN_STOPWORDS = {'recently', 'soon', 'past', 'now', 'recent', 'previously', 'quickly', "'d", "'ll", "'m", "'re", "'s", "'ve", "a", "a's", "able", "about", "above", "abst", "accordance", "according", "accordingly", "across", "act", "acts", "acting", "actually", "add", "adds", "added", "adding", "adj", "ae", "af", "affect", "affected", "affecting", "affectingly", "affects", "after", "afterward", "afterwards", "ag", "again", "against", "ago", "ah", "ain't", "aint", "all", "allow", "allows", "allowing", "allowed", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "announced", "announcing", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "apart", "apparent", "apparently", "appear", "appears", "appearing", "appeared", "appreciate", "appreciating", "appreciates", "appreciated", "appropriate", "appropriately", "approximately", "aq", "ar", "are", "aren", "aren't", "arent", "around", "as", "aside", "ask", "asks", "asked", "asking", "associated", "at", "au", "auth", "available", "aw", "away", "az", "b", "ba", "back", "bb", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bf", "bg", "bh", "biol", "bj", "bm", "bn", "bo", "both", "bottom", "br", "brief", "briefly", "bs", "bt", "but", "bv", "bw", "by", "bz", "c", "c'mon", "c's", "ca", "call", "came", "can", "can't", "cannot", "cant", "cause", "causes", "cc", "certain", "certainly", "cf", "cg", "ch", "changes", "change", "changing", "changed", "ci", "ck", "cl", "clearly", "cm", "cn", "come", "comes", "coming", "consequently", "consider", "considering", "consideringly", "contain", "containing", "contains", "contained", "corresponding", "correspondingly", "could", "couldn", "couldn't", "couldnt", "cr", "cs", "cu", "currently", "cv", "cx", "cy", "cz", "d", "de", "deci", "definitely", "definite", "describe", "described", "describing", "despite", "detail", "detailing", "details", "detailed", "did", "didn", "didn't", "didnt", "different", "direct", "directly", "dj", "dk", "dm", "do", "does", "doesn", "doesn't", "doesnt", "doing", "don't", "done", "dont", "down", "downwards", "due", "during", "dz", "e", "each", "ec", "ed", "edu", "ee", "effect", "eg", "eh", "eight", "eighty", "either", "eleven", "else", "elsewhere", "empty", "end", "ending", "ended", "enough", "entire", "entirely", "er", "es", "especially", "et", "et-al", "etc", "even", "ever", "evermore", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "fairly", "far", "farther", "few", "fewer", "ff", "fi", "fill", "find", "first", "five", "fix", "fj", "fk", "fo", "followed", "following", "follows", "follow", "for", "forever", "former", "formerly", "forth", "forty", "forward", "found", "four", "fr", "from", "front", "full", "further", "furthermore", "fx", "g", "ga", "gave", "gb", "gd", "ge", "get", "gets", "getting", "gf", "gg", "gh", "gi", "giga", "give", "given", "gives", "giving", "gl", "gm", "gn", "go", "goes", "going", "gone", "got", "gotten", "gp", "gq", "gr", "greetings", "greets", "greet", "greeted", "gs", "gt", "gu", "gw", "gy", "h", "had", "hadn't", "hadnt", "happen", "happened", "happening", "happens", "hardly", "has", "hasn", "hasn't", "hasnt", "have", "haven't", "havent", "having", "he", "he'd", "he'll", "he's", "hed", "hello", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "heres", "hereupon", "hers", "her's", "herself", "hes", "hi", "hid", "him", "himself", "his", "hk", "hm", "hn", "hopefully", "how", "how's", "hows", "howbeit", "however", "hr", "ht", "hu", "hundred", "i", "i'd", "i'll", "i'm", "i've", "id", "ie", "if", "million", "billion", "trillion", "thousand", "ten", "ignored", "ignore", "ignoring", "ignores", "ii", "il", "im", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "indicating", "information", "inner", "insofar", "instead", "inter", "interest", "into", "intra", "inward", "ir", "is", "isn", "isn't", "isnt", "it", "it'd", "it'll", "it's", "itd", "itll", "its", "itself", "ive", "j", "je", "jm", "jo", "jp", "just", "k", "ke", "keep", "keeps", "kept", "keeping", "kg", "kh", "ki", "km", "kn", "knew", "know", "knowing", "known", "knows", "kp", "kr", "kw", "ky", "kz", "l", "la", "largely", "last", "lastly", "lately", "later", "latter", "latterly", "lb", "lc", "least", "less", "lest", "let", "let's", "lets", "li", "like", "liked", "likely", "likewise", "lk", "ll", "look", "looked", "looking", "looks", "lr", "ls", "lt", "ltd", "lu", "lv", "m", "ma", "made", "main", "mainly", "make", "makes", "many", "may", "maybe", "mayn't", "maynt", "mc", "md", "me", "mean", "means", "meantime", "meanwhile", "mega", "merely", "mg", "mh", "micro", "might", "mightn't", "mightnt", "mili", "mine", "mini", "mk", "ml", "mm", "mn", "mo", "more", "moreover", "most", "mostly", "move", "moving", "moved", "moves", "mp", "mq", "mr", "mrs", "ms", "mt", "much", "mug", "must", "mustn't", "mustnt", "mv", "mw", "mx", "my", "myself", "mz", "n", "n't", "na", "name", "namely", "named", "names", "nano", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needless", "needlessly", "needn't", "neednt", "needs", "neither", "never", "nevertheless", "new", "next", "nf", "ng", "ni", "nine", "ninety", "nl", "no", "no-one", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "notwithstanding", "nowhere", "np", "nr", "nz", "o", "obtain", "obtained", "obtains", "obtaining", "obviously", "obvious", "of", "often", "oh", "ok", "okay", "old", "omit", "omitted", "once", "one", "one's", "ones", "only", "onto", "or", "ord", "other", "others", "otherwise", "ought", "oughtn't", "oughtnt", "our", "ours", "ourselves", "out", "outside", "over", "overall", "owing", "own", "p", "pa", "page", "pages", "part", "particular", "particularly", "pe", "per", "perhaps", "peta", "pf", "pg", "ph", "pk", "pl", "placed", "please", "plus", "pm", "pn", "poorly", "possible", "possibly", "potentially", "pp", "pr", "predominantly", "presumably", "primarily", "probably", "promptly", "provides", "provide", "providing", "pt", "put", "pw", "py", "q", "qa", "que", "quite", "qv", "r", "rather", "rd", "re", "readily", "really", "reasonably", "reasoning", "reason", "reasons", "reasoned", "ref", "refs", "regard", "regarding", "regardless", "regards", "related", "relate", "relates", "relating", "relatively", "research", "researching", "researched", "respectively", "result", "resulted", "resulting", "results", "right", "rightly", "rights", "ro", "ru", "rw", "s", "sa", "said", "same", "saw", "say", "saying", "says", "sb", "sc", "sd", "se", "sec", "second", "secondly", "section", "sections", "sectioning", "sectioned", "see", "seeing", "sees", "seem", "seemed", "seeming", "seemingly", "seems", "seen", "self", "selves", "sensible", "sensibly", "sent", "serious", "seriously", "seven", "several", "sg", "sh", "shall", "shan't", "shant", "she", "she'd", "she'll", "she's", "shed", "sheds", "shedding", "shes", "should", "shouldn", "shouldn't", "shouldnt", "show", "showed", "showing", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "sincerely", "six", "sj", "sk", "sl", "slight", "slightly", "slights", "sm", "sn", "so", "some", "somebody", "someday", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "specifically", "specified", "specifies", "specify", "specifying", "sr", "st", "still", "stop", "stops", "stopping", "stopped", "strongly", "su", "sub", "substantially", "substantial", "successfully", "such", "sufficient", "sufficiently", "suggest", "sup", "sure", "surely", "sv", "sy", "sz", "t", "t's", "take", "takes", "taken", "taking", "tc", "td", "tell", "tells", "telling", "told", "tend", "tendency", "tends", "tending", "tera", "tf", "tg", "th", "than", "thank", "thanks", "thanking", "thanx", "thankfully", "that", "that'll", "that's", "that've", "thatll", "thats", "thatve", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "there'd", "there'll", "there're", "there's", "there've", "thereafter", "thereby", "thered", "therefore", "therein", "therell", "thereof", "therere", "theres", "thereto", "thereupon", "thereve", "these", "they", "they'd", "they'll", "they're", "they've", "theyd", "theyll", "theyre", "theyve", "think", "third", "thirdly", "firstly", "thought", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "three", "throug", "through", "throughout", "thru", "thus", "til", "till", "tip", "tj", "tk", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tr", "tried", "tries", "truly", "try", "trying", "ts", "tt", "tw", "thrice", "twice", "two", "tz", "oneself", "u", "ua", "ug", "uk", "um", "un", "under", "unfortunate", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "up", "upon", "ups", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "uy", "uz", "v", "va", "various", "vc", "ve", "versus", "very", "vg", "vi", "via", "viz", "vn", "vol", "vols", "vs", "vu", "w", "want", "wants", "wanting", "wanted", "was", "wasn", "wasn't", "wasnt", "way", "we", "we'd", "we'll", "we're", "we've", "wed", "welcome", "well", "went", "were", "weren't", "werent", "weve", "wf", "what", "what'll", "what's", "what've", "whatever", "whatll", "whats", "whatve", "when", "when's", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "whichever", "while", "whiled", "whilst", "whim", "whither", "who", "who'd", "who'll", "who's", "whod", "whoever", "whole", "wholl", "whom", "whomever", "whos", "whose", "why", "why's", "widely", "will", "wills", "willing", "willed", "wish", "wishes", "wishing", "wished", "with", "within", "without", "won't", "wonder", "wont", "woul", "would", "wouldn", "wouldn't", "wouldnt", "ws", "x", "y", "ye", "yes", "yet", "you", "you'd", "you'll", "you're", "you've", "youd", "youll", "your", "youre", "yours", "yourself", "yourselves", "youve", "z", "zero"}
        
    EN_OPENWEB_STOPWORDS = {"requests", "requesting", "requested", "request", "access", "accessing", "accesses", "accessed", "ad", "add", "adds", "adding", "add-on", "addon", "address", "addresses", "addressed", "addressing", "adn", "ads", "adsl", "advertise", "advertised", "advertisement", "advertising", "agent", "ai", "analyse", "analysis", "analyze", "analyzed", "anonymous", "api", "app", "applet", "application", "archie", "arpanet", "article", "articles", "ascii", "asp", "auth", "authority", "back", "back-button", "backbutton", "band", "bandwidth", "base-name", "basename", "baud", "bbs", "binary", "binhex", "bit", "bitnet", "bits", "blog", "bloging", "blogging", "blogosphere", "blogsphere", "board", "bold", "bookmark", "bottom", "bounce", "bps", "broadband", "browse", "browsed", "browser", "browsers", "browsing", "btw", "button", "buttons", "buy", "byte", "bytes", "cable", "cache", "callback", "caps-lock", "capslock", "cart", "catp", "cdma", "certificate", "cgi", "cgi-bin", "cgibin", "chat", "chatroom", "chatting", "check-out", "checkout", "chrome", "click", "clickjacking", "clicks", "clickstream", "clicktivism", "client", "cntrl", "co", "co-location", "code", "colocation", "column", "columns", "com", "command", "command-line", "commandline", "comment", "comments", "commenting", "commerce", "computer", "computers", "connect", "connecting", "connects", "connection", "content", "cookie", "crawl", "crawled", "crawls", "crawler", "crawler-based", "crawlerbased", "crawling", "crowd", "crowdsource", "crowdsourcing", "crowdsourced", "css", "cyber", "cyber-space", "cybercafe", "cybershopping", "cyberspace", "cyberspaces", "data", "decimal", "delete", "description", "descriptions", "designer", "designers", "designer's", "dhcp", "dhtml", "dial", "dials", "dial-up", "dialup", "digerati", "digital", "directories", "directory", "dislike", "dislikes", "disliking", "disliked", "dns", "doc", "docs", "document", "documents", "documented", "documenting", "domain", "domain-name", "domainname", "dot", "dotcom", "download", "downloads", "downloading", "downloaded", "drop-down", "dropdown", "dsl", "dynamic", "e", "e-", "e-commerce", "e-learning", "e-signature", "e-tail", "e-wallet", "e-zine", "ecommerce", "edit", "edu", "elearning", "email", "emails", "emailing", "emailed", "emoji", "esc", "esignature", "etail", "ethernet", "ewallet", "extension", "extranet", "ezine", "faq", "fddi", "feed", "field", "file", "find", "fire-wall", "firefox", "firewall", "folksonomy", "footer", "form", "forms", "forward", "forward-button", "forwardbutton", "freenet", "ftp", "gateway", "gif", "giffy", "gify", "gigabyte", "gmail", "go-to", "google", "gopher", "goto", "gov", "gprs", "graphics", "hardware", "header", "heading", "headline", "help", "hexadecimal", "home", "home-button", "home-page", "homebutton", "homepage", "host", "host-name", "hosting", "hostname", "hot-link", "hot-list", "hotlink", "hotlist", "hotspot", "htm", "html", "http", "https", "hyper-text", "hypertext", "icon", "im", "image", "imap", "imho", "impression", "impressions", "inbox", "incognito", "index", "indexed", "input", "inputting", "inputs", "input-field", "inputfield", "instant", "instantly", "instance", "instants", "internet", "internet-explorer", "internetexplorer", "intranet", "io", "ip", "ip-number", "ipnumber", "ipv4", "ipv6", "irc", "isdn", "isp", "it", "italic", "java", "javascript", "jdk", "jpeg", "js", "keyword", "keywords", "kilobyte", "lan", "leased-line", "leasedline", "legend", "like", "likes", "liking", "liked", "line", "lines", "link", "linking", "linked", "links", "linux", "list", "lists", "listed", "listing", "listings", "listserv", "load", "loading", "loads", "loaded", "log", "log-in", "log-out", "logging", "logs", "logged", "login", "logo", "logout", "lower-case", "lowercase", "ly", "m-commerce", "m-learning", "mail-list", "mailing", "mailing-list", "mailinglist", "maillist", "markup", "maximize", "mcommerce", "media", "megabyte", "meme", "menu", "message", "messages", "messaged", "messaging", "messenger", "meta", "meta-data", "meta-field", "meta-tag", "meta-tags", "metadata", "metafield", "metatag", "metatags", "micro-blogging", "microblogging", "mime", "mimes", "miming", "minimize", "mlearning", "moblogging", "mod_perl", "modem", "moo", "mosaic", "mp3", "mp4", "msn", "mud", "nano-publishing", "nanopublishing", "navigate", "navigates", "navigation", "net", "net-speak", "netiquette", "netizen", "netscape", "netspeak", "network", "news", "news-article", "news-group", "news-reader", "newsarticle", "newsgroup", "newsreader", "next", "nic", "nntp", "node", "nodes", "online", "open-content", "open-source", "open-web", "opencontent", "opensource", "openweb", "optimisation", "optimization", "option", "options", "org", "otp", "ott", "outbox", "packet", "packets", "page", "page-break", "page-content", "page-object", "pagebreak", "pagecontent", "pageobject", "pages", "paragraph", "password", "pdf", "perl", "permalink", "petabyte", "photo", "php", "picture", "ping", "pixel", "plug-in", "plugin", "png", "pod-cast", "pod-casting", "podcast", "podcasting", "pop", "port", "portal", "positioning", "post", "posting", "ppp", "pptp", "previous", "protocol", "proxy", "pstn", "publish", "publisher", "publishing", "query", "quota", "quote", "quotes", "quoting", "quoted", "rank", "ranks", "ranked", "ranking", "rankings", "rate", "rating", "rates", "rated", "rdf", "refresh", "refreshes", "relay", "relays", "relaying", "relayed", "relevancy", "rest", "resting", "rests", "rested", "resubmit", "resubmits", "resubmitting", "resubmited", "results", "result", "return", "rfc", "roam", "roaming", "roams", "roamed", "router", "row", "rows", "rss", "rtsp", "rtt", "script", "scroll", "scroll-bar", "scroll-down", "scroll-up", "scrollbar", "scrolldown", "scrolling", "scrollup", "sdsl", "search", "search-engine", "searchengine", "select", "selection", "selects", "selecting", "selected", "sentence", "sentences", "seo", "server", "servlet", "session", "sessions", "sheet", "sheets", "shopping-cart", "shoppingcart", "short-url", "shortened-url", "shortenedurl", "shorturl", "sidebar", "site", "sites", "skype", "smds", "smtp", "snmp", "software", "source", "sources", "spam", "spams", "spamming", "speed", "spidered", "spread-sheet", "spreadsheet", "spreadsheets", "sql", "ssl", "start", "starts", "starting", "started", "stream", "streams", "streamed", "streaming", "submission", "submissions", "submit", "submitted", "submitting", "summary", "summaries", "surf", "surfs", "surfed", "surfing", "switch", "switches", "switching", "sysop", "t-1", "t-3", "t1", "t3", "tab", "tag", "tags", "tagged", "tagging", "tags", "talk-board", "talkboard", "task", "tcp", "telnet", "terabyte", "terminal", "terms", "text-field", "textfield", "thread", "time", "title", "tld", "top", "top-level", "toplevel", "traffic", "tv", "tweet", "udp", "unicode", "unix", "upload", "upper-case", "uppercase", "uri", "url", "urls", "urn", "usenet", "user", "users", "username", "usernames", "utf", "utf-8", "utf8", "uuencode", "video", "video-chat", "videochat", "view", "viewing", "viewed", "views", "vlog", "vlogs", "vlogging", "voip", "vpn", "wais", "wan", "wap", "web", "web-browser", "web-cast", "web-hosting", "web-page", "web-site", "webbrowser", "webcast", "webcasting", "webhosting", "webinar", "webmaster", "webmasters", "webpage", "website", "webzine", "white-page", "white-pages", "whitepage", "whitepages", "wi-fi", "wifi", "window", "windows", "wireless", "worm", "www", "xhtml", "xml", "xmlrpc", "xpfe", "xul", "yahoo", "zine"}
    
    EN_ALL_STOPWORDS = EN_STOPWORDS.union(EN_OPENWEB_STOPWORDS)
    
    '''load_spark_stopwords = StopWordsRemover.loadDefaultStopWords

    LANGS_STOPWORDS = {
        'da': set(load_spark_stopwords('danish')),
        'de': set(load_spark_stopwords('german')),
        'en': EN_STOPWORDS,
        'es': set(load_spark_stopwords('spanish')),
        'fi': set(load_spark_stopwords('finnish')),
        'fr': set(load_spark_stopwords('french')),
        'hu': set(load_spark_stopwords('hungarian')),
        'it': set(load_spark_stopwords('italian')),
        'nl': set(load_spark_stopwords('dutch')),
        'no': set(load_spark_stopwords('norwegian')),
        'pt': set(load_spark_stopwords('portuguese')),
        'ru': set(load_spark_stopwords('russian')),
        'sv': set(load_spark_stopwords('swedish')),
        'tr': set(load_spark_stopwords('turkish')),
    }

    ALL_LANG_STOPWORDS = set.union(*LANGS_STOPWORDS.values())
    ALL_STOPWORDS = ALL_LANG_STOPWORDS.union(EN_OPENWEB_STOPWORDS)

    LANGS_STOPWORDS['all_lang'] = ALL_LANG_STOPWORDS
    LANGS_STOPWORDS['web'] = EN_OPENWEB_STOPWORDS
    LANGS_STOPWORDS['en_web'] = EN_ALL_STOPWORDS
    LANGS_STOPWORDS['all'] = ALL_STOPWORDS'''
    
    return EN_ALL_STOPWORDS 


# In[6]:


stopwords = generate_stopwords()


# In[7]:


def preprocess_step1(my_string, stopwords):
    my_string = str(my_string)
    try:
        new_string = my_string.replace("’", "'").replace("–", "-").replace("‘", "'")
    except:
        new_string = my_string.encode('utf-8')
        new_string = new_string.replace("’", "'").replace("–", "-").replace("‘", "'")
    
    new_string = new_string.lower()
    
    for char in new_string:
        if ((not char.isalnum()) and (char != ' ')):
            new_string = new_string.replace(char, ' ')
    
    new_string_list = new_string.split()
    
    no_length_two = []
    for el in new_string_list:
        if len(el) > 2:
            no_length_two.append(el)
    
    no_digits = []
    for el in no_length_two:
        if (not ((el.isdigit()) or ((el.isalnum()) and (len(el) == 1)))):
            no_digits.append(el)
    
    no_stopwords = []
    
    for el in no_digits:
        if el not in stopwords:
            no_stopwords.append(el)
    
    return no_stopwords


# In[8]:


def preprocess_step2(data, stopwords):
    first_step = []
    count = 0
    start_time = time.time()
    for line in data:
        preprocess = preprocess_step1(line, stopwords)
        first_step.append(preprocess)
        count += 1
        if count % 1000 == 0:
            print(count)
            print("--- %s seconds ---" % (time.time() - start_time))
        
    return first_step


# In[9]:


def fit_count(train_data, stopwords):
    text = list(train_data['Text'])
    preprocessed_text = preprocess_step2(text, stopwords)
    
    flat = []
    for array in preprocessed_text:
        flat.append(' '.join(array))
    
    vectorizer = CountVectorizer(max_df = .8, min_df = 20)
    count_dict = vectorizer.fit_transform(flat)
    vocabulary = vectorizer.get_feature_names()
    vocabulary = set(vocabulary)
    
    return vocabulary


# In[10]:


def preprocess_step3(data, stopwords, vocabulary):
    labels = list(data['Labels'])
    text = list(data['Text'])
    second_round = preprocess_step2(text, stopwords)
    
    new_strings = []
    new_labels = []
    for i in range(len(second_round)):
        new_array = []
        for el in second_round[i]:
            if el in vocabulary:
                new_array.append(el)
        new_strings.append(new_array)
        new_labels.append(labels[i])

    final_strings = []
    for array in new_strings:
        joined = ' '.join(array)
        final_strings.append(joined)
        
    return final_strings, new_labels


# In[11]:


def preprocess_without_labels(data, stopwords, vocabulary):
    text = data
    second_round = preprocess_step2(text, stopwords)
    print('Done with preprocessing.')

    new_strings = []
    for i in range(len(second_round)):
        new_array = []
        for el in second_round[i]:
            if el in vocabulary:
                new_array.append(el)
        new_strings.append(new_array)
    print('Done removing vocabulary.')

    final_strings = []
    for array in new_strings:
        joined = ' '.join(array)
        final_strings.append(joined)
        
    return final_strings


# In[12]:


def get_labels_with_crowd(data):
    label_to_text_data = []
    for el in data:
        try:
            analyst_label = el['analyst_label'].lower()
        except:
            analyst_label = 'na'
        try:
            crowd_label = el['crowd_label'].lower()
        except:
            crowd_label = 'na'
        labels_to_text = {}

        if analyst_label != 'na':
            text = data_concat(el)
            labels_to_text[analyst_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
        elif crowd_label != 'na':
            text = data_concat(el)
            labels_to_text[crowd_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
    
    return label_to_text_data


# In[13]:


def get_labels_without_crowd(data):
    label_to_text_data = []
    for el in data:
        try:
            analyst_label = el['analyst_label'].lower()
        except:
            analyst_label = 'na'
        labels_to_text = {}

        if analyst_label != 'na':
            text = data_concat(el)
            labels_to_text[analyst_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
    
    return label_to_text_data


# In[84]:


def get_binary_labels_with_crowd(data, binary_label, cat):
    label_to_text_data = []
    for el in data:
        try:
            analyst_label = el['analyst_label'].lower()
        except:
            analyst_label = 'na'
        try:
            crowd_label = el['crowd_label'].lower()
        except:
            crowd_label = 'na'
        labels_to_text = {}

        if analyst_label != 'na':
            text = data_concat(el)
            if analyst_label == binary_label:
                labels_to_text[cat + binary_label] = text.encode('utf-8')
            else:
                labels_to_text[cat + '_not_' + binary_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
            
        elif crowd_label != 'na':
            text = data_concat(el)
            if crowd_label == binary_label:
                labels_to_text[cat + binary_label] = text.encode('utf-8')
            else:
                labels_to_text[cat + '_not_' + binary_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
    
    return label_to_text_data


# In[14]:


def get_binary_labels_without_crowd(data, binary_label, cat):
    label_to_text_data = []
    for el in data:
        try:
            analyst_label = el['analyst_label'].lower()
        except:
            analyst_label = 'na'
        labels_to_text = {}

        if analyst_label != 'na':
            text = data_concat(el)
            if analyst_label == binary_label:
                labels_to_text[cat + binary_label] = text.encode('utf-8')
            else:
                labels_to_text[cat + '_not_' + binary_label] = text.encode('utf-8')
            label_to_text_data.append(labels_to_text)
    
    return label_to_text_data


# In[15]:


def data_concat(line):
    ready_to_concat = []
    if line['url'] != None:
        ready_to_concat.append(line['url'])
    if line['title'] != None:
        ready_to_concat.append(line['title'])
    if line['pageText'] != None:
        ready_to_concat.append(line['pageText'])
    if line['metaKeywords'] != None:
        ready_to_concat.append(line['metaKeywords'])
    if line['metaDescription'] != None:
        ready_to_concat.append(line['metaDescription'])
    if line['metaSubject'] != None:
        ready_to_concat.append(line['metaSubject'])
        
    concat = ' '.join(ready_to_concat)
    
    return concat


# In[16]:


def pagetext_data(line):
    if line['pageText'] != None:
        return line['pageText']


# In[17]:


def convert_to_dataframe_for_xgb(data, labels):
    workable = []
    for el, label in zip(data, labels):
        el = list(el)
        workable.append([label] + el)
    
    df = pd.DataFrame(workable)
        
    return df


# In[18]:


def convert_to_dataframe(data, labels):
    workable = []
    for el, label in zip(data, labels):
        tmp = []
        tmp.append(label)
        tmp.append(el)
        workable.append(tmp)
    
    df = pd.DataFrame(workable, columns = ['Labels', 'Text'])
        
    return df


# In[19]:


def format_and_upload_data(data, file_name):
    formatted_data = []
    for el in data.values:
        cur_row = []
        label = '__label__' + el[1]
        cur_row.append(label)
        cur_row.append(el[0])
        formatted_data.append(cur_row)
        
    with open(file_name, 'w') as csvoutfile:
        csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
        csv_writer.writerows(formatted_data)


# In[20]:


def tfidf_fit(processed_text):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit(processed_text)
    
    return X


# In[21]:


def tfidf_transform(processed_text, transformer):
    vectors = transformer.transform(processed_text)
    return vectors.toarray()


# In[22]:


def logistic(train_data, test_data, train_labels, test_labels):
    logistic = LogisticRegression()
    logistic.fit(train_data, train_labels)
    predicted_labels = logistic.predict(test_data)
    score = precision(test_labels, predicted_labels, average=None, labels = ['low', 'moderate', 'high'])
    
    return logistic, score


# In[23]:


def fasttext_pipeline(train_data, test_data):
    format_and_upload_data(train_data, 'fasttext_train.txt')
    format_and_upload_data(test_data, 'fasttext_test.txt')
    
    classifier = fasttext.supervised('/Users/dnissani/Desktop/Test_SageMaker/fasttext_train.txt', 'model', label_prefix = '__label__')
    
    result = classifier.test('/Users/dnissani/Desktop/Test_SageMaker/fasttext_test.txt')
    
    return classifier, result
    


# In[24]:


def split_data(data, labels):
    train_data, td_data, train_labels, td_labels = train_test_split(data, 
                                                                labels, 
                                                                test_size=0.3, 
                                                                random_state=28, 
                                                                shuffle = True,
                                                                stratify = labels)
    
    test_data, dev_data, test_labels, dev_labels = train_test_split(td_data, 
                                                                td_labels, 
                                                                test_size=0.5, 
                                                                random_state=28, 
                                                                shuffle = True,
                                                                stratify = td_labels)
    
    return train_data, train_labels, test_data, test_labels, dev_data, dev_labels
    


# In[25]:


def grab_data_from_json(data, lang=None, cat=None):
    if (lang == None) and (cat == None):
        print('You have to choose a category or a language')
    
    elif (cat == None):
        lang_data = []
        for el in data:
            if el['language'] == lang:
                lang_data.append(el)
        
        return lang_data
    
    elif (lang == None):
        cat_data = []
        for el in data:
            if el['category'] == cat:
                cat_data.append(el)
        
        return cat_data
                
    else:
        lang_data = []
        for el in data:
            if el['language'] == lang:
                lang_data.append(el)
                
        cat_data = []
        for el in lang_data:
            if el['category'] == cat:
                cat_data.append(el)
        
        return cat_data


# # Getting Raw Text
# 
# This pipeline is designed to grab text from the json objects we get from hive tables. It grabs the text and puts them into dataframes to be read to .csv files. This generates labels and organizes data into raw text (concatenated from allow known text classes) and the labels provided by Analysts.

# In[71]:


start_time = time.time()

with io.open('/Users/dnissani/Desktop/Test_SageMaker/training_20190624.txt', 'r', encoding = 'utf-8') as f:
    data = []
    for line in f:
        data.append(json.loads(line))
            
            
print("--- %s seconds ---" % (time.time() - start_time))


# In[72]:


drug_data = grab_data_from_json(data, cat = 'vio')


# In[73]:


len(drug_data)


# In[85]:


without_crowd = get_binary_labels_with_crowd(drug_data, binary_label = 'high', cat = 'violence' )


# In[86]:


len(without_crowd)


# In[87]:


df_data = []
for el in without_crowd:
    line = []
    line.append(list(el.keys())[0])
    line.append(list(el.values())[0])
    df_data.append(line)


# In[88]:


df = pd.DataFrame(df_data, columns = ['label', 'text'])


# In[89]:


df['label'].value_counts()


# In[90]:


text = list(df['text'])


# In[91]:


len(text)


# In[92]:


labels = list(df['label'])


# In[93]:


train_data, train_labels, test_data, test_labels, dev_data, dev_labels = split_data(text, labels)


# In[94]:


train_df = convert_to_dataframe(train_data, train_labels)
test_df = convert_to_dataframe(test_data, test_labels)
dev_df = convert_to_dataframe(dev_data, dev_labels)


# In[96]:


train_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/violence_all_lang_binary_high_train.csv', index = False, encoding = 'utf-8')
test_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/violence_all_lang_binary_high_test.csv', index = False, encoding = 'utf-8')
dev_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/violence_all_lang_binary_high_dev.csv', index = False, encoding = 'utf-8')


# # Preprocess text
# 
# This pipeline is designed to preprocess text by removing special characters, stop words, and frequent and infrequent words. We use this pipeline to train tfidf features and word embedding models that we are experimenting with.

# In[ ]:


len(train_df) + len(test_df) + len(dev_df)


# In[ ]:


vocabulary = fit_count(train_df, [])


# In[ ]:


processed_train_text, processed_train_labels = preprocess_step3(train_df, [], vocabulary)
processed_test_text, processed_test_labels = preprocess_step3(test_df, [], vocabulary)
processed_dev_text, processed_dev_labels = preprocess_step3(dev_df, [], vocabulary)


# In[ ]:


len(processed_train_text) + len(processed_test_text) + len(processed_dev_text)


# In[ ]:


train_df = convert_to_dataframe(processed_train_text, processed_train_labels)
test_df = convert_to_dataframe(processed_test_text, processed_test_labels)
dev_df = convert_to_dataframe(processed_dev_text, processed_dev_labels)


# In[ ]:


set(processed_train_labels)


# In[ ]:


transformer = tfidf_fit(processed_train_text)
tfidf_train = tfidf_transform(processed_train_text, transformer)
tfidf_test = tfidf_transform(processed_test_text, transformer)
tfidf_dev = tfidf_transform(processed_dev_text, transformer)


# In[ ]:


model, score = logistic(tfidf_train, tfidf_test, processed_train_labels, processed_test_labels)


# In[ ]:


score


# In[ ]:


_, score = logistic(tfidf_train, tfidf_train, processed_train_labels, processed_train_labels)


# In[ ]:


score


# In[ ]:


nb = ComplementNB()
nb.fit(tfidf_train, processed_train_labels)
predicted_labels = nb.predict(tfidf_test)
print(precision(processed_test_labels, predicted_labels, average=None, labels=['low', 'moderate', 'high']))


# In[ ]:


train_df = convert_to_dataframe_for_xgb(tfidf_train, processed_train_labels)
test_df = convert_to_dataframe_for_xgb(tfidf_test, processed_test_labels)
dev_df = convert_to_dataframe_for_xgb(tfidf_dev, processed_dev_labels)


# In[ ]:


train_df.shape


# In[ ]:


dev_df.head()


# In[ ]:


train_df.columns = ['label'] + ['feature {}'.format(i) for i in range(train_df.shape[1]-1)]
test_df.columns = ['label'] + ['feature {}'.format(i) for i in range(train_df.shape[1]-1)]
dev_df.columns = ['label'] + ['feature {}'.format(i) for i in range(train_df.shape[1]-1)]


# In[ ]:


train_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/drug_all_lang_train.csv', index = False, encoding = 'utf-8')
test_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/drug_all_lang_test.csv', index = False, encoding = 'utf-8')
dev_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/drug_all_lang_dev.csv', index = False, encoding = 'utf-8')


# # Format Data
# 
# This pipeline is used to format data for crowd sourcing.

# In[35]:


nlp = spacy.load('en')
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)


# In[39]:


with io.open('/Users/dnissani/Desktop/Test_SageMaker/sampled_6months_en.txt', 'r', encoding = 'utf-8') as f:
    new_prod_data = []
    for line in f:
        new_prod_data.append(json.loads(line))


# In[40]:


print(new_prod_data[0].keys())


# In[41]:


keys = ['url', 'title', 'pageText', 'metaDescription', 'metaKeywords', 'metaSubject']


# In[42]:


new_prod_text = {}
new_prod_url = []
new_prod_text_no_key = []
for el in new_prod_data:
    helper = []
    for key in keys:
        helper.append(el[key])
    new_prod_text[el['url']] = helper

for el in new_prod_text.keys():
    new_prod_text_no_key.append(new_prod_text[el])
    new_prod_url.append(el)


# In[43]:


print(len(new_prod_text_no_key))


# In[44]:


print(len(new_prod_text))


# In[45]:


print(len(new_prod_url))


# In[46]:


new_prod_text_no_key[:10]


# In[47]:


prod_df = pd.DataFrame(new_prod_text_no_key, columns = keys)


# In[48]:


concat_prod = []
for el in new_prod_text_no_key:
    string = ''
    for text in el:
        string += text
    concat_prod.append(string)


# In[62]:


start_time = time.time()
language_score = []
for text in concat_prod:
    doc = nlp(text)
    language_score.append((doc._.language['language'], doc._.language['score']))
    print("--- %s seconds ---" % (time.time() - start_time))
print("final: --- %s seconds ---" % (time.time() - start_time))


# In[110]:


en_count = 0
not_count = 0
scores = []
indices = []
lang_not_eng = []
count = 0
for el in language_score:
    if el[0] == 'en':
        en_count += 1
    else:
        not_count +=1
        scores.append(el[1])
        indices.append(count)
        lang_not_eng.append(el[0])
    count += 1


# In[67]:


en_count


# In[112]:


scores[6]


# In[111]:


lang_not_eng[5]


# In[123]:


concat_prod[indices[16]]


# In[49]:


scoring_prod_df = pd.DataFrame(concat_prod, columns = ['Text'])


# In[ ]:


len(scoring_prod_df)


# In[ ]:


scoring_prod_df.values[:20]


# In[ ]:


scoring_prod_df.to_csv('/Users/dnissani/Desktop/Test_SageMaker/eng_prod_data.csv')


# In[ ]:


prod_df.columns = ['url', 'title', 'pagetext', 'metadescription', 'metakeywords', 'metasubject']


# In[ ]:


prod_df.head()


# # Label

# In[ ]:


prod_labels = pd.read_csv('/Users/dnissani/Desktop/Test_SageMaker/eng_drug_prod_labels.csv')


# In[ ]:


prod_labels.head()


# In[ ]:


prod_df = pd.concat([prod_df, prod_labels], axis = 1)


# In[ ]:


prod_df.head()


# In[ ]:


prod_df = prod_df.drop('Unnamed: 0', axis = 1)


# In[ ]:


prod_df = prod_df[['url','prediction', 'probabilities', 'title', 'metasubject', 'metakeywords', 'metadescription', 'pagetext']]


# In[ ]:


prod_df.head()


# In[ ]:


high_prod = prod_df[prod_df['prediction'] == 'high']


# In[ ]:


high_prod.head()


# In[ ]:


len(high_prod)


# In[ ]:


moderate_prod = prod_df[prod_df['prediction'] == 'moderate']


# In[ ]:


moderate_prod.head()


# In[ ]:


moderate_300 = moderate_prod.sample(300)


# In[ ]:


for_chris = pd.concat([high_prod, moderate_300], axis = 0)


# In[ ]:


for_chris


# In[ ]:


for_chris.to_csv('/Users/dnissani/Desktop/Test_SageMaker/EN_DRG_scored.csv', index = False, encoding = 'utf-8')


# In[ ]:


test = pd.read_csv('/Users/dnissani/Desktop/Test_SageMaker/EN_DRG_scored.csv')


# In[ ]:


len(test)


# In[ ]:


import csv

with open('/Users/dnissani/Desktop/Test_SageMaker/EN_DRG_trying.csv', 'w', encoding = 'utf-8') as f:
    writer = csv.writer(f, delimiter = ',',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(list(for_chris.columns))
    
    for line in for_chris.values:
        writer.writerow(line)


# In[ ]:




