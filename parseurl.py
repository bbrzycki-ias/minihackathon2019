import html
import urllib
import string
import re


def parseurl(url):
    # If url is not http, add it in true hacky fashion :)
    if url[:4] != 'http':
        url = 'http://' + url
    
    # Get url components
    parsed = urllib.parse.urlsplit(url)

    # Get domain
    netloc = parsed.netloc
    netloc = netloc.split('.')

    # Remove www if it exists
    if netloc[0] == 'www':
        del netloc[0]

    # Remove TLD
    netloc = ' '.join(netloc[:-1])

    # Get remaining things -- for now exclude query and fragment sections
    suffix = ' '.join([parsed.path, ])#parsed.query, parsed.fragment])

    # Both url unescape chars with & and unquote chars with %
    suffix = urllib.parse.unquote_plus(html.unescape(suffix)).split('.')[0]

    # Remove punctuation in suffix
    translator = str.maketrans('', '', string.punctuation)
    suffix = ' '.join(re.compile(r'[\:/?=\-_&]+', re.UNICODE).split(suffix)).translate(translator).strip()

    # Get full text
    words = netloc + ' ' + suffix

    # Remove all digits and single characters
    final = ' '.join([x for x in ''.join([i for i in words if not i.isdigit()]).split() if len(x) > 1])

    return final