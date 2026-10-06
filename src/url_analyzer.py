import re
import math
import ipaddress
from urllib.parse import urlparse
import numpy as np


def calculate_url_features(url):

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc
    if ":" in domain:
        domain = domain.split(":")[0]

    path = parsed.path
    query = parsed.query

    full_url = url

    # Basic lengths
    url_length = len(full_url)
    domain_length = len(domain)

    # IP address detection
    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except:
        is_domain_ip = 0

    # TLD
    parts = domain.split(".")
    tld = parts[-1] if len(parts) > 1 else ""
    tld_length = len(tld)

    # Subdomains
    no_of_subdomain = max(len(parts) - 2, 0)

    # Obfuscation
    obfuscated_chars = len(re.findall(r"%[0-9a-fA-F]{2}", full_url))

    has_obfuscation = 1 if obfuscated_chars > 0 else 0

    obfuscation_ratio = (
        obfuscated_chars / url_length if url_length > 0 else 0
    )

    # Letters
    letters = len(re.findall(r"[A-Za-z]", full_url))

    letter_ratio = (
        letters / url_length if url_length > 0 else 0
    )

    # Digits
    digits = len(re.findall(r"[0-9]", full_url))

    digit_ratio = (
        digits / url_length if url_length > 0 else 0
    )

    # Special characters
    equals_count = full_url.count("=")
    qmark_count = full_url.count("?")
    ampersand_count = full_url.count("&")

    special_chars = len(
        re.findall(r"[^A-Za-z0-9]", full_url)
    )

    other_special_chars = len(
        re.findall(r"[^A-Za-z0-9:/?.=&_-]", full_url)
    )

    special_ratio = (
        special_chars / url_length if url_length > 0 else 0
    )

    # Character continuation rate
    runs = re.findall(r"[A-Za-z0-9]+", full_url)

    if runs:
        continuation = max(len(x) for x in runs)
        char_continuation_rate = continuation / url_length
    else:
        char_continuation_rate = 0

    # URL similarity heuristic
    suspicious_chars = sum(
        full_url.count(c)
        for c in ["@", "-", "_", "%", "=", "?"]
    )

    url_similarity_index = max(
        0,
        min(
            100,
            100 - (suspicious_chars / max(url_length, 1)) * 100
        )
    )

    # URL character probability heuristic
    if url_length > 0:
        common_chars = sum(
            1 for c in full_url
            if c.isalnum() or c in "/:.-_?=&"
        )

        url_char_prob = common_chars / url_length
    else:
        url_char_prob = 0

    features = {
        "URLLength": url_length,
        "DomainLength": domain_length,
        "IsDomainIP": is_domain_ip,
        "URLSimilarityIndex": url_similarity_index,
        "CharContinuationRate": char_continuation_rate,
        "URLCharProb": url_char_prob,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obfuscation,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": equals_count,
        "NoOfQMarkInURL": qmark_count,
        "NoOfAmpersandInURL": ampersand_count,
        "NoOfOtherSpecialCharsInURL": other_special_chars,
        "SpacialCharRatioInURL": special_ratio
    }

    return features