def normalize_title(title):
    cleaned = title.strip().lower()
    return cleaned


result = normalize_title("  Colab 안내  ")
print(result)