def calc_percentage(li, value):
    """calculate the percentage of elements less or equal than value in li.
    """
    numer = sum([elem <= value for elem in li])
    demo = len(li)
    return numer/deno

def binary_search_inc(fn, target, init = (0, 1000)):
    """search for the smallest val such that fn(val) is greater or equal to
    target in an increasing sequence.
    """
    lower, upper = init
    fn_lower, fn_upper = map(fn, init)
    mid = (lower + upper) // 2
    fn_mid = fn(mid)
    if lower == mid:
        if fn_mid >= fn_lower:
            return lower
        else:
            return upper

    if fn_mid >= target:
        return binary_search_inc(fn, target, init = (lower, mid))
    else:
        return binary_search_inc(fn, target, init = (mid, upper))



def find_split(file_name, trunc = 160, percentage = 0.99, split_token = '__eot__'):
    """search for minimum params that covers `percentage` of truncated training data.

    Args:
        file_name: The path to the training CSV file.
        trunc: (optional) The length of content passed to the model.
        percentage: (optional) The percentage to cover.
        split_token: (optional) The token that separates the vector into sentences.

    Returns:
        (num_sentences, num_words)
    """
    with open(train_csv, newline='') as csv_file:
        csv_reader=csv.reader(csv_file)
        num_sentence_counts=[]
        num_word_counts=[]
        next(csv_reader)
        for row in csv_reader:
            splitted=row[0][:trunc].split(split_token)
            num_word_counts+=[max([sen.count(' ') for sen in splitted])]
            num_sentence_counts+=[len(splitted)]

        num_sentences = binary_search_inc(lambda x: calc_percentage(num_sentence_counts, x), percentage)
        num_words = binary_search_inc(lambda x: calc_percentage(num_word_counts, x), percentage)
        return (num_sentences, num_words)
