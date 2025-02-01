from collections import defaultdict

vocab = {'example': 0, 'word': 1, 'another': 2}  # etc.
inverse_vocab = {index: word for word, index in vocab.items()}
def get_word_embedding(word, model, vocab):
    index = vocab.get(word)
    if index is not None:
        return model.W1[index]
    else:
        return None  # or a zero vector

def compute_document_embedding(text, model, vocab):
    """Compute the average embedding for a document."""
    tokens = word_tokenize(text.lower())
    embeddings = []
    for token in tokens:
        emb = get_word_embedding(token, model, vocab)
        if emb is not None:
            embeddings.append(emb)
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.embedding_dim)

# Precompute embeddings for all articles
document_embeddings = []
for text in df['text']:
    emb = compute_document_embedding(text, model, vocab)
    document_embeddings.append(emb)

document_embeddings = np.array(document_embeddings)  # Shape: (num_articles, embedding_dim)
