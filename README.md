# ktLDA

This is an implementation of Latent Dirichlet Allocation for pedagogical purposes.

### Dependencies

- numpy
- tqdm

### Examples

```python
from ktlda import KtLDA
import pickle

with open('ourdata-cleaned.pickle', 'rb') as f:
    comp, rec = pickle.load(f)
X = comp + rec
Y = [0] * len(comp) + [1] * len(rec)

lda = KtLDA(n_components=2, alpha=0.5, beta=0.5, iterations=10, max_vocab=5000, random_state=663)
lda.fit(X)
print(lda.doc_topic_dist)
```
