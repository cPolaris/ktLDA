import numpy as np
import cython
cimport cython
import time


@cython.boundscheck(False)
@cython.wraparound(False)
def inf(int K, double alpha, double beta, long[:,:] docs, double vocab_len, int iteration, long[:,:] topic_doc_word, double[:,:] cnt_doc_topic, double[:,:] cnt_topic_word, double[:] cnt_topic):
    """
    Cython Optimized inference function.
    Pseudocode Available in the report.
    """
    for it in range(iteration):

        start = time.time()
        for d_i, doc in enumerate(docs):
            for i, wo in enumerate(doc):
                if wo < 0:
                    continue
                word = int(wo)

                #Take one word out
                topic = topic_doc_word[d_i][i]

                # Decrease the counters
                cnt_doc_topic[d_i][topic] -= 1
                cnt_topic_word[topic, word] -= 1
                cnt_topic[topic] -= 1

                #Calculate the probabilty for the word assiging to topic 1...K
                p_z = []
                for k in range(K):
                    p_zk = (cnt_doc_topic[d_i][k] + alpha) * ((cnt_topic_word[k, word] + beta) / (
                            cnt_topic[k] + vocab_len * beta))
                    p_z.append(p_zk)

                # Normalize P(z)
                sum_p_z = sum(p_z)
                for idx, p_z_i in enumerate(p_z):
                    p_z[idx] = p_z[idx] / sum_p_z

                # Generate new z from multinomial distribution.
                dist = np.random.multinomial(1,p_z)
                new_z = dist.argmax()

                #Update the counter accordingly.
                topic_doc_word[d_i][i] = new_z
                cnt_doc_topic[d_i][new_z] += 1
                cnt_topic_word[new_z, word] += 1
                cnt_topic[new_z] += 1

        end = time.time()
        print("Iteration: ", it," Time: ", end - start)
