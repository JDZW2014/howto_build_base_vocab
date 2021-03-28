# import modules
from random import shuffle
from math import sqrt
from sklearn import cluster

__all__ = ["auto_label_generation"]


# define function
def normalize_matrix(matrix, dimension):
    for i in range(dimension):
        sum = 0
        sum2 = 0
        for j in range(len(matrix)):
            sum += matrix[j][i]
            sum2 += matrix[j][i] * matrix[j][i]
        avg = sum / len(matrix)
        avg2 = sum2 / len(matrix)
        variance = avg2 - avg * avg
        stderror = sqrt(variance)
        for j in range(len(matrix)):
            matrix[j][i] = (matrix[j][i] - avg)
            if stderror > 1e-8:
                matrix[j][i] /= stderror
    return matrix


def normalize(word):
    word = word.lower()
    result = []
    for i in range(len(word)):
        if word[i].isalpha() or word[i] == '\'':
            result.append(word[i])
        else:
            result.append(' ')
    word = ''.join(result)
    return ' '.join(word.split())


def auto_label_generation(knowledge_base, knowledge_base_large, feature_table, patterns, generated_label):

    ground_truth = {}
    for line in open(knowledge_base, 'r'):
        word = line.strip()
        # word = normalize(word)
        ground_truth[word] = True

    kb_phrases_all = set()
    for line in open(knowledge_base_large, 'r'):
        word = line.strip()
        word = normalize(word)
        kb_phrases_all.add(word)

    patterns_support = list()
    for line in open(patterns, 'r'):
        tokens = line.split(',')
        patterns_support.append((tokens[0].strip(), int(tokens[1])))

    sorted_patterns = sorted(patterns_support, key=lambda tup: -tup[1])
    patterns_candidates = set([tup[0] for tup in sorted_patterns[:len(sorted_patterns) / 2]])

    # loading
    dimension = 0
    attributes = []
    forbid = ['outsideSentence', 'log_occur_feature', 'constant', 'frequency']
    matrix_wiki = []
    phrase_wiki = []
    matrix_other = []
    phrase_other = []

    for line in open(feature_table, 'r'):
        tokens = line.split(',')
        if tokens[0] == 'pattern':
            attributes = tokens
            # print attributes
            continue
        coordinates = []
        for i in range(1, len(tokens)):
            if attributes[i] in forbid:
                continue
            coordinates.append(float(tokens[i]))
        dimension = len(coordinates)
        if tokens[0] in ground_truth and tokens[0] in patterns_candidates:
            matrix_wiki.append(coordinates)
            phrase_wiki.append(tokens[0])
        else:
            matrix_other.append(coordinates)
            phrase_other.append(tokens[0])

    # normalization
    matrix_wiki = normalize_matrix(matrix_wiki, dimension=dimension)
    matrix_other = normalize_matrix(matrix_other, dimension=dimension)

    # k-means
    kmeans = cluster.MiniBatchKMeans(n_clusters=min(200, len(matrix_wiki)), max_iter=300, batch_size=5000)
    kmeans.fit(matrix_wiki)
    labels_wiki = kmeans.labels_
    bins = []
    for i in range(1000):
        bins.append([])

    for i in range(len(labels_wiki)):
        bins[labels_wiki[i]].append(phrase_wiki[i])

    labels = []
    for bin in bins:
        shuffle(bin)
        if len(bin) > 0:
            labels.append(bin[0] + '\t1\n')
    npos = len(labels)
    # k-means
    kmeans = cluster.MiniBatchKMeans(n_clusters=min(npos * 2, len(matrix_other)), max_iter=300, batch_size=5000)
    kmeans.fit(matrix_other)
    labels_other = kmeans.labels_
    bins = []
    for i in range(min(npos * 2, len(matrix_other))):
        bins.append([])

    for i in range(len(labels_other)):
        bins[labels_other[i]].append(phrase_other[i])

    for bin in bins:
        shuffle(bin)
        if len(bin) > 0:
            for i in range(len(bin)):
                if bin[i] not in kb_phrases_all:
                    labels.append(bin[i] + '\t0\n')
                    break

    out = open(generated_label, 'w')
    out.write(''.join(labels))
    out.close()

    print(len(labels), 'generated,', npos, 'positive')
