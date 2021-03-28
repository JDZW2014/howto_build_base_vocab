"""
date：          2021-03-18
Description : mining freq phrase
auther : wcy
"""
# import modules
import argparse

# arg parse
parser = argparse.ArgumentParser()
parser.add_argument('--raw_input', type=str, default='/keyword_extraction/seg_phrase/data/DBLP.5K.txt')
parser.add_argument('--threshold', type=int, default=1000)
parser.add_argument('--pattern_output', type=str, default="patterns.csv")
args = parser.parse_args()

# define config
ENDINGS = ".!?,;:\"[]"

__all__ = ["frequent_pattern_mining"]


# define function
def frequent_pattern_mining(tokens, pattern_output_filename, threshold):
    result = {}

    tokens_number = len(tokens)
    for i in range(tokens_number):
        token = tokens[i]
        if token == '$':
            continue
        if token in result:
            result[token].append(i)
        else:
            result[token] = [i]
    print("# of distinct tokens = ", len(result))

    pattern_output = open(pattern_output_filename, 'w')

    frequent_patterns = []
    pattern_length = 1
    while len(result) > 0:
        if pattern_length > 6:
            break
        # print "working on length = ", patternLength
        pattern_length += 1
        new_dict = {}
        for pattern, positions in result.items():
            occurrence = len(positions)
            if occurrence >= threshold:
                frequent_patterns.append(pattern)

                pattern_output.write(pattern + "," + str(occurrence) + "\n")
                for i in positions:
                    if i + 1 < tokens_number:
                        if tokens[i + 1] == '$':
                            continue
                        new_pattern = pattern + " " + tokens[i + 1]
                        if new_pattern in new_dict:
                            new_dict[new_pattern].append(i + 1)
                        else:
                            new_dict[new_pattern] = [i + 1]
        result.clear()
        result = new_dict
    pattern_output.close()
    return frequent_patterns


# define main
def main(threshold, raw_text_input, pattern_output_filename):

    raw = open(raw_text_input, 'r')
    tokens = []
    for line in raw:
        inside = 0
        chars = []
        for ch in line:
            if ch == '(':
                inside += 1
            elif ch == ')':
                inside -= 1
            elif inside == 0:
                if ch.isalpha():
                    chars.append(ch.lower())
                elif ch == '\'':
                    chars.append(ch)
                else:
                    if len(chars) > 0:
                        tokens.append(''.join(chars))
                    chars = []
            if ch in ENDINGS:
                tokens.append('$')
        if len(chars) > 0:
            tokens.append(''.join(chars))
            chars = []
        
    print("# tokens = ", len(tokens))

    frequent_patterns = frequent_pattern_mining(tokens, pattern_output_filename, threshold)

    print("# of frequent pattern = ", len(frequent_patterns))


if __name__ == "__main__":
    main(threshold=args.threshold, raw_text_input=args.raw_input, pattern_output_filename=args.pattern_output)
