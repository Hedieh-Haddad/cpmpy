def geometric_sequence(i):
    return 1.2 * i

def luby_sequence(i, current_timeout, timeout_list):
    timeout_list.append(current_timeout)
    sequence = [timeout_list[0]]
    while len(sequence) <= i:
        sequence += sequence + [2 * sequence[-1]]
    return sequence[i]




