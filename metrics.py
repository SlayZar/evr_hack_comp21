def metric1(answers, user_csv):
    delta_c = np.abs(answers - user_csv)
    hit_rate_c = np.int64(delta_c < 20)
    N = np.size(answers)

    return np.sum(hit_rate_c) / N
    
def metric2(answers, user_csv):
    delta_t = np.abs(np.array(answers) - np.array(user_csv))
    hit_rate_t = np.int64(delta_t < 0.02)
    N = np.size(answers)

    return np.sum(hit_rate_t) / N