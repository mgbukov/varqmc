import pstats

p = pstats.Stats('profile_CNN_sgd.txt')
p.sort_stats('tottime').print_stats(5)

p = pstats.Stats('profile_DNN_sgd.txt')
p.sort_stats('tottime').print_stats(5)