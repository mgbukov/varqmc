import pstats

p = pstats.Stats('profile_sgd.txt')
p.sort_stats('tottime').print_stats(20)