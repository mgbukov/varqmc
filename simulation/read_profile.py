import pstats

p = pstats.Stats('profile_RK.txt')
p.sort_stats('tottime').print_stats(20)