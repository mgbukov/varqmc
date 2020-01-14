


template<typename I>
I rol(I x, uint rot, uint bits) {
// INLINE
// Rotate left a complete word.
// x: value to be rotated
// rot: rotate count, negative values will rotate right
// 1 cycle (should be if inlined)

  // A shift by >= bits is undefined by the C/C++ standard.
  // We take care for this case with the "& (bits - 1)" below.
  // For many CPUs this stunt is not neccessary.
  return (x << (rot & (bits - 1))) | (x >> ((-rot) & (bits - 1)));
  }


template<typename I>
I bit_permute_step(I x, I m, uint shift) {
// INLINE
// Can be replaced by bit_permute_step_simple,
// if for the relevant bits n the following holds:
// nr_1bits(bit_permute_step_simple(n,m,shift)) == nr_1bits(n)
// x86: >= 6/5 cycles
// ARM: >= 4/4 cycles

  // assert(((m << shift) & m) == 0);
  // assert(((m << shift) >> shift) == m);
  I t = ((x >> shift) ^ x) & m;
  x = x ^ t;  t = t << shift;  x = x ^ t;  // x = (x ^ t) ^ (t << shift);
  return x;
  }



template<typename I>
I bit_permute_step_simple(I x, I m, uint shift) {
// INLINE
// Simplified replacement of bit_permute_step
// Can always be replaced by bit_permute_step (not vice-versa).
// x86: >= 5/4 (5/3) cycles
// ARM: >= 3/2 cycles

  // assert(((m << shift) & m) == 0);
  // assert(((m << shift) >> shift) == m);
  // assert(((m << shift) | m) == all_bits);  // for permutations
  return ((x & m) << shift) | ((x >> shift) & m);
  }


template<typename I>
I general_reverse_bits(I x, uint k, const I a_bfly_mask[], uint ld_bits) {
// Swap all subwords of given levels.
// See Hacker's Delight, 7.1 "Generalized Bit Reversal"
// k: set of t_subword, i.e. one bit per subword size.

  uint i,j;
  I m;

  for (i = 0; i <= ld_bits-1; ++i) {  // UNROLL
    j = 1 << i;
    if ((k & j) != 0) {
      // x = bit_index_complement(x,j);
      m = a_bfly_mask[i];
      x = bit_permute_step_simple(x,m,j);
      }
    }
  return x;
  }


template<typename I>
I bswap(I x, const I a_bfly_mask[], uint ld_bits) {
// INLINE
// Exchange byte order.
// This can be expressed in assembler:
// bits = 8: n/a
// bits = 16: "xchg al,ah" or "rol ax,16"
// bits = 32: "bswap eax"
// bits = 64: "bswap rax"
// bits = 128: "xchg rax,rdx; bswap rax; bswap rdx"

  return general_reverse_bits(x, ~7, a_bfly_mask, ld_bits);
  }



