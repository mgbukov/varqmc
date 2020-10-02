#include "numpy/ndarraytypes.h"
//#include "./DNN.h"
//#include "models/RBM_real_symm.h"
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;
#include <unordered_map>
//#include <mpi.h>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */





template<typename I>
int op(I &r,double &m,const int n_op,const char opstr[],const int indx[], const int N){
	const I s = r;
	const I one = 1;
	for(int j=n_op-1;j>-1;j--){

		const int ind = N-indx[j]-1;
		const I b = (one << ind);
		const bool a = (bool)((r >> ind)&one);
		const char op = opstr[j];
		switch(op){
			case '+':
				m *= (a?0:1);
				r ^= b;
				break;
			case '-':
				m *= (a?1:0);
				r ^= b;
				break;
			case 'z':
				m *= (a?0.5:-0.5);
				break;
			/*
			case 'x':
				r ^= b;
				m *= 0.5;
				break;
			case 'y':
				m *= 0.5; //(a?std::complex<double>(0,0.5):std::complex<double>(0,-0.5));
				r ^= b;
				break;
			case 'I':
				break;
			*/
			
			default:
				return -1;
		}

		if(std::abs(m)==0){
			r = s;
			break;
		}

	}

	return 0;
}


template<class I>
inline int count_bits (I s) 
{
    int count=0;
    while (s!=0)
    {
        s = s & (s-1);
        count++;
    }
    return count;
}


template<class T>
inline T swap_bits(const T b,int i,int j)
{
	T one=1;
	T x = ( (b>>i)^(b>>j) ) & one;
	return b ^ ( (x<<i) | (x<<j) );
}


inline int choose_n_k(int n, int k) {
    double res = 1;
    for (int i = 1; i <= k; ++i)
        res = res * (n - k + i) / i;
    return (int)(res + 0.01);
}


template<class T>
inline T magnetized_int(int m, int N, T ordinal)
{
	// https://cs.stackexchange.com/questions/67664/prng-for-generating-numbers-with-n-set-bits-exactly

	T one = 1;
	
	//cout << ordinal << endl;

    T s0 = 0; // output integer
    for (int bit = N; m > 0; --bit)
    {
        T nCk = choose_n_k(bit, m);
        if (ordinal >= nCk)
        {
            ordinal -= nCk;
            s0 |= (one << bit);
            --m;
        }
    }
    return s0;
}




template<class T, class J>
inline void _int_to_spinstate(const int N,T state,J *out)
{	
	T one=1;
	for(int i=0;i<N;i++){
		
		//cout << i << " , " << N-i-1 << " , " << one<<(N-i-1) << endl;
		
		out[i] = (state & (one<<(N-i-1) )) ? 1 : -1; // [-1,+1] state representation

	}
}


template<class I, class J>
I rep_int_to_spinstate(const int N,I s,J out[])
{	
	
	I t = s;
	I r = s;
	I p = s;

	int counter=0;

	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				for(int m=0;m<_L;m++){
					for(int n=0;n<_L;n++){

						p = t;
						for(int l=0;l<2;l++){
							if(p > r) r = p;

							p = inv_spin(p);
						}

						//if(t > r) r = t;	

						
						_int_to_spinstate(N,t,&out[counter*N]);
						t = shift_x(t);
						counter++;
													
					}
					t = shift_y(t);
				}
				t = flip_x(t);
			}
			t = flip_y(t);
		}
		t = flip_d(t);
	}
		

	return r;	
	
		
}



template<class I, class J>
void int_to_spinstate(const int N,I s,J out[])
{	
	
	I t = s;

	npy_uint16 counter=0;

	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				for(int m=0;m<_L;m++){
					for(int n=0;n<_L;n++){

						_int_to_spinstate(N,t,&out[counter*N]);
						t = shift_x(t);
						counter++;
													
					}
					t = shift_y(t);
				}
				t = flip_x(t);
			}
			t = flip_y(t);
		}
		t = flip_d(t);
	}
		
}



template<class I>
npy_uint32 cyclicity(const int N,I t)
{	
	
	//t=ref_state(t);

	int counter=0;
	bool encountered=0;
	
	std::vector<I> Ts(cyclicity_factor,0);
	Ts[counter]=t;
	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){

				for(int l=0;l<2;l++){

					for(int m=0;m<_L;m++){

						for(int n=0;n<_L;n++){

							
						
							t = shift_x(t);

							// check if state has been encountered
							for(int _k=0;_k<counter+1;_k++){
								if(t==Ts[_k]){
									encountered=1;
									break;
								}
							}

							// break loop if state has been encountered
							if(encountered){
								encountered=0;
								break;
							}
							else{
								Ts[counter]=t;
								counter++;
								
							}


								
						}
						t = shift_y(t);
					}

					t = inv_spin(t);
				}

				t = flip_x(t);
			}
			t = flip_y(t);
		}
		t = flip_d(t);
	}	

	return counter;	
	
		
}




template<class I, class J>
I rep_int_to_spinstate_conv(const int N,I s,J out[])
{	
	
	I t = s;
	I r = s;
	I p = s;

	int counter=0;

	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				
				p = t;
				for(int l=0;l<2;l++){
					for(int m=0;m<_L;m++){
						for(int n=0;n<_L;n++){
							if(p > r) r = p;

							p = shift_x(p);
						}
						p = shift_y(p);
					}
					p = inv_spin(p);
				}
				_int_to_spinstate(N,t,&out[counter*N]);
				t = flip_x(t);
				counter++;

			}
			t = flip_y(t);
		}
		t = flip_d(t);
	}
		

	return r;	
	
		
}



template<class I, class J>
void int_to_spinstate_conv(const int N,I s,J out)
{	
	
	I t = s;

	int counter=0;

	
	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				
				//cout << counter << " , " << N << " , " << counter*N << endl;
				_int_to_spinstate(N,t,&out[counter*N]);
				t = flip_x(t);
				counter++;												
					
			}
			t = flip_y(t);
		}
		t = flip_d(t);
	}

	
}



template<class T, class J>
inline void _int_to_spinstate2(const int N,T state,J *out, const int m, const int n)
{	
	T one=1;
	for(int i=0;i<N;i++){
		//cout << m << " , " << n << " , " << m+n+i << endl;		
		out[m+n+i] = (state & (one<<(N-i-1) )) ? 1 : -1; // [-1,+1] state representation

	}
}


template<class I, class J>
void int_to_spinstate_conv2(const int N,I s,J *out,const int m)
{	
	
	I t = s;
	J one=1;

	cout << sizeof(int) << endl;

	int counter=0;


	for(int i=0;i<2;i++){
		for(int j=0;j<2;j++){
			for(int k=0;k<2;k++){
				
				for(int ii=0;ii<N;ii++){

					//cout << m+counter*N+ii << endl;

					out[m+counter*N+ii] = one; 

				}
				counter++;														
			}
		}
	}

	
	// for(int i=0;i<2;i++){
	// 	for(int j=0;j<2;j++){
	// 		for(int k=0;k<2;k++){
				
	// 			//cout << counter << " , " << N << " , " << counter*N << endl;
	// 			_int_to_spinstate2(N,t,out,m,counter*N);
	// 			t = flip_x(t);
	// 			counter++;												
					
	// 		}
	// 		t = flip_y(t);
	// 	}
	// 	t = flip_d(t);
	// }

	
}


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////


template<class I>
int update_offdiag_exact(const int n_op,
				  const char opstr[],
				  const int indx[],
				  const double A,
				  const int Ns,
				  const	I ket[], // col
				  		I bra_rep[],
				  		npy_int32 ket_index[],
				  		double M[]
				  )
{	
	int N=_L*_L;
	
	#pragma omp parallel
	{		

		int a=Ns/(100*omp_get_num_threads());
		int b=1;

		//cout << omp_get_num_threads() << " , " << omp_get_max_threads() << endl;

		const npy_intp chunk = (a < b) ? b : a; // std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);
		#pragma omp for schedule(dynamic,chunk) 
		for(int j=0;j<Ns;j++){		

			double m = A;
			I r = ket[j];

			op(r,m,n_op,opstr,indx,N);
			//bool pcon_bool = (count_bits(r)==N/2);

			//cout << pcon_bool << " , " << m << endl;

			if(m!=0.0){ // r state in same particle-number sector
	
				bra_rep[j] = ref_state(r);
				M[j] = m;
				ket_index[j]=j;

			}
		}
	}


	int l=0;
	for(int j=0;j<Ns;j++){

		if(ket_index[j]>=0){

			ket_index[l]=j;

			l++;
		}
	}



	return l;
}



template<class I>
int update_offdiag(const int n_op,
				  const char opstr[],
				  const int indx[],
				  const double A,
				  const int Ns,
				  const int N_symm,
				  const	I ket[], // col
				  		I bra_rep[],
				  		npy_int8 spin_bra[], // row
				  		npy_int32 ket_index[],
				  		double M[],
				  I rep_int_to_spinconfig(const int,I,npy_int8*)
				  )
{	
	int N=_L*_L;
	
	#pragma omp parallel
	{		

		int a=Ns/(100*omp_get_num_threads());
		int b=1;

		//cout << omp_get_num_threads() << " , " << omp_get_max_threads() << endl;

		const npy_intp chunk = (a < b) ? b : a; // std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);
		#pragma omp for schedule(dynamic,chunk) 
		for(int j=0;j<Ns;j++){
			
			
			// const I one = 1; 
			// I r = ket[j];

			// // operator site indices
			// const int ind_0 = N-indx[0]-1;
			// const int ind_1 = N-indx[1]-1;
			
			// const bool a_0 = (bool)((r >> ind_0)&one);
			// const bool a_1 = (bool)((r >> ind_1)&one);

			// const bool bool_A = (opstr[0]=='+' && opstr[1]=='-' && (!a_0) &&   a_1 );
			// const bool bool_B = (opstr[0]=='-' && opstr[1]=='+' &&   a_0  && (!a_1));

			// if(bool_A || bool_B){

			// 	r ^= (one << ind_0)^(one << ind_1);

			// 	M[j] = A;
			// 	ket_index[j]=j;

			// 	bra_rep[j] = rep_int_to_spinstate(N,r,&spin_bra[N*N_symm*j]);

			// }
			

			double m = A;
			I r = ket[j];

			op(r,m,n_op,opstr,indx,N);
			//bool pcon_bool = (count_bits(r)==N/2);

			//cout << pcon_bool << " , " << m << endl;

			if(m!=0.0){ // r state in same particle-number sector
	
				bra_rep[j] = rep_int_to_spinconfig(N,r,&spin_bra[N*N_symm*j]);
				M[j] = m;
				ket_index[j]=j;

			}
		}
	}


	int l=0;
	for(int j=0;j<Ns;j++){

		if(ket_index[j]>=0){

			ket_index[l]=j;

			l++;
		}
	}



	return l;
}





template<class I>
void update_diag(const int n_op,
				const char opstr[],
				const int indx[],
				const double A,
				const int Ns,
				const I ket[], // col
				  	  double M[]
				  )
{	
	#pragma omp parallel
	{		
		int a=Ns/(100*omp_get_num_threads());
		int b=1;

		//cout << omp_get_num_threads() << " , " << omp_get_max_threads() << endl;

		const npy_intp chunk = (a < b) ? b : a;
		#pragma omp for schedule(dynamic,chunk) 
		for(int j=0;j<Ns;j++){
			
			double m = A;
			I r = ket[j];
			
			op(r,m,n_op,opstr,indx,_L*_L);
			M[j] += m;
		}
	}

}


void offdiag_sum(int Ns,
				int n_per_term[], 
				double Eloc_cos[],
				double Eloc_sin[],
				npy_uint32 ket_ind[],
				double MEs[],
				const double log_psi_bras[],
				const double phase_psi_bras[],
				const double log_psi_kets[]
	)
{
	int n_cum=0;
	double aux=0.0;

	for(int l=0;l<Ns;l++){

		int n=n_per_term[l];

		for(int i=0;i<n;i++){

			int j=n_cum+i;
			
			//cout << l << " , "<< i << " , "<< n << " , "<<j<< " , " << Eloc_cos[ket_ind[j]] << " , " << ket_ind[j] << " , " << psi_bras[j] << " , " << std::cos(phase_psi_bras[j]) << endl;
			
			aux=MEs[j] * std::exp(log_psi_bras[j]-log_psi_kets[ket_ind[j]]);

			Eloc_cos[ket_ind[j]] += aux * std::cos(phase_psi_bras[j]);
			Eloc_sin[ket_ind[j]] += aux * std::sin(phase_psi_bras[j]);
		}

		n_cum+=n;
	}
}




void offdiag_sum(int Ns,
				int n_per_term[], 
				double Eloc_cos[],
				double Eloc_sin[],
				npy_uint32 ket_ind[],
				double MEs[],
				const double log_psi_bras[],
				const double phase_psi_bras[],
				const double log_psi_kets[],
				const double dlog_psi_kets[]
	)
{
	int n_cum=0;
	double aux=0.0;

	for(int l=0;l<Ns;l++){

		int n=n_per_term[l];

		for(int i=0;i<n;i++){

			int j=n_cum+i;
			
			//cout << l << " , "<< i << " , "<< n << " , "<<j<< " , " << Eloc_cos[ket_ind[j]] << " , " << ket_ind[j] << " , " << psi_bras[j] << " , " << std::cos(phase_psi_bras[j]) << endl;
			
			aux=MEs[j] * std::exp(log_psi_bras[j]-log_psi_kets[ket_ind[j]]) * dlog_psi_kets[j];

			Eloc_cos[ket_ind[j]] += aux * std::cos(phase_psi_bras[j]);
			Eloc_sin[ket_ind[j]] += aux * std::sin(phase_psi_bras[j]);
		}

		n_cum+=n;
	}
}


