#include "numpy/ndarraytypes.h"
//#include "./DNN.h"
#include "models/RBM_real_symm.h"
#include <random>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;
#include <unordered_map>
//#include <mpi.h>
#include <omp.h>





template<typename I>
int op(I &r,double &m,const int n_op,const char opstr[],const int indx[], const int N){
	//const I s = r;
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
		/*
		if(std::abs(m)==0){
			r = s;
			break;
		}
		*/
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


template<class T>
inline void _int_to_spinstate(const int N,T state,npy_int8 out[])
{	npy_int8 n;
	T one=1;
	for(int i=0;i<N;i++){
		n=(state / (one<<(N-i-1)) ) % 2;
		//out[i] = (n + 2) % 2; // [0,1] state representation
		out[i] = 2 * ( (n + 2) % 2 ) - 1; // [-1,+1] state representation
	}
}


template<class I>
I rep_int_to_spinstate(const int N,I s,npy_int8 out[])
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



template<class I>
void int_to_spinstate(const int N,I s,npy_int8 out[])
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



///////////////////////////////////////////////////////////






template <class I>
class Monte_Carlo{

    private:
        std::random_device rd;
        std::mt19937 gen;

	public:

		Monte_Carlo() {
            std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        };

		~Monte_Carlo(){};
  
  		unsigned int seed;
		void set_seed(unsigned int u) {
            gen.seed(u);
            seed=u; 
        };

        /*
        int world_size=0;
        int world_rank=0;

        void mpi_init() {
		    // Initialize the MPI environment
		    MPI_Init(NULL, NULL);

		    // Get the number of processes
		    //int world_size;
		    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

		    // Get the rank of the process
		    //int world_rank;
		    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

		    // Get the name of the processor
		    char processor_name[MPI_MAX_PROCESSOR_NAME];
		    int name_len;
		    MPI_Get_processor_name(processor_name, &name_len);

		    // Print off a hello world message
		    printf("initialized processor %s, rank %d/%d\n", processor_name, world_rank, world_size);

		}

		void mpi_close() {
			MPI_Finalize();
		}

		template<class T>
		void mpi_allgather(T* send_data,int send_count,T* recv_data,int recv_count	
	    )
	    {	int size_T = sizeof(T);
	    	MPI_Allgather(send_data, size_T*send_count, MPI_CHAR, recv_data, size_T*recv_count, MPI_CHAR, MPI_COMM_WORLD);
	    }
		*/
		

        // Produce a uniform random sample from the open interval (0, 1).
        double uniform() {
            std::uniform_real_distribution<double> unif(0,1);
            return unif(gen);
        };
		



        std::unordered_map<I, double> phase_dict;
        std::unordered_map<I, double> mod_dict; 

        int build_ED_dicts(int sign, int L, double J2){

        	ifstream infile;
        	I state;
        	int sign_1, sign_2;
        	double norm;
        	double log_psi;
        	


        	if(J2 > 0.0001){
        		if(L==4){
	        		infile.open("../ED_data/data-GS_J1-J2_Lx=4_Ly=4_J1=1.0000_J2=0.5000.txt");
	        	}
	        	else{
	        		infile.open("../ED_data/data-GS_J1-J2_Lx=6_Ly=6_J1=1.0000_J2=0.5000.txt");
	        	}
        	}
        	else{
        		if(L==4){
	        		infile.open("../ED_data/data-GS_J1-J2_Lx=4_Ly=4_J1=1.0000_J2=0.0000.txt");
	        	}
	        	else{
	        		infile.open("../ED_data/data-GS_J1-J2_Lx=6_Ly=6_J1=1.0000_J2=0.0000.txt");
	        	}
        	}
        	
		   

		    if(infile.fail()) // checks to see if file opended 
		    { 
		    	cout << "error loading file" << endl; 
		    	return 1; // no point continuing if the file didn't open...
		    }

		    while(!infile.eof()) // reads file to end of *file*, not line
			{ 
				infile >> state; // read first column number
				infile >> log_psi; // read second column number
				infile >> norm;
				infile >> sign_1; // J1 flipped
				infile >> sign_2; // J1 not flipped

				//cout << state << " , " << norm << " , " << check_state(state) << endl;

				if(sign<0){
					if(sign_1 > 0){
						phase_dict[state]=0.0;	
					}
					else{
						phase_dict[state]=M_PI;
					}
				}
				else{
					if(sign_2 > 0){
						phase_dict[state]=0.0;	
					}
					else{
						phase_dict[state]=M_PI;
					}
				}

				
				
				mod_dict[state]=std::exp(log_psi)*std::sqrt(norm/cyclicity_factor);

			} 
			infile.close();

			cout << "finished creating ED data dicts.\n" << endl;

			return 0;
        }


        void evaluate_mod_dict(I keys[], double values[], int Ns){
        	for(int j=0; j<Ns; ++j){
        		values[j]=mod_dict[keys[j]];
        	}
        }

        void evaluate_phase_dict(I keys[], double values[], int Ns){
        	for(int j=0; j<Ns; ++j){
        		values[j]=phase_dict[keys[j]];
        	}
        }



		int sample_DNN(I s,
						int N_MC_points,
						int thermalization_time,
						int auto_correlation_time,
						//
						npy_int8 spin_states[],
						I ket_states[],
						double mod_kets[],
						double phase_kets[],
						//
						const double W_fc_real[],
						const double W_fc_imag[],
						const int N_fc

			)
		{			
			int N_sites=_L*_L;

			I t;
			double mod_psi_s, mod_psi_t, phase_psi_s;

			npy_uint16 _i,_j;
			std::uniform_int_distribution<> random_site(0,N_sites-1);

			std::vector<npy_int8> spinstate_s(N_sites*N_symms), spinstate_t(N_sites*N_symms);
		 

			int_to_spinstate(N_sites,s,&spinstate_s[0]);
			mod_psi_s=evaluate_mod(&spinstate_s[0],W_fc_real,W_fc_imag,N_sites,N_fc);
				  		    

		    int j=0;
		    int k=0;
		    int accepted=0;

		  
		    while(k < N_MC_points){

		    	t=s;
		    	//spinstate_t=spinstate_s;

		    	// propose a new state until a nontrivial configuration is drawn
		    	while(t==s){
		    		_i = random_site(gen);
		    		_j = random_site(gen);
		    		t = swap_bits(s,_i,_j);
		    	};

		    	//cout << _i << " , " << _j << " , " << s << " , " << t << endl;

		    	
		    	// swap spin configuration
		    	// for(int _k=0;_k<N_symms;_k++){
		    	// 	spinstate_t[N_sites-1-_i + _k*N_sites] = spinstate_s[N_sites-1-_j + _k*N_sites];
		    	// 	spinstate_t[N_sites-1-_j + _k*N_sites] = spinstate_s[N_sites-1-_i + _k*N_sites];
		    	// }

		    	int_to_spinstate(N_sites,t,&spinstate_t[0]);
		    	
		    	   	
		    	// compute amplitude
		    	mod_psi_t=evaluate_mod(&spinstate_t[0],W_fc_real,W_fc_imag,N_sites,N_fc);

		    	//cout << s << " , " << t << " , " << mod_psi_s << " , " << mod_psi_t << endl;
		    
	
		    	double eps = uniform();
		    	if(eps*mod_psi_s*mod_psi_s <= mod_psi_t*mod_psi_t){ // accept
					s = t;
					mod_psi_s = mod_psi_t;
					spinstate_s = spinstate_t;
					accepted++;
				}; 
				

				if( (j > thermalization_time) && (j % auto_correlation_time) == 0){
					
					//cout << j << " , " << thermalization_time << " , " << auto_correlation_time << endl;

					for(int i=0;i<N_sites*N_symms;++i){
						spin_states[k*N_sites*N_symms + i] = spinstate_s[i];
					};
				
					
					phase_psi_s=evaluate_phase(&spinstate_s[0],W_fc_real,W_fc_imag,N_sites,N_fc);
					
					
					ket_states[k] = s;
					phase_kets[k]=phase_psi_s;
					mod_kets[k]=mod_psi_s;


					k++;
					
				};	

				j++;

		    };

		    return accepted;

		};



};






////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////



template<class I>
int update_offdiag(const int n_op,
						  const char opstr[],
						  const int indx[],
						  const double A,
						  const int Ns,
						  const	I ket[], // col
						  		I bra_rep[],
						  		npy_int8 spin_bra[], // row
						  		npy_int32 ket_index[],
						  		double M[]
						  )
{	int N=_L*_L;
	
	
	#pragma omp parallel
	{		

		int a=Ns/(100*omp_get_num_threads());
		int b=1;

		cout << omp_get_num_threads() << " , " << omp_get_max_threads() << endl;

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

			// 	M[l] = A;
			// 	ket_index[l]=j;

			// 	bra_rep[l] = rep_int_to_spinstate(N,r,&spin_bra[N*N_symms*l]);

			// 	l++;
			// }
			

			double m = A;
			I r = ket[j];

			op(r,m,n_op,opstr,indx,N);
			//bool pcon_bool = (count_bits(r)==N/2);

			//cout << pcon_bool << " , " << m << endl;

			if(m!=0.0){ // r state in same particle-number sector

				
				// bra_rep[l] = ref_state(r);
				// int_to_spinstate(N,r,&spin_bra[N*N_symms*l]);
				
				// bra_rep[l] = rep_int_to_spinstate(N,r,&spin_bra[N*N_symms*l]);
				// M[l] = m;
				// ket_index[l]=j;
				// l++;

				bra_rep[j] = rep_int_to_spinstate(N,r,&spin_bra[N*N_symms*j]);
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
{	//#pragma omp parallel
	{		
		//const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);
		//#pragma omp for schedule(dynamic,chunk) 
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
				const double psi_bras[],
				const double phase_psi_bras[]
	)
{
	int n_cum=0;

	for(int l=0;l<Ns;l++){

		int n=n_per_term[l];

		for(int i=0;i<n;i++){

			int j=n_cum+i;
			
			//cout << l << " , "<< i << " , "<< n << " , "<<j<< " , " << Eloc_cos[ket_ind[j]] << " , " << ket_ind[j] << " , " << psi_bras[j] << " , " << std::cos(phase_psi_bras[j]) << endl;
			 
			Eloc_cos[ket_ind[j]] += MEs[j] * std::exp(psi_bras[j]) * std::cos(phase_psi_bras[j]);
			Eloc_sin[ket_ind[j]] += MEs[j] * std::exp(psi_bras[j]) * std::sin(phase_psi_bras[j]);
		}

		n_cum+=n;
	}
}








