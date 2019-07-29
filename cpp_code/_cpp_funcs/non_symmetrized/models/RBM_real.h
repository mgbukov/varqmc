using namespace std;
#include <iostream>
#include <cmath>
#include <omp.h>


inline double relu(double x){
	return (0.0 < x) ? x : 0.0 ;
}

inline double logcosh(double x){
	return std::log(std::cosh(x)) ;
}

inline double logmodsinh(double x){
	return ((x > 0) ? 1 : ((x < 0) ? -1 : 0))*std::log(std::cosh(x)) ;
}


////////////////////////////////////////////////////////////////////
template<class I, class T>
inline void rbm(
					const I s,
					T spinstate[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc,
					double *mod_out,
					double *phase_out
			)
{
	
	I  t = s;
	I one=1;

	// compute spin configuration
	for(int _i=0; _i<N_sites; _i++){
		npy_int8 n = (t / (one<<(N_sites - _i-1)) ) % 2;
		spinstate[_i] = 2 * ( (n + 2) % 2 ) - 1; // [-1,+1] state representation
	}
	

	/////////////////////////////////////////
	// apply rbm
	double local_mod_out=0.0;
	double local_phase_out=0.0;

	for(int _j=0; _j<N_fc; ++_j){

		double Re_Ws=0.0;
		double Im_Ws=0.0;

		for(int _k=0; _k<N_sites; ++_k){
			Re_Ws += spinstate[_k] * W_fc_real[_k + _j*N_sites];
			Im_Ws += spinstate[_k] * W_fc_imag[_k + _j*N_sites];
		}

		double Re = std::cos(Im_Ws)*std::cosh(Re_Ws);
		double Im = std::sin(Im_Ws)*std::sinh(Re_Ws);

		local_mod_out += 0.5*std::log(Re*Re + Im*Im);
		local_phase_out += std::atan2(Im,Re);
	}

	*mod_out+=local_mod_out;
	*phase_out+=local_phase_out;


}


template<class T>
inline void rbm_mod(
					T spinstate[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc,
					double *out
			)
{
	
	
	// apply rbm
	double local_out=0.0;

	for(int _j=0; _j<N_fc; ++_j){

		double Re_Ws=0.0;
		double Im_Ws=0.0;

		for(int _k=0; _k<N_sites; ++_k){
			Re_Ws += spinstate[_k] * W_fc_real[_k + _j*N_sites];
			Im_Ws += spinstate[_k] * W_fc_imag[_k + _j*N_sites];
		}

		double Re = std::cos(Im_Ws)*std::cosh(Re_Ws);
		double Im = std::sin(Im_Ws)*std::sinh(Re_Ws);

		local_out += 0.5*std::log(Re*Re + Im*Im);
	}

	*out+=local_out;

							

}


template<class T>
inline void rbm_phase(
					T spinstate[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc,
					double *out
			)
{
	

	// apply rbm
	double local_out=0.0;

	for(int _j=0; _j<N_fc; ++_j){

		double Re_Ws=0.0;
		double Im_Ws=0.0;

		for(int _k=0; _k<N_sites; ++_k){
			Re_Ws += spinstate[_k] * W_fc_real[_k + _j*N_sites];
			Im_Ws += spinstate[_k] * W_fc_imag[_k + _j*N_sites];
		}

		double Re = std::cos(Im_Ws)*std::cosh(Re_Ws);
		double Im = std::sin(Im_Ws)*std::sinh(Re_Ws);
	
		//local_out += std::atan( std::tan(Im_Ws)*std::tanh(Re_Ws)  );
		local_out += std::atan2(Im,Re);
		
	}

	*out+=local_out;
	

}



////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////


template<class I>
void evaluate_rbm(
					I states[],
					npy_int8 spinstate[],
					double mod_psi[],
					double phase_psi[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc,
					const int Ns
					
	)
{	
	
	for(int i=0;i<Ns;i++){

		double mod_out=0.0;
		double phase_out=0.0;

		I s = states[i];
		
		rbm(s,&spinstate[i*N_sites], &W_fc_real[0],&W_fc_imag[0], N_sites,N_fc, &mod_out, &phase_out);

		mod_psi[i]=std::exp(mod_out);
		phase_psi[i]=phase_out;
		//phase_psi[i]=fmod(fmod(phase_out, 2.0*M_PI) + 2.0*M_PI, 2.0*M_PI);
		
	}
	
}





template<class I>
double evaluate_mod(
					I s,
					npy_int8 spinstate[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc
					
	)
{	
	
	double out=0.0;
	
	rbm_mod(&spinstate[0], &W_fc_real[0],&W_fc_imag[0], N_sites,N_fc, &out);
	
	return std::exp(out);
}


template<class I>
double evaluate_phase(
					I s,
					npy_int8 spinstate[],
					const double W_fc_real[],
					const double W_fc_imag[],
					const int N_sites,
					const int N_fc
					
	)
{	
	
	double out=0.0;
	
	rbm_phase(&spinstate[0], &W_fc_real[0],&W_fc_imag[0], N_sites,N_fc, &out);
	
	return out;
	
}


