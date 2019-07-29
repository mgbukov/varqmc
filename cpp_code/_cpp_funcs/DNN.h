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

 // fill in pointer to a_fc[j], fix sizes of arrays (if stored in cache it'll fly)
template<class T>
inline void deep_layer(T spinstate[],
					const double W_fc[],
					const double b_fc[],
					const int N_sites,
					const int N_fc,
					double sigma(double),
					double a_fc[]
					// double (*sigma)(double)  // to call inside func (*sigma)(args); call it &sigma
			)
{
	#pragma omp parallel for schedule(static) // NOTE: constant workload
	for(int j=0; j<N_fc; ++j){
		double aux_fc=0.0;
		for(int k=0; k<N_sites; ++k){
			aux_fc += spinstate[k]*W_fc[k+j*N_sites];
		}
		aux_fc += b_fc[j];
		a_fc[j]=sigma(aux_fc);
		
	}
	
}


template<class T>
inline void output_layer(T a_fc_previous[],
					const double W_fc[],
					const double b_fc[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_fc_previous,
					const int N_fc,
					double sigma(double),
					double *out
			)
{
	
	# pragma omp parallel shared(out)
	{

		double local_out=0.0;

		#pragma omp for schedule(static) // NOTE: constant workload
		for(int j=0; j<N_fc; ++j){

			double aux_fc=0.0;
			for(int k=0; k<N_fc_previous; ++k){
				aux_fc += a_fc_previous[k]*W_fc[k+j*N_fc_previous];
			}
			aux_fc += b_fc[j];

			local_out+=sigma(aux_fc)*W_fc_out[j];
		}


		#pragma omp critical
		{	
			//cout << local_out << endl;
			*out+=local_out;
		}
		
	}

	*out+=b_fc_out[0];			

}



double evaluate_mod(npy_int8 spinstate[],
					const double W_fc1[],
					const double b_fc1[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_sites,
					const int N_fc1
					
	)
{	
	
	double out=0.0;
	
	output_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],&W_fc_out[0],&b_fc_out[0],N_sites,N_fc1, relu, &out);
	
	return std::exp(out);
}


/*

double evaluate_mod(npy_int8 spinstate[],
					const double W_fc1[],
					const double b_fc1[],
					const double W_fc2[],
					const double b_fc2[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_sites,
					const int N_fc1,
					const int N_fc2
	)
{	
	
	std::vector<double> a_fc1(N_fc1);
	double out=0.0;
	
	deep_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],N_sites,N_fc1, relu, &a_fc1[0]);
	output_layer(&a_fc1[0],&W_fc2[0],&b_fc2[0],&W_fc_out[0],&b_fc_out[0],N_fc1,N_fc2, relu, &out);
	
	return std::exp(out);
}

*/

/*



double evaluate_mod(npy_int8 spinstate[],
					const double W_fc1[],
					const double b_fc1[],
					const double W_fc2[],
					const double b_fc2[],
					const double W_fc3[],
					const double b_fc3[],
					const double W_fc4[],
					const double b_fc4[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_sites,
					const int N_fc1,
					const int N_fc2,
					const int N_fc3,
					const int N_fc4
	)
{	
	
	std::vector<double> a_fc1(N_fc1);
	std::vector<double> a_fc2(N_fc2);
	std::vector<double> a_fc3(N_fc3);
	double out=0.0;
	
	deep_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],N_sites,N_fc1, relu, &a_fc1[0]);
	deep_layer(&a_fc1[0],&W_fc2[0],&b_fc2[0],N_fc1, N_fc2, relu, &a_fc2[0]);
	deep_layer(&a_fc2[0],&W_fc3[0],&b_fc3[0],N_fc2, N_fc3, relu, &a_fc3[0]);
	
	output_layer(&a_fc3[0],&W_fc4[0],&b_fc4[0],&W_fc_out[0],&b_fc_out[0],N_fc3,N_fc4, relu, &out);
	
	return std::exp(out);
}

*/


///////////////////////////////////////////////////////////


/*

double evaluate_phase(npy_int8 spinstate[],
					const double W_fc1[],
					const double b_fc1[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_sites,
					const int N_fc1
	)
{	
	double out=0.0;

	//output_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],&W_fc_out[0],&b_fc_out[0],N_sites,N_fc1, std::tanh, &out);
	output_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],&W_fc_out[0],&b_fc_out[0],N_sites,N_fc1, relu, &out);
				
	return out;
	//return fmod(fmod(out, 2.0*M_PI) + 2.0*M_PI, 2.0*M_PI);
}

*/



double evaluate_phase(npy_int8 spinstate[],
					const double W_fc1[],
					const double b_fc1[],
					const double W_fc2[],
					const double b_fc2[],
					const double W_fc_out[],
					const double b_fc_out[],
					const int N_sites,
					const int N_fc1,
					const int N_fc2
	)
{	
	std::vector<double> a_fc1(N_fc1);
	double out=0.0;

	//deep_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],N_sites,N_fc1, std::tanh, &a_fc1[0]);
	//output_layer(&a_fc1[0],&W_fc2[0],&b_fc2[0],&W_fc_out[0],&b_fc_out[0],N_fc1,N_fc2, std::tanh, &out);

	deep_layer(&spinstate[0],&W_fc1[0],&b_fc1[0],N_sites,N_fc1, relu, &a_fc1[0]);
	output_layer(&a_fc1[0],&W_fc2[0],&b_fc2[0],&W_fc_out[0],&b_fc_out[0],N_fc1,N_fc2, relu, &out);
	
	return out;			
	//return fmod(fmod(out, 2.0*M_PI) + 2.0*M_PI, 2.0*M_PI);
}


