/*
 * The plant. 
 *
 */
#ifndef __PLANT_H
#define __PLANT_H


//#include <vector>
#include "../../core/lib/math/matrix.cpp"
#include "helperControl.h"
#include "omp.h"
#include <iostream>
#include <algorithm>    // std::fill_n

using std::invalid_argument;
using namespace std;

// Make sure to define the correct path of the data folder
const std::string DATAFOLDER = "../plantData/";

template<class templElement>
struct Costs{
	std::vector<templElement> diagQ;
	std::vector<templElement> diagR;
	double lamg;
	double lamy;
	double lamu;
};

template<class templElement>
class Plant{

private:
	/* State at current time, vector in R^n*/
	lbcrypto::Matrix<templElement> x;
	/* State at next time step, vector in R^n*/	
	lbcrypto::Matrix<templElement> xplus;	
	/* Control action at current time, vector in R^m */
	lbcrypto::Matrix<templElement> u;
	/* Measurement at current time, vector in R^n */
	lbcrypto::Matrix<templElement> y;
	/* Set point at current time, vector in R^(pN) */
	lbcrypto::Matrix<templElement> r;	
	/* Previous values of measurements, vector in R^(pM) */
	lbcrypto::Matrix<templElement> yini;	
	/* Previous values of actions, vector in R^(mM) */
	lbcrypto::Matrix<templElement> uini;		
	/* State Matrix in R^(n x n) */
	lbcrypto::Matrix<templElement> A;
	/* State-input Matrix in R^(n x m) */
	lbcrypto::Matrix<templElement> B;
	/* Measurement Matrix in R^(p x n) */
	lbcrypto::Matrix<templElement> C;
	/* Measurements over time steps */
	lbcrypto::Matrix<templElement> Ym;
	/* Actions over time steps */
	lbcrypto::Matrix<templElement> Um;	
	/* Precollected output past Hankel Matrix that keeps increasing with time in the online case */
	lbcrypto::Matrix<templElement> Yp;
	/* Precollected output future Hankel Matrix that keeps increasing with time in the online case */
	lbcrypto::Matrix<templElement> Yf;
	/* Precollected intput past Hankel Matrix that keeps increasing with time in the online case*/
	lbcrypto::Matrix<templElement> Up;		
	/* Precollected intput future Hankel Matrix that keeps increasing with time in the online case */
	lbcrypto::Matrix<templElement> Uf;		
	/* The Hankel matrix for output measurements */
	lbcrypto::Matrix<templElement> HY;
	/* The Hankel matrix for inputs */
	lbcrypto::Matrix<templElement> HU;		
	/* The output trajectory that contains all the information in HY */
	lbcrypto::Matrix<templElement> TrajectY;
	/* The intput trajectory that contains all the information in HU */
	lbcrypto::Matrix<templElement> TrajectU;	
	/* LQR-like gains */
	lbcrypto::Matrix<templElement> Kuini;
	lbcrypto::Matrix<templElement> Kyini;
	lbcrypto::Matrix<templElement> Kur;
	lbcrypto::Matrix<templElement> Kyr;
	/* Intermediate matrix of which we keep track of its inverse */
	lbcrypto::Matrix<templElement> M_1;
	/* Cost for reference following output */
	lbcrypto::Matrix<templElement> Q;
	/* Cost for reference following input */
	lbcrypto::Matrix<templElement> R;
	/* Cost for regularization of parameters */
	double lamg;
	/* Cost for regularization output difference */
	double lamy;
	/* Cost for regularization input difference */
	double lamu;			
	/* Aggregate of costs */
	Costs<templElement> costs;
	/* Standard deviation for one component of the process noise */
	double W;
	/* Standard deviation for one component of the measurement noise */
	double V;	

public:

	/* Time step k */
	uint32_t k;
	/* Dimension of state x */		
	uint32_t n;
	/* Dimension of control input u */		
	uint32_t m;
	/* Dimension of output y */		
	uint32_t p;		
	/* Size of past inputs */
	uint32_t pu;
	/* Size of future inputs */
	uint32_t fu;
	/* Size of past outputs */
	uint32_t py;
	/* Size of future outputs */
	uint32_t fy;			
	/* Number of columns of the Hankel matrices */
	uint32_t S;	
	/* Initial number of past precollected values */
	uint32_t M;
	/* Initial number of future precollected values */
	uint32_t N;	

	// For debugging
	/* Last column of the Hankel matrix for output measurements */
	lbcrypto::Matrix<templElement> hY;
	/* Last column of the Hankel matrix for inputs */
	lbcrypto::Matrix<templElement> hU;			

	/* Empty constructor */
	Plant();

	/* Constructor that gets the dimension of the variables */
	Plant(const uint32_t _n, const uint32_t _m, const uint32_t _p);

	/* Constructor that initializes the plant with all parameters */
	Plant(const lbcrypto::Matrix<templElement>& _A, const lbcrypto::Matrix<templElement>& _B, const lbcrypto::Matrix<templElement>& _C, const lbcrypto::Matrix<templElement>& _x0, double _W = 0, double _V = 0);

	/* Virtual destructor */
	virtual ~Plant();

	/* Update state x given the input u*/
	void updatex(const lbcrypto::Matrix<templElement>& _u);
	void updatex(const std::vector<templElement>& _u);

	/* Update state x given the input u in the online case*/
	void onlineUpdatex(const lbcrypto::Matrix<templElement>& _u);	
	void onlineUpdatex(const std::vector<templElement>& _u);	

	/* Get current state x */
	lbcrypto::Matrix<templElement>& getx();

	/* Get current input u */
	lbcrypto::Matrix<templElement>& getu();

	/* Get current measurement y */
	lbcrypto::Matrix<templElement>& gety();		

	/* Get set point r*/
	lbcrypto::Matrix<templElement>& getr();			

	/* Set the value of the set point r*/
	void setr(const lbcrypto::Matrix<templElement>& _r);	

	/* Get yini*/
	lbcrypto::Matrix<templElement>& getyini();			

	/* Set the value of yini */
	void setyini(const lbcrypto::Matrix<templElement>& _yini);		

	void setyini(const std::vector<templElement>& _yini);				

	/* Get uini*/
	lbcrypto::Matrix<templElement>& getuini();			

	/* Set the value of uini */
	void setuini(const lbcrypto::Matrix<templElement>& _uini);		

	void setuini(const std::vector<templElement>& _uini);			

	/* Get gain corresponding to the reference Kur */
	lbcrypto::Matrix<templElement>& getKur();	

	/* Get gain corresponding to uini */
	lbcrypto::Matrix<templElement>& getKuini();

	/* Get gain corresponding to yini */
	lbcrypto::Matrix<templElement>& getKyini();						

	/* Get intermediate matrix inverse */
	lbcrypto::Matrix<templElement>& getM_1();						

	/* Get Hankel matrix of output measurements */
	lbcrypto::Matrix<templElement>& getHY();

	/* Get Hankel matrix of input measurements */
	lbcrypto::Matrix<templElement>& getHU();

	/* Get output trajectory */
	lbcrypto::Matrix<templElement>& getTrajectY();	

	/* Get intput trajectory */
	lbcrypto::Matrix<templElement>& getTrajectU();	

	/* Get output measurements */
	lbcrypto::Matrix<templElement>& getYm();	

	/* Get intput measurements */
	lbcrypto::Matrix<templElement>& getUm();		

	/* Get costs: diag(Q), diag(R), lamg, lamy, lamu */
	Costs<templElement>& getCosts();	

	/* Set precollected values, with flagU = 1 if setting Up and Uf and flagU = 0 if setting Yp and Yf */
	void precollect(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _pu, const uint32_t _fu, const uint32_t _flagU);

	/* Set directly the Hankel matrices after the trajectory concatenation, with flagU = 1 if setting Up and Uf and flagU = 0 if setting Yp and Yf */
	void precollectH(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _pu, const uint32_t _fu, const uint32_t _flagU);	

	/* Create block Hankel Matrix of orde L from precollected vectors _Hd */
	lbcrypto::Matrix<templElement> blockHankel(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _L);

	/* Set costs */
	void setCosts(const lbcrypto::Matrix<templElement>& _Q, const lbcrypto::Matrix<templElement>& _R, const double _lamg, const double _lamy, const double _lamu);

	/* Compute LQR-like gain */
	void constLQR();

	/* Compute LQR-like gain but provide the matrix inverse because it is too slow to compute*/
	void constLQR(const lbcrypto::Matrix<templElement>& _M_1);

	/* Compute the online gains, based on the inverse of the previous matrix gain*/
	void onlineLQR();

	/* Compute new control input */
	lbcrypto::Matrix<templElement> updateu(const lbcrypto::Matrix<templElement>& _r, const lbcrypto::Matrix<templElement>& _uini, const lbcrypto::Matrix<templElement>& _yini);

	/* Update Hankel matrices and recompute inverse of matrix M after trajectory concatenation and update parameters*/
	void concatLQR(const lbcrypto::Matrix<templElement>& _temp_y, const lbcrypto::Matrix<templElement>& _temp_u);

	/* Print input-output measurements */
	void printYU();
};


/* Empty constructor */
template<class templElement>
Plant<templElement>::Plant(){}

/* Constructor that gets the dimension of the variables. */
template<class templElement>
Plant<templElement>::Plant(const uint32_t _n, const uint32_t _m, const uint32_t _p)
{
	n = _n; m = _m; p = _p;
	k = 0;
}

/* Constructor that initializes the plant with all parameters. */
template<class templElement>
Plant<templElement>::Plant(const lbcrypto::Matrix<templElement>& _A, const lbcrypto::Matrix<templElement>& _B, const lbcrypto::Matrix<templElement>& _C, const lbcrypto::Matrix<templElement>& _x0, double _W, double _V)
{
	A = _A; B = _B; C = _C;
	x = _x0;
	xplus = _x0;
	n = A.GetRows();
	m = B.GetCols();
	p = C.GetRows();
	k = 0;
	y = _C * _x0;

	W = _W;
	V = _V;

}

/* Destructor */
template<class templElement>
Plant<templElement>::~Plant() {}

/* Update state x given the input u*/
template<class templElement>
void Plant<templElement>::updatex(const lbcrypto::Matrix<templElement>& _u)
{
	auto zeroAlloc = [=]() { return 0; };
	/*if(k==0) // Necessary hack when in the lbcrypto::Matrix copy constructor the zeroAlloc function was not copied.
	{
		auto zeroAlloc = [=]() { return 0; };
		A.SetAllocator(zeroAlloc);
		x.SetAllocator(zeroAlloc);
		B.SetAllocator(zeroAlloc);
		u.SetAllocator(zeroAlloc);
		xplus.SetAllocator(zeroAlloc);
		y.SetAllocator(zeroAlloc);
		C.SetAllocator(zeroAlloc);
	}
	*/
	
	// Update the current control action
	u = _u;

	// Update the current state
	x = xplus;	

	// Noise generation
	std::random_device device_random;
	std::default_random_engine generator(device_random());	
 	std::normal_distribution<> distribution_x(0, W);
 	std::normal_distribution<> distribution_y(0, V);	

 	std::vector<templElement> vec_w(n);
 	std::vector<templElement> vec_v(p);

 	for (size_t i = 0; i < n; i ++)
 		vec_w[i] = (templElement)distribution_x(generator);
 	for (size_t i = 0; i < p; i ++)
  		vec_v[i] = (templElement)distribution_y(generator);		

	lbcrypto::Matrix<templElement> w = lbcrypto::Matrix<templElement>(zeroAlloc, n, 1);
	w.SetCol(0, vec_w);
	lbcrypto::Matrix<templElement> v = lbcrypto::Matrix<templElement>(zeroAlloc, p, 1);
	v.SetCol(0, vec_v);

	// Compute the state at the next time step, that is perturbed by process noise
    xplus = A * x + B * u + w;

    // Measure the output, that is perturbed by measurement noise
	y = C * x + v;

	// Add the current measurement to the history of measurements
	// Add the current control input to the history of control inputs	
	if(k == 0){
		Ym = y;
		Um = _u;
	}
	else{
		Ym.HStack(y);
		Um.HStack(u);
	}

	// Increase the time step count
	k += 1;
}

/* Update state x given the input u*/
template<class templElement>
void Plant<templElement>::updatex(const std::vector<templElement>& _u)
{
	auto zeroAlloc = [=]() { return 0; };
	lbcrypto::Matrix<templElement> u = lbcrypto::Matrix<templElement>(zeroAlloc, _u.size(),1);
	u.SetCol(0,_u);
	updatex(u);
}

/* Get current state x */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getx()
{
	return x;
}
	

/* Get current input u */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getu()
{
	return u;
}

/* Get current measurement y */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::gety()
{
	return y;
}	

/* Get set point r*/
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getr()
{
	return r;
}

/* Set the value of the set point r*/
template<class templElement>
void Plant<templElement>::setr(const lbcrypto::Matrix<templElement>& _r)
{
	r = _r;
}			

/* Get yini*/
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getyini()
{
	return yini;
}

/* Set the value of yini*/
template<class templElement>
void Plant<templElement>::setyini(const lbcrypto::Matrix<templElement>& _yini)
{
	yini = _yini;
}		

/* Set the value of yini*/
template<class templElement>
void Plant<templElement>::setyini(const std::vector<templElement>& _yini)
{
	yini.SetCol(0,_yini);
}

/* Get uini*/
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getuini()
{
	return uini;
}

/* Set the value of uini*/
template<class templElement>
void Plant<templElement>::setuini(const lbcrypto::Matrix<templElement>& _uini)
{
	uini = _uini;
}	
	
/* Set the value of uini*/
template<class templElement>
void Plant<templElement>::setuini(const std::vector<templElement>& _uini)
{
	uini.SetCol(0,_uini);
}

/* Get gain corresponding to the reference Kur */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getKur()
{
	return Kur;
}

/* Get gain corresponding to uini */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getKuini()
{
	return Kuini;
}

/* Get gain corresponding to yini */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getKyini()
{
	return Kyini;
}		

/* Get intermediate matrix inverse */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getM_1()
{
	return M_1;
}	

/* Get Hankel matrix of output measurements */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getHY()
{
	return HY;
}

/* Get Hankel matrix of input measurements */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getHU()
{
	return HU;
}

/* Get output trajectory */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getTrajectY()
{
	return TrajectY;
}

/* Get input trajectory */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getTrajectU()
{
	return TrajectU;
}

/* Get output measurements */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getYm()
{
	return Ym;
}

/* Get input measurements */
template<class templElement>
lbcrypto::Matrix<templElement>& Plant<templElement>::getUm()
{
	return Um;
}

/* Get costs: diag(Q), diag(R), lamg, lamy, lamu */
template<class templElement>
Costs<templElement>& Plant<templElement>::getCosts()
{
	return costs;
}

/* Set precollected values, with flagU = 1 if setting HU and flagU = 0 if setting HY */
template<class templElement>
void Plant<templElement>::precollect(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _pu, const uint32_t _fu, const uint32_t _flagU)
{
	auto zeroAlloc = [=]() { return 0; };

	if(_flagU)
	{
		TrajectU = lbcrypto::Matrix<templElement>(zeroAlloc, _Hd.GetRows()*_Hd.GetCols(), 1);
		for (size_t j = 0; j < _Hd.GetCols(); j ++)		
			for (size_t i = 0; i < _Hd.GetRows(); i ++ )
				TrajectU(j*_Hd.GetRows() + i, 0) = _Hd(i,j);

		lbcrypto::Matrix<templElement> H = Plant::blockHankel(_Hd, uint32_t((_pu+_fu)/m));
		Up = H.ExtractRows(0,_pu-1);
		Uf = H.ExtractRows(_pu,_pu+_fu-1);
		pu = _pu; fu = _fu; 
		S = Up.GetCols();
		HU = H;

		hU = H.ExtractCol(H.GetCols()-1);

		// cout << "Up\n" << Up << endl;
		// cout << "Uf\n" << Uf << endl;
	}
	else
	{
		TrajectY = lbcrypto::Matrix<templElement>(zeroAlloc, _Hd.GetRows()*_Hd.GetCols(), 1);
		for (size_t j = 0; j < _Hd.GetCols(); j ++)		
			for (size_t i = 0; i < _Hd.GetRows(); i ++ )
				TrajectY(j*_Hd.GetRows() + i, 0) = _Hd(i,j);

		lbcrypto::Matrix<templElement> H = Plant::blockHankel(_Hd, uint32_t((_pu+_fu)/p));
		Yp = H.ExtractRows(0,_pu-1);
		Yf = H.ExtractRows(_pu,_pu+_fu-1);	
		py = _pu; fy = _fu;
		S = Yp.GetCols();	
		HY = H;

		hY = H.ExtractCol(H.GetCols()-1);

		// cout << "Yp\n" << Yp << endl;
		// cout << "Yf\n" << Yf << endl;		
	}		

}

/* Set directly the Hankel matrices after the trajectory concatenation, with flagU = 1 if setting Up and Uf and flagU = 0 if setting Yp and Yf */
template<class templElement>
void Plant<templElement>::precollectH(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _pu, const uint32_t _fu, const uint32_t _flagU)
{
	if(_flagU)
	{
		Up = _Hd.ExtractRows(0,_pu-1);
		Uf = _Hd.ExtractRows(_pu,_pu+_fu-1);
		pu = _pu; fu = _fu; 
		S = Up.GetCols();
		HU = _Hd;

		hU = _Hd.ExtractCol(_Hd.GetCols()-1);
	}
	else
	{
		Yp = _Hd.ExtractRows(0,_pu-1);
		Yf = _Hd.ExtractRows(_pu,_pu+_fu-1);
		py = _pu; fy = _fu;
		S = Yp.GetCols();	
		HY = _Hd;

		hY = _Hd.ExtractCol(_Hd.GetCols()-1);
	}		

}	



/* Create block Hankel Matrix from precollected vectors _Hd */
template<class templElement>
lbcrypto::Matrix<templElement> Plant<templElement>::blockHankel(const lbcrypto::Matrix<templElement>& _Hd, const uint32_t _L)
{
	uint32_t T_ = _Hd.GetCols();
	uint32_t m_ = _Hd.GetRows();

	auto zeroAlloc = [=]() { return 0; };

	lbcrypto::Matrix<templElement> H = lbcrypto::Matrix<templElement>(zeroAlloc, _L * m_, T_ - _L + 1);

// #pragma omp parallel for	
	for(int j = 0; j < T_ - _L + 1; j++)
		for(int i = 0; i < _L; i++)
			for(int l = 0; l < m_; l++)
				H(m_*i + l,j) = _Hd(l,i+j);

	return H;
}

/* Set costs */
template<class templElement>
void Plant<templElement>::setCosts(const lbcrypto::Matrix<templElement>& _Q, const lbcrypto::Matrix<templElement>& _R, const double _lamg, const double _lamy, const double _lamu)
{
	Q = _Q;
	R = _R;
	lamg = _lamg;
	lamy = _lamy;
	lamu = _lamu;

	costs = {extractDiag2Vec(Q, 0), extractDiag2Vec(R, 0), lamg, lamy, lamu};

}

/* Compute LQR-like gain */
template<class templElement>
void Plant<templElement>::constLQR()
{
	auto zeroAlloc = [=]() { return templElement(0); };
	lbcrypto::Matrix<templElement> I = lbcrypto::Matrix<templElement>(zeroAlloc, Yf.GetCols(), Yf.GetCols());
	I.Identity();
	lbcrypto::Matrix<templElement> K = I.ScalarMult(lamg) + Yf.Transpose() * Q * Yf + Uf.Transpose() * R * Uf + (Yp.Transpose() * Yp).ScalarMult(lamy) + (Up.Transpose() * Up).ScalarMult(lamu);

	templElement determinant( zeroAlloc() );
	K.Determinant(&determinant);
	K = K.CofactorMatrix();
	K = K.ScalarMult(templElement(1)/determinant); // this is equivalent to M_1 from constLQR and onlineLQR

	lbcrypto::Matrix<templElement> Kuf = Uf * K;
	lbcrypto::Matrix<templElement> Kyf = Yf * K;
	lbcrypto::Matrix<templElement> YQ = Yf.Transpose() * Q;
	Kur = Kuf * YQ; 
	Kyr = Kyf * YQ;
	Kuini = (Kuf * Up.Transpose()).ScalarMult(lamu);
	Kyini = (Kuf * Yp.Transpose()).ScalarMult(lamy);

	M_1 = K;
}

/* Compute LQR-like gain but provide the matrix inverse because it is too slow to compute*/
template<class templElement>
void Plant<templElement>::constLQR(const lbcrypto::Matrix<templElement>& _M_1)
{
	lbcrypto::Matrix<templElement> Kuf = Uf * _M_1;
	lbcrypto::Matrix<templElement> Kyf = Yf * _M_1;
	lbcrypto::Matrix<templElement> YQ = Yf.Transpose() * Q;
	Kur = Kuf * YQ; 
	Kyr = Kyf * YQ;
	Kuini = (Kuf * Up.Transpose()).ScalarMult(lamu);
	Kyini = (Kuf * Yp.Transpose()).ScalarMult(lamy);

	M_1 = _M_1;

	// cout.precision(8);
	// cout << "_M_1\n" << _M_1 << endl;
	// cout << "Kuf:\n" << Kuf << endl;
	// cout << "YQ:\n" << YQ << endl;
	// cout << "Kur:\n" << Kur << endl;
	// cout << "Kuini:\n" << Kuini << endl;
	// cout << "Kyini:\n" << Kyini << endl;
}

/* Compute new control input */
template<class templElement>
lbcrypto::Matrix<templElement> Plant<templElement>::updateu(const lbcrypto::Matrix<templElement>& _r, const lbcrypto::Matrix<templElement>& _uini, const lbcrypto::Matrix<templElement>& _yini)
{
	if (k == 0)
		r = _r;
	else
		if (r != _r)
			r = _r;

	uini = _uini;
	yini = _yini;

	lbcrypto::Matrix<templElement> Upred = Kur * _r + Kuini * _uini + Kyini * _yini;

	return Upred.ExtractRows(0, m-1);
}

/* Compute the online gains, based on the inverse of the previous matrix gain*/
template<class templElement>
void Plant<templElement>::onlineLQR()
{
	// Compute the new LQR-like gains
	lbcrypto::Matrix<templElement> Kuf = Uf * M_1;
	lbcrypto::Matrix<templElement> Kyf = Yf * M_1;
	lbcrypto::Matrix<templElement> YQ = Yf.Transpose() * Q;
	Kur = Kuf * YQ; 
	Kyr = Kyf * YQ;
	Kuini = (Kuf * Up.Transpose()).ScalarMult(lamu);
	Kyini = (Kuf * Yp.Transpose()).ScalarMult(lamy);
}

/* Update x in the online case given the action u*/
template<class templElement>
void Plant<templElement>::onlineUpdatex(const lbcrypto::Matrix<templElement>& _u)
{
	auto zeroAlloc = [=]() { return 0; };
	/*if(k==0) // Necessary hack when in the lbcrypto::Matrix copy constructor the zeroAlloc function was not copied.
	{
		auto zeroAlloc = [=]() { return 0; };
		A.SetAllocator(zeroAlloc);
		x.SetAllocator(zeroAlloc);
		B.SetAllocator(zeroAlloc);
		u.SetAllocator(zeroAlloc);
		xplus.SetAllocator(zeroAlloc);
		y.SetAllocator(zeroAlloc);
		C.SetAllocator(zeroAlloc);
	}
	*/

	// Update the current control action
	u = _u;

	// Update the current state
	x = xplus;	

	// Noise generation
	std::random_device device_random;
	std::default_random_engine generator(device_random());	
 	std::normal_distribution<> distribution_x(0, W);
 	std::normal_distribution<> distribution_y(0, V);	

 	std::vector<templElement> vec_w(n);
 	std::vector<templElement> vec_v(p);

 	for (size_t i = 0; i < n; i ++)
 		vec_w[i] = (templElement)distribution_x(generator);
 	for (size_t i = 0; i < p; i ++)
  		vec_v[i] = (templElement)distribution_y(generator);		

	lbcrypto::Matrix<templElement> w = lbcrypto::Matrix<templElement>(zeroAlloc, n, 1);
	w.SetCol(0, vec_w);
	lbcrypto::Matrix<templElement> v = lbcrypto::Matrix<templElement>(zeroAlloc, p, 1);
	v.SetCol(0, vec_v);

	// Compute the state at the next time step, that is perturbed by process noise
    xplus = A * x + B * u + w;

    // Measure the output, that is perturbed by measurement noise
	y = C * x + v;

	// Add the current measurement to the history of measurements
	// Add the current control input to the history of control inputs	
	if(k == 0){
		Ym = y;
		Um = _u;
	}
	else{
		Ym.HStack(y);
		Um.HStack(u);
	}	

	// Increase the time step count
	k += 1;	

	std::vector<templElement> yV = mat2Vec(y);
	std::vector<templElement> uV = mat2Vec(u);

	std::vector<templElement> hYV = mat2Vec(hY);
	Rotate(hYV, p);
	std::copy(yV.begin(),yV.end(),hYV.begin()+hYV.size()-p);
	hY.SetCol(0,hYV);

	lbcrypto::Matrix<templElement> yp = lbcrypto::Matrix<templElement>(zeroAlloc, py, 1);
	lbcrypto::Matrix<templElement> yf = lbcrypto::Matrix<templElement>(zeroAlloc, fy, 1);
	yp.SetCol(0,std::vector<templElement>(hYV.begin(), hYV.begin()+py));
	yf.SetCol(0,std::vector<templElement>(hYV.begin()+py, hYV.end()));

	std::vector<templElement> hUV = mat2Vec(hU);
	Rotate(hUV, m);
	std::copy(uV.begin(),uV.end(),hUV.begin()+hUV.size()-m);	
	hU.SetCol(0,hUV);

	lbcrypto::Matrix<templElement> up = lbcrypto::Matrix<templElement>(zeroAlloc, pu, 1);
	lbcrypto::Matrix<templElement> uf = lbcrypto::Matrix<templElement>(zeroAlloc, fu, 1);
	up.SetCol(0,std::vector<templElement>(hUV.begin(), hUV.begin()+pu));
	uf.SetCol(0,std::vector<templElement>(hUV.begin()+pu, hUV.end()));	

	lbcrypto::Matrix<templElement> mVec = yf.Transpose() * Q * Yf + uf.Transpose() * R * Uf + (yp.Transpose() * Yp).ScalarMult(lamy) + (up.Transpose() * Up).ScalarMult(lamu);

	lbcrypto::Matrix<templElement> mSc = yf.Transpose() * Q * yf + uf.Transpose() * R * uf + (yp.Transpose() * yp).ScalarMult(lamy) + (up.Transpose()).ScalarMult(lamu) * up + templElement(lamg);

	lbcrypto::Matrix<templElement> M_1mVecT = M_1 * mVec.Transpose();

	lbcrypto::Matrix<templElement> Temp = mSc - mVec * M_1mVecT;

	templElement mSchur = Temp(0,0);
	templElement mSchur_1 = templElement(1)/mSchur;

	lbcrypto::Matrix<templElement> M11 = M_1 + (M_1mVecT * M_1mVecT.Transpose()).ScalarMult(mSchur_1);
	lbcrypto::Matrix<templElement> M12 = M_1mVecT.ScalarMult(-mSchur_1);
	lbcrypto::Matrix<templElement> M21 = M12.Transpose();
	lbcrypto::Matrix<templElement> M22 = lbcrypto::Matrix<templElement>(zeroAlloc, 1, 1); M22(0,0) = mSchur_1;

	M11.HStack(M12);
	M21.HStack(M22);
	M11.VStack(M21);
	M_1 = M11;

	Yp.HStack(yp);
	Up.HStack(up);		
	Yf.HStack(yf);
	Uf.HStack(uf);
	HY.HStack(hY);
	HU.HStack(hU);	
	
}

/* Update state x in the online case given the input u*/
template<class templElement>
void Plant<templElement>::onlineUpdatex(const std::vector<templElement>& _u)
{
	auto zeroAlloc = [=]() { return 0; };
	lbcrypto::Matrix<templElement> u = lbcrypto::Matrix<templElement>(zeroAlloc, _u.size(),1);
	u.SetCol(0,_u);
	onlineUpdatex(u);
}


/* Update Hankel matrices and recompute inverse of matrix M after trajectory concatenation and update parameters*/
template<class templElement>
void Plant<templElement>::concatLQR(const lbcrypto::Matrix<templElement>& _temp_y, const lbcrypto::Matrix<templElement>& _temp_u)
{
	auto zeroAlloc = [=]() { return 0; };

	hY = _temp_y;	
	hU = _temp_u;	

	lbcrypto::Matrix<templElement> yp = hY.ExtractRows(0,py-1);
	lbcrypto::Matrix<templElement> yf = hY.ExtractRows(py,py+fy-1);

	lbcrypto::Matrix<templElement> up = hU.ExtractRows(0,pu-1);
	lbcrypto::Matrix<templElement> uf = hU.ExtractRows(pu,pu+fu-1);

	lbcrypto::Matrix<templElement> mVec = yf.Transpose() * Q * Yf + uf.Transpose() * R * Uf + (yp.Transpose() * Yp).ScalarMult(lamy) + (up.Transpose() * Up).ScalarMult(lamu);

	lbcrypto::Matrix<templElement> mSc = yf.Transpose() * Q * yf + uf.Transpose() * R * uf + (yp.Transpose() * yp).ScalarMult(lamy) + (up.Transpose()).ScalarMult(lamu) * up + templElement(lamg);

	lbcrypto::Matrix<templElement> M_1mVecT = M_1 * mVec.Transpose();

	lbcrypto::Matrix<templElement> Temp = mSc - mVec * M_1mVecT;

	templElement mSchur = Temp(0,0);
	templElement mSchur_1 = templElement(1)/mSchur;

	lbcrypto::Matrix<templElement> M11 = M_1 + (M_1mVecT * M_1mVecT.Transpose()).ScalarMult(mSchur_1);
	lbcrypto::Matrix<templElement> M12 = M_1mVecT.ScalarMult(-mSchur_1);
	lbcrypto::Matrix<templElement> M21 = M12.Transpose();
	lbcrypto::Matrix<templElement> M22 = lbcrypto::Matrix<templElement>(zeroAlloc, 1, 1); M22(0,0) = mSchur_1;

	M11.HStack(M12);
	M21.HStack(M22);
	M11.VStack(M21);
	M_1 = M11;

	Yp.HStack(yp);
	Up.HStack(up);		
	Yf.HStack(yf);
	Uf.HStack(uf);
	HY.HStack(hY);
	HU.HStack(hU);

	S = HU.GetCols();			
	
}

/* Print input-output measurements */
template<class templElement>
void Plant<templElement>::printYU()
{
	std::cout << "Ym = " << Ym << std::endl;
	std::cout << "Um = " << Um << std::endl;	
}

/**
 * Simulates the unencrypted plant control
 */

Plant<complex<double>>* plantInitRoom()
{
	auto zeroAlloc = [=]() { return 0; };
	std::string SYSTEM = "room_";
	std::string FILETYPE = ".txt";

	// Construct plant

	uint32_t n = 4, m = 1, p = 1; 
	// double W = 0.001, V = 0.01;
	double W = 0, V = 0;
	lbcrypto::Matrix<complex<double>> A = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, n);
	lbcrypto::Matrix<complex<double>> B = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, m);
	lbcrypto::Matrix<complex<double>> C = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, n);	
	readMatrix(A, n, DATAFOLDER + SYSTEM + "A" + FILETYPE);	
	readMatrix(B, n, DATAFOLDER + SYSTEM + "B" + FILETYPE);	
	readMatrix(C, p, DATAFOLDER + SYSTEM + "C" + FILETYPE);	

	lbcrypto::Matrix<complex<double>> x0 = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, 1);
	readVector(x0, DATAFOLDER + SYSTEM + "x0" + FILETYPE, 0);	

	Plant<complex<double>>* plant = new Plant<complex<double>>();
	*plant = Plant<complex<double>>(A, B, C, x0, W, V);

	// Precollect values
	uint32_t Tini = 4;
	uint32_t Tfin = 10;
	uint32_t T = 40;
	plant->M = Tini; plant->N = Tfin;

	uint32_t pu = m*Tini;
	uint32_t fu = m*Tfin;
	uint32_t py = p*Tini;
	uint32_t fy = p*Tfin;	

	lbcrypto::Matrix<complex<double>> ud = lbcrypto::Matrix<complex<double>>(zeroAlloc, m, T);
	lbcrypto::Matrix<complex<double>> yd = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, T);
	readMatrix(yd, p, DATAFOLDER + SYSTEM + "yd" + FILETYPE);	
	readMatrix(ud, m, DATAFOLDER + SYSTEM + "ud" + FILETYPE);	

	plant->precollect(ud, pu, fu, 1);
	plant->precollect(yd, py, fy, 0);


	lbcrypto::Matrix<complex<double>> uini = lbcrypto::Matrix<complex<double>>(zeroAlloc, m*Tini, 1);
	lbcrypto::Matrix<complex<double>> yini = lbcrypto::Matrix<complex<double>>(zeroAlloc, p*Tini, 1);
	readVector(yini, DATAFOLDER + SYSTEM + "yini" + FILETYPE, 0);	
	readVector(uini, DATAFOLDER + SYSTEM + "uini" + FILETYPE, 0);


	// Costs
	lbcrypto::Matrix<complex<double>> Q = lbcrypto::Matrix<complex<double>>(zeroAlloc, fy, fy);
	readMatrix(Q, fy, DATAFOLDER + SYSTEM + "Q" + FILETYPE);
	lbcrypto::Matrix<complex<double>> R = lbcrypto::Matrix<complex<double>>(zeroAlloc, fu, fu);
	readMatrix(R, fu, DATAFOLDER + SYSTEM + "R" + FILETYPE);

	std::vector<double> lam;
	readVector(lam, DATAFOLDER + SYSTEM + "lambda" + FILETYPE);
	double lamg = lam[0];	
	double lamy = lam[1];
	double lamu = lam[2];
	
	plant->setCosts(Q, R, lamg, lamy, lamu);

	// Set point
	lbcrypto::Matrix<complex<double>> r = lbcrypto::Matrix<complex<double>>(zeroAlloc, p*Tfin, 1);
	readVector(r, DATAFOLDER + SYSTEM + "ry" + FILETYPE, 0);

	// plant->constLQR(); // the inversion is too slow so read the inverse from a file
	lbcrypto::Matrix<complex<double>> K = lbcrypto::Matrix<complex<double>>(zeroAlloc, plant->S, plant->S);
	readMatrix(K, fu, DATAFOLDER + SYSTEM + "K" + FILETYPE);

	plant->constLQR(K);

	plant->setr(r);
	plant->setyini(yini);
	plant->setuini(uini);

	// This is if you want no delay between HU, HY and uini, yini
	lbcrypto::Matrix<complex<double>> temp_u = plant->getHU().ExtractCol(plant->S-1);
	lbcrypto::Matrix<complex<double>> temp_y = plant->getHY().ExtractCol(plant->S-1);

	// Update temp_u, temp_y with uini and yini
	for (size_t j = 0; j < fu; j ++)
		temp_u(j,0) = temp_u(j+pu,0);
	for (size_t j = fu; j < pu+fu; j ++)
		temp_u(j,0) = uini(j-fu,0);

	for (size_t j = 0; j < fy; j ++)
		temp_y(j,0) = temp_y(j+py,0);
	for (size_t j = fy; j < py+fy; j ++)
		temp_y(j,0) = yini(j-fy,0);

	/* 
	 * Uncomment to see the plaintext behavior
	 */

	// cout.precision(10);

	// uint32_t N = Tfin; 

	// for(int i = 0; i < N; i ++)
	// {

	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->updatex(u);

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);	

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);		

	// 	// plant->constLQR(K);		

	// 	// save the measurements to concatenate the trajectories
	// 		for (size_t j = 0; j < pu+fu-m; j ++)
	// 			temp_u(j,0) = temp_u(j+m,0);
	// 		for (size_t j = pu+fu-m; j < pu+fu; j ++)
	// 			temp_u(j,0) = plant->getu()(j-pu-fu+m,0);

	// 		for (size_t j = 0; j < py+fy-p; j ++)
	// 			temp_y(j,0) = temp_y(j+p,0);
	// 		for (size_t j = py+fy-p; j < py+fy; j ++)
	// 			temp_y(j,0) = plant->gety()(j-py-fy+p,0);		

	// 	// plant->printYU();


	// plant->printYU();	

	// // concatenate trajectories and recompute inverse
	// plant->concatLQR(temp_y, temp_u);
	// plant->onlineLQR();
	// // plant->hU = temp_u;
	// // plant->hY = temp_y;

	// N = 15;

	// for(int i = 0; i < N; i ++)
	// {

	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->onlineUpdatex(u);	

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);	
	
	// 	plant->onlineLQR();

	// }

	// plant->printYU();


	// uint32_t Nnon = 10;
	// for(int i = 0; i < Nnon; i ++)
	// {

	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->updatex(u);

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);	

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);		

	// }	
	// plant->printYU();

	return plant;
}

Plant<complex<double>>* plantInitStableSys()
{
	auto zeroAlloc = [=]() { return 0; };

	std::string SYSTEM = "stable_system_";
	std::string FILETYPE = ".txt";

	// Construct plant

	uint32_t n = 4, m = 2, p = 4; 
	// double W = 0.001, V = 0.01;
	double W = 0, V = 0;
	lbcrypto::Matrix<complex<double>> A = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, n);
	lbcrypto::Matrix<complex<double>> B = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, m);
	lbcrypto::Matrix<complex<double>> C = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, n);	
	readMatrix(A, n, DATAFOLDER + SYSTEM + "A" + FILETYPE);	
	readMatrix(B, n, DATAFOLDER + SYSTEM + "B" + FILETYPE);	
	readMatrix(C, p, DATAFOLDER + SYSTEM + "C" + FILETYPE);	

	lbcrypto::Matrix<complex<double>> x0 = lbcrypto::Matrix<complex<double>>(zeroAlloc, n, 1);
	readVector(x0, DATAFOLDER + SYSTEM + "x0" + FILETYPE, 0);	

	Plant<complex<double>>* plant = new Plant<complex<double>>();
	*plant = Plant<complex<double>>(A, B, C, x0, W, V);

	// Precollect values
	uint32_t Tini = 2;
	uint32_t Tfin = 4;
	uint32_t T = 20;
	plant->M = Tini; plant->N = Tfin;

	uint32_t pu = m*Tini;
	uint32_t fu = m*Tfin;
	uint32_t py = p*Tini;
	uint32_t fy = p*Tfin;	

	lbcrypto::Matrix<complex<double>> ud = lbcrypto::Matrix<complex<double>>(zeroAlloc, m, T);
	lbcrypto::Matrix<complex<double>> yd = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, T);
	readMatrix(yd, p, DATAFOLDER + SYSTEM + "yd" + FILETYPE);	
	readMatrix(ud, m, DATAFOLDER + SYSTEM + "ud" + FILETYPE);	

	plant->precollect(ud, pu, fu, 1);
	plant->precollect(yd, py, fy, 0);


	lbcrypto::Matrix<complex<double>> uini = lbcrypto::Matrix<complex<double>>(zeroAlloc, m*Tini, 1);
	lbcrypto::Matrix<complex<double>> yini = lbcrypto::Matrix<complex<double>>(zeroAlloc, p*Tini, 1);
	readVector(yini, DATAFOLDER + SYSTEM + "yini" + FILETYPE, 0);	
	readVector(uini, DATAFOLDER + SYSTEM + "uini" + FILETYPE, 0);


	// Costs
	lbcrypto::Matrix<complex<double>> Q = lbcrypto::Matrix<complex<double>>(zeroAlloc, fy, fy);
	readMatrix(Q, fy, DATAFOLDER + SYSTEM + "Q" + FILETYPE);
	lbcrypto::Matrix<complex<double>> R = lbcrypto::Matrix<complex<double>>(zeroAlloc, fu, fu);
	readMatrix(R, fu, DATAFOLDER + SYSTEM + "R" + FILETYPE);

	std::vector<double> lam;
	readVector(lam, DATAFOLDER + SYSTEM + "lambda" + FILETYPE);
	double lamg = lam[0];	
	double lamy = lam[1];
	double lamu = lam[2];
	
	plant->setCosts(Q, R, lamg, lamy, lamu);

	// Set point
	lbcrypto::Matrix<complex<double>> r = lbcrypto::Matrix<complex<double>>(zeroAlloc, p*Tfin, 1);
	readVector(r, DATAFOLDER + SYSTEM + "ry" + FILETYPE, 0);

	// plant->constLQR(); // the inversion is too slow so read the inverse from a file
	lbcrypto::Matrix<complex<double>> K = lbcrypto::Matrix<complex<double>>(zeroAlloc, plant->S, plant->S);
	readMatrix(K, fu, DATAFOLDER + SYSTEM + "K" + FILETYPE);

	plant->constLQR(K);

	plant->setr(r);
	plant->setyini(yini);
	plant->setuini(uini);

	// This is if you want no delay between HU, HY and uini, yini
	lbcrypto::Matrix<complex<double>> temp_u = plant->getHU().ExtractCol(plant->S-1);
	lbcrypto::Matrix<complex<double>> temp_y = plant->getHY().ExtractCol(plant->S-1);

	// Update temp_u, temp_y with uini and yini
	for (size_t j = 0; j < fu; j ++)
		temp_u(j,0) = temp_u(j+pu,0);
	for (size_t j = fu; j < pu+fu; j ++)
		temp_u(j,0) = uini(j-fu,0);

	for (size_t j = 0; j < fy; j ++)
		temp_y(j,0) = temp_y(j+py,0);
	for (size_t j = fy; j < py+fy; j ++)
		temp_y(j,0) = yini(j-fy,0);

	/* 
	 * Uncomment to see the plaintext behavior
	 */

	// cout.precision(10);

	// uint32_t N = Tfin; 	

	// for(int i = 0; i < N; i ++)
	// {

	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->updatex(u);	

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);	

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);		

	// 	// save the measurements to concatenate the trajectories
	// 		for (size_t j = 0; j < pu+fu-m; j ++)
	// 			temp_u(j,0) = temp_u(j+m,0);
	// 		for (size_t j = pu+fu-m; j < pu+fu; j ++)
	// 			temp_u(j,0) = plant->getu()(j-pu-fu+m,0);

	// 		for (size_t j = 0; j < py+fy-p; j ++)
	// 			temp_y(j,0) = temp_y(j+p,0);
	// 		for (size_t j = py+fy-p; j < py+fy; j ++)
	// 			temp_y(j,0) = plant->gety()(j-py-fy+p,0);			
	// }

	// plant->printYU();	

	// // concatenate trajectories and recompute inverse
	// plant->concatLQR(temp_y, temp_u);
	// plant->onlineLQR();


	// N = 5;

	// for(int i = 0; i < N; i ++)
	// {

	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->onlineUpdatex(u);	

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);	
	

	// 	plant->onlineLQR();

	// }

	// plant->printYU();


	// uint32_t Nnon = 10;
	// for(int i = 0; i < Nnon; i ++)
	// {
	// 	lbcrypto::Matrix<complex<double>> u = plant->updateu(r, uini, yini);
	// 	plant->updatex(u);

	// 	for (size_t j = 0; j < pu-m; j ++)
	// 		uini(j,0) = uini(j+m,0);
	// 	for (size_t j = pu-m; j < pu; j ++)
	// 		uini(j,0) = plant->getu()(j-pu+m,0);	

	// 	for (size_t j = 0; j < py-p; j ++)
	// 		yini(j,0) = yini(j+p,0);
	// 	for (size_t j = py-p; j < py; j ++)
	// 		yini(j,0) = plant->gety()(j-py+p,0);		

	// }	
	// plant->printYU();

	return plant;
}

#endif