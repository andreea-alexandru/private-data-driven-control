/*
 * Code for model-free control with behavioral paradigm. 
 *
 * The exact functionality is described in the paper "Towards Private Data-driven Cloud-based Control", 
 * by Andreea B. Alexandru, Anastasios Tsiamis and George J. Pappas.
 *
 * At every time step, we use both the offline precollected input-output trajectory, as well as the 
 * input-output trajectory collected at the previous time steps to determine the online feedback.
 * The offline preprocessing (including a matrix inversion) is considered to be done separately; online, 
 * matrix-vector encrypted multiplications and summations, along with the inverse of rank-one updated
 * matrix are performed.
 *
 */


// Define PROFILE to enable TIC-TOC timing measurements
#define PROFILE

#include "palisade.h"
#include "Plant.h"
#include "helperControl.h"

using namespace lbcrypto;

// Specify IDENTITY_FOR_TYPE(T) in order to be able to construct an identity matrix
IDENTITY_FOR_TYPE(complex<double>)

// Make sure to define the correct path of the data folder
const std::string DATAFOLDER = "../plantData/";


Plant<complex<double>>* plantInitStableSys();

Plant<complex<double>>* plantInitRoom();

void OnlineFeedbackStop();
void OnlineFeedbackRefreshStop();
void OfflineFeedback();


int main()
{
	OnlineFeedbackStop(); // use the online measurements to compute the feedback, after Tstop steps, do not collect new samples anymore
	// OnlineFeedbackRefreshStop(); // refresh the matrix at a multiple of Trefresh steps; after Tstop steps, do not collect new samples anymore
	// OfflineFeedback(); // use only the precollected values to compute the feedback

	// plantInitStableSys();
	// plantInitRoom();

	return 0;
}


void OnlineFeedbackStop()
{
	TimeVar t0,t,t1,t2,t3;
	TIC(t0);
	TIC(t);
	double timeInit(0.0), timeEval(0.0), timeStep(0.0);
	double timeClientUpdate(0.0), timeClientDec(0.0), timeClientEnc(0.0);
	double timeServer(0.0), timeServerUpdate(0.0);

	/*
	 * Simulation parameters
	 */
	uint32_t T = 10;

	uint32_t Tstop = 5;	

	/* 
	 * Initialize the plant
	 */
	// Plant<complex<double>>* plant = plantInitStableSys(); // M = 1, N = 3, T = 10
	Plant<complex<double>>* plant = plantInitRoom(); // M = 4, N = 10, T = 40

	// Scale M_1 up by a factor = scale (after t = 0)
	std::complex<double> scale = 1000;	

//////////////////////// Get inputs: r, yini, uini ////////////////////////
	std::vector<complex<double>> r(plant->getr().GetRows()); // The setpoint
	mat2Vec(plant->getr(), r);
	std::vector<complex<double>> yini(plant->getyini().GetRows()); // This is the initial yini; from now on, we compute it encrypted
	mat2Vec(plant->getyini(), yini);
	std::vector<complex<double>> uini(plant->getuini().GetRows()); // This is the initial uini; from now on, we compute it encrypted
	mat2Vec(plant->getuini(), uini);
//////////////////////// Got inputs: r, yini, uini ////////////////////////	

//////////////////////// These are necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 
	// Which diagonals to extract depend on the relationship between the 
	// # of rows and the # of columns of the matrices

	Matrix<complex<double>> Kur = plant->getKur(); // if m < p it will be wide, otherwise tall
	Matrix<complex<double>> Kyini = plant->getKyini(); // if mN > pM it will be tall, otherwise wide
	Matrix<complex<double>> Kuini = plant->getKuini(); // in general it is tall

	size_t colsKur = Kur.GetCols(); size_t rowsKur = Kur.GetRows();
	size_t colsKyini = Kyini.GetCols(); size_t rowsKyini = Kyini.GetRows(); 
	size_t colsKuini = Kuini.GetCols(); size_t rowsKuini = Kuini.GetRows();

	// Baby step giant step choices
	int32_t BsGs_r = 0;
	int32_t dim1_r, dim2_r;
	if (rowsKur >= colsKur) // tall matrix
	{
		dim1_r = std::ceil(std::sqrt(colsKur));
		dim2_r = std::ceil((double)colsKur/dim1_r);		
	}
	else // wide matrix
	{
		if ( rowsKur % colsKur == 0 ) // wide Ef
		{
			dim1_r = std::ceil(std::sqrt(rowsKur)); 
			dim2_r = std::ceil((double)rowsKur/dim1_r);	
		}
		else //plain wide
		{
			dim1_r = std::ceil(std::sqrt(colsKur));
			dim2_r = std::ceil((double)colsKur/dim1_r);		
		}
	}
	int32_t BsGs_y = 0;
	int32_t dim1_y, dim2_y;
	if (rowsKyini >= colsKyini) // tall matrix
	{
		dim1_y = std::ceil(std::sqrt(colsKyini));
		dim2_y = std::ceil((double)colsKyini/dim1_y);		
	}
	else // wide matrix
	{
		if ( rowsKyini % colsKyini == 0 ) // wide Ef
		{
			dim1_y = std::ceil(std::sqrt(rowsKyini)); 
			dim2_y = std::ceil((double)rowsKyini/dim1_y);	
		}
		else // plain wide
		{
			dim1_y = std::ceil(std::sqrt(colsKyini));
			dim2_y = std::ceil((double)colsKyini/dim1_y);		
		}
	}

	int32_t BsGs_u = 0;
	int32_t dim1_u = std::ceil(std::sqrt(colsKuini));
	int32_t dim2_u = std::ceil((double)colsKuini/dim1_u);		

	std::vector<std::vector<complex<double>>> dKur;
	if (rowsKur >= colsKur) // tall
	{
		dKur.resize(colsKur);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKur; i++)
		 	dKur[i] = std::vector<complex<double>>(rowsKur);		

		mat2HybridDiags(Kur, dKur);	
		if (BsGs_r == 1)
		{
	# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_r; j ++)
				for (int32_t i = 0; i < dim1_r; i ++)
					if (dim1_r*j + i < colsKur)
					{
						Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, rowsKur));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKur % colsKur == 0) // wideEf
	 	{
			dKur.resize(rowsKur);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);

			mat2HybridDiags(Kur, dKur);	
			if (BsGs_r == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < rowsKur)
						{
							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKur.resize(colsKur);
			for (size_t i = 0; i < colsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);		 	

			mat2Diags(Kur, dKur);

			if (BsGs_r == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < colsKur)
						{
							std::vector<complex<double>> temp1 = dKur[dim1_r*j+i];
							std::vector<complex<double>> temp2 = dKur[dim1_r*j+i];
							for (int32_t k = 1; k < colsKur/rowsKur; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKur%rowsKur,back_inserter(temp1));	
							dKur[dim1_r*j+i] = temp1;

 							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKyini;
	if (rowsKyini >= colsKyini) // tall
	{
		dKyini.resize(colsKyini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKyini; i++)
		 	dKyini[i] = std::vector<complex<double>>(rowsKyini);

		mat2HybridDiags(Kyini, dKyini);	
		if (BsGs_y == 1)
		{
	# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_y; j ++)
				for (int32_t i = 0; i < dim1_y; i ++)
					if (dim1_y*j + i < colsKyini)
					{
						Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, rowsKyini));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKyini % colsKyini == 0) // wideEf
	 	{
			dKyini.resize(rowsKyini);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);

			mat2HybridDiags(Kyini, dKyini);	
			if (BsGs_y == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < rowsKyini)
						{
							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKyini.resize(colsKyini);
			for (size_t i = 0; i < colsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);		 	

			mat2Diags(Kyini, dKyini);

			if (BsGs_y == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < colsKyini)
						{
							std::vector<complex<double>> temp1 = dKyini[dim1_y*j+i];
							std::vector<complex<double>> temp2 = dKyini[dim1_y*j+i];
							for (int32_t k = 1; k < colsKyini/rowsKyini; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKyini%rowsKyini,back_inserter(temp1));	
							dKyini[dim1_y*j+i] = temp1;

 							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKuini(colsKuini);
	#pragma omp parallel for	
	for (size_t i = 0; i < colsKuini; i++)
	 	dKuini[i] = std::vector<complex<double>>(rowsKuini);

	mat2HybridDiags(Kuini, dKuini);	
	if (BsGs_u == 1)
	{
	# pragma omp parallel for 
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < colsKuini)
				{
					Rotate(dKuini[dim1_u*j+i],ReduceRotation(-dim1_u*j, rowsKuini));
				}
	}	
//////////////////////// These were necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Get inputs: M_1, HY, HU ////////////////////////
	Matrix<complex<double>> M_1 = plant->getM_1(); // The initial inverse matrix = Yf'*Q*Yf + Uf'*R*Uf + lamy*Yp'*Yp + lamu*Up'*Up + lamg*I
	Matrix<complex<double>> HY = plant->getHY(); // The matrix of Hankel matrix for output measurements used for past and future prediction
	Matrix<complex<double>> HU = plant->getHU(); // The matrix of Hankel matrix for input measurements used for past and future prediction

	size_t S = M_1.GetRows();
	size_t Ty = HY.GetRows();
	size_t Tu = HU.GetRows(); 

	// Transform HY and HU into column representation
	std::vector<std::vector<complex<double>>> cHY(S);
	std::vector<std::vector<complex<double>>> cHU(S);		
	for (size_t i = 0; i < S; i ++ )
	{
		cHY[i] = std::vector<complex<double>>(Ty);
		cHU[i] = std::vector<complex<double>>(Tu);
	}

	mat2Cols(HY, cHY); 
	mat2Cols(HU, cHU); 

//////////////////////// Got inputs: M_1, HY, HU ////////////////////////

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////
	Costs<complex<double>> costs = plant->getCosts();
	std::vector<complex<double>> lamQ(plant->py, costs.lamy);
	lamQ.insert(std::end(lamQ), std::begin(costs.diagQ), std::end(costs.diagQ));
	std::vector<complex<double>> lamR(plant->pu, costs.lamu);
	lamR.insert(std::end(lamR), std::begin(costs.diagR), std::end(costs.diagR));	

	std::vector<complex<double>> lamQ_sc = lamQ;
	std::vector<complex<double>> lamR_sc = lamR;
	for (size_t i = 0; i < lamQ_sc.size(); i ++)
		lamQ_sc[i] /= scale;
	for (size_t i = 0; i < lamR_sc.size(); i ++)
		lamR_sc[i] /= scale;	

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////		


	// Step 1: Setup CryptoContext

	// A. Specify main parameters
	/* A1) Multiplicative depth:
	 * The CKKS scheme we setup here will work for any computation
	 * that has a multiplicative depth equal to 'multDepth'.
	 * This is the maximum possible depth of a given multiplication,
	 * but not the total number of multiplications supported by the
	 * scheme.
	 *
	 * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
	 * a multiplicative depth of 1, but requires a total of 3 multiplications.
	 * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
	 * either as a computation of multiplicative depth 3 as
	 * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
	 * as g(x_i) = (x1*x2)*(x3*x4).
	 *
	 * For performance reasons, it's generally preferable to perform operations
	 * in the shortest multiplicative depth possible.
	 */

	/* For the online model-free control, we need a multDepth = 2t + 6 to compute
	 * the control action at time t. In this case, we assume that the client 
	 * transmits uini and yini at each time step.
	 */

	uint32_t multDepth = 2*(Tstop-2) + 7;

	cout << " # of time steps = " << T << ", stop collecting new samples after " << Tstop << \
	", total circuit depth = " << multDepth << endl << endl;


	/* A2) Bit-length of scaling factor.
	 * CKKS works for real numbers, but these numbers are encoded as integers.
	 * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
	 * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
	 * integer). Say the result of a computation based on m' is 130, then at
	 * decryption, the scaling factor is removed so the user is presented with
	 * the real number result of 0.13.
	 *
	 * Parameter 'scaleFactorBits' determines the bit-length of the scaling
	 * factor D, but not the scaling factor itself. The latter is implementation
	 * specific, and it may also vary between ciphertexts in certain versions of
	 * CKKS (e.g., in EXACTRESCALE).
	 *
	 * Choosing 'scaleFactorBits' depends on the desired accuracy of the
	 * computation, as well as the remaining parameters like multDepth or security
	 * standard. This is because the remaining parameters determine how much noise
	 * will be incurred during the computation (remember CKKS is an approximate
	 * scheme that incurs small amounts of noise with every operation). The scaling
	 * factor should be large enough to both accommodate this noise and support results
	 * that match the desired accuracy.
	 */
	uint32_t scaleFactorBits = 50;

	/* A3) Number of plaintext slots used in the ciphertext.
	 * CKKS packs multiple plaintext values in each ciphertext.
	 * The maximum number of slots depends on a security parameter called ring
	 * dimension. In this instance, we don't specify the ring dimension directly,
	 * but let the library choose it for us, based on the security level we choose,
	 * the multiplicative depth we want to support, and the scaling factor size.
	 *
	 * Please use method GetRingDimension() to find out the exact ring dimension
	 * being used for these parameters. Give ring dimension N, the maximum batch
	 * size is N/2, because of the way CKKS works.
	 */

	// In the one-shot model-free control, we need the batch size to be the 
	// whole N/2 because we need to pack vectors repeatedly, without trailing zeros.
	uint32_t batchSize = max(Tu,Ty); // what to display and for EvalSum.

	// Has to take into consideration not only security, but dimensions of matrices 
	// and how much trailing zeros are neeeded
	uint32_t slots = 1024; 

	/* A4) Desired security level based on FHE standards.
	 * This parameter can take four values. Three of the possible values correspond
	 * to 128-bit, 192-bit, and 256-bit security, and the fourth value corresponds
	 * to "NotSet", which means that the user is responsible for choosing security
	 * parameters. Naturally, "NotSet" should be used only in non-production
	 * environments, or by experts who understand the security implications of their
	 * choices.
	 *
	 * If a given security level is selected, the library will consult the current
	 * security parameter tables defined by the FHE standards consortium
	 * (https://homomorphicencryption.org/introduction/) to automatically
	 * select the security parameters. Please see "TABLES of RECOMMENDED PARAMETERS"
	 * in  the following reference for more details:
	 * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
	 */


	// SecurityLevel securityLevel = HEStd_128_classic;
	SecurityLevel securityLevel = HEStd_NotSet;

	RescalingTechnique rsTech = APPROXRESCALE; 
	KeySwitchTechnique ksTech = HYBRID;	

	uint32_t dnum = 0;
	uint32_t maxDepth = 3;
	// This is the size of the first modulus
	uint32_t firstModSize = 60;	
	uint32_t relinWin = 10;
	MODE mode = OPTIMIZED; // Using ternary distribution

	/* 
	 * The following call creates a CKKS crypto context based on the arguments defined above.
	 */
	CryptoContext<DCRTPoly> cc =
			CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			   multDepth,
			   scaleFactorBits,
			   batchSize,
			   securityLevel,
			   slots*4, // set this to zero when security level = HEStd_128_classic
			   rsTech,
			   ksTech,
			   dnum,
			   maxDepth,
			   firstModSize,
			   relinWin,
			   mode);

	uint32_t RD = cc->GetRingDimension();
	cout << "CKKS scheme is using ring dimension " << RD << endl;
	uint32_t cyclOrder = RD*2;
	cout << "CKKS scheme is using the cyclotomic order " << cyclOrder << endl << endl;


	// Enable the features that you wish to use
	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
	cc->Enable(LEVELEDSHE);

	// B. Step 2: Key Generation
	/* B1) Generate encryption keys.
	 * These are used for encryption/decryption, as well as in generating different
	 * kinds of keys.
	 */
	auto keys = cc->KeyGen();

	/* B2) Generate the relinearization key
	 * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
	 * we get a result with some components that are valid under key s, and
	 * with an additional component that's valid under key s^2.
	 *
	 * In most cases, we want to perform relinearization of the multiplicaiton result,
	 * i.e., we want to transform the s^2 component of the ciphertext so it becomes valid
	 * under original key s. To do so, we need to create what we call a relinearization
	 * key with the following line.
	 */
	cc->EvalMultKeyGen(keys.secretKey);

	/* B3) Generate the rotation keys
	 * CKKS supports rotating the contents of a packed ciphertext, but to do so, we
	 * need to create what we call a rotation key. This is done with the following call,
	 * which takes as input a vector with indices that correspond to the rotation offset
	 * we want to support. Negative indices correspond to right shift and positive to left
	 * shift. Look at the output of this demo for an illustration of this.
	 *
	 * Keep in mind that rotations work on the entire ring dimension, not the specified
	 * batch size. This means that, if ring dimension is 8 and batch size is 4, then an
	 * input (1,2,3,4,0,0,0,0) rotated by 2 will become (3,4,0,0,0,0,1,2) and not
	 * (3,4,1,2,0,0,0,0). Also, as someone can observe in the output of this demo, since
	 * CKKS is approximate, zeros are not exact - they're just very small numbers.
	 */

	/* 
	 * Find rotation indices
	 */
//////////////////////// These are necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 		
	size_t maxNoRot = max(max(r.size(),yini.size()),uini.size());
	std::vector<int> indexVec(maxNoRot-1);
	std::iota (std::begin(indexVec), std::end(indexVec), 1);
	if (BsGs_r == 1)
	{
		if (dim1_r > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_r; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_r;
		}
		for (int32_t j = 0; j < dim2_r; j ++)
			for (int32_t i = 0; i < dim1_r; i ++)
				if (dim1_r*j + i < dKur.size())
				{
					if (rowsKur >= colsKur) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_r*j, rowsKur)); 
						indexVec.push_back(ReduceRotation(dim1_r*j, rowsKur));	
					}
					else // wide ef
						if (rowsKur % colsKur == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));	
						}
				}
	}
	if (BsGs_y == 1)
	{
		if (dim1_y > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_y; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_y;
		}
		for (int32_t j = 0; j < dim2_y; j ++)
			for (int32_t i = 0; i < dim1_y; i ++)
				if (dim1_y*j + i < dKyini.size())
				{
					if (rowsKyini >= colsKyini) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_y*j, rowsKyini)); 
						indexVec.push_back(ReduceRotation(dim1_y*j, rowsKyini));	
					}
					else // wide
						if (rowsKyini % colsKyini == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));							
						}
				}
	}
	if (BsGs_u == 1)
	{
		if (dim1_u > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_u; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_u;
		}
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < dKuini.size())
				{
					indexVec.push_back(ReduceRotation(-dim1_u*j, rowsKuini));
					indexVec.push_back(ReduceRotation(dim1_u*j, rowsKuini));
				}
	}	
//////////////////////// These were necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 
	for (size_t i = 0; i < plant->fu; i ++) // rotations to compute the elements of Uf
		indexVec.push_back(plant->pu + i);	
	for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
		indexVec.push_back(-i);
//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 

//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 	
	indexVec.push_back(plant->p); indexVec.push_back(plant->m);
	indexVec.push_back(-Ty+plant->p); indexVec.push_back(-Tu+plant->m);
	indexVec.push_back(-plant->py+plant->p); indexVec.push_back(-plant->pu+plant->m);	
//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 		

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr, keys.secretKey, indexVec);				

	/* 
	 * B4) Generate keys for summing up the packed values in a ciphertext, needed for inner product
	 */
	cc->EvalSumKeyGen(keys.secretKey);

	// Step 3: Encoding and encryption of inputs

	/* 
	 * Encoding as plaintexts
	 */

	// Vectors r, yini and uini need to be repeated in the packed plaintext for the first time step and with zero encryptions for the following time steps
	std::vector<std::complex<double>> rep_r(Fill(r,slots));
	Plaintext ptxt_rep_r = cc->MakeCKKSPackedPlaintext(rep_r);
	std::vector<std::complex<double>> zero_r(Ty); 
	std::copy(r.begin(),r.end(),zero_r.begin()+plant->py);
	Plaintext ptxt_r = cc->MakeCKKSPackedPlaintext(zero_r);

	std::vector<std::complex<double>> rep_yini(Fill(yini,slots));	
	Plaintext ptxt_rep_yini = cc->MakeCKKSPackedPlaintext(rep_yini);
	Plaintext ptxt_yini = cc->MakeCKKSPackedPlaintext(yini);

	std::vector<std::complex<double>> rep_uini(Fill(uini,slots));	
	Plaintext ptxt_rep_uini = cc->MakeCKKSPackedPlaintext(rep_uini);
	Plaintext ptxt_uini = cc->MakeCKKSPackedPlaintext(uini);

	Plaintext ptxt_u, ptxt_y;

//////////////////////// These are necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 
	std::vector<Plaintext> ptxt_dKur(dKur.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
	{
		if (BsGs_r == 1) 
		{
			std::vector<std::complex<double>> rep_dKur(Fill(dKur[i],slots)); 
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(rep_dKur);
		}
		else
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(dKur[i]);
	}

	std::vector<Plaintext> ptxt_dKuini(dKuini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
	{
		if (BsGs_u == 1) 
		{
			std::vector<std::complex<double>> rep_dKuini(Fill(dKuini[i],slots)); // if we want to use baby step giant step
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(rep_dKuini);
		}
		else
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(dKuini[i]);
	}

	std::vector<Plaintext> ptxt_dKyini(dKyini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
	{
		if (BsGs_y == 1) 
		{
			std::vector<std::complex<double>> rep_dKyini(Fill(dKyini[i],slots)); // if we want to use baby step giant step
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(rep_dKyini);
		}
		else		
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(dKyini[i]);
	}
//////////////////////// These were necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 

	std::vector<Plaintext> ptxt_cHY(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ptxt_cHY[i] = cc->MakeCKKSPackedPlaintext(cHY[i]);

	std::vector<Plaintext> ptxt_cHU(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ptxt_cHU[i] = cc->MakeCKKSPackedPlaintext(cHU[i]);	

	std::vector<std::vector<Plaintext>> ptxt_M_1(S);
	for (size_t i = 0; i < S; ++i) // symmetric matrix
	{
		ptxt_M_1[i] = std::vector<Plaintext>(S-i); 
		for (size_t j = 0; j < S-i; ++j)
			ptxt_M_1[i][j] = cc->MakeCKKSPackedPlaintext({M_1(i,j+i)*scale});	
	}

	Plaintext ptxt_lamQ = cc->MakeCKKSPackedPlaintext(lamQ);
	Plaintext ptxt_lamR = cc->MakeCKKSPackedPlaintext(lamR);
	Plaintext ptxt_lamg = cc->MakeCKKSPackedPlaintext({costs.lamg});

	Plaintext ptxt_1 = cc->MakeCKKSPackedPlaintext({1});
	std::vector<complex<double>> complex_1 = {1};

	Plaintext ptxt_sclamQ = cc->MakeCKKSPackedPlaintext(lamQ_sc);
	Plaintext ptxt_sclamR = cc->MakeCKKSPackedPlaintext(lamR_sc);

	Plaintext ptxt_scale = cc->MakeCKKSPackedPlaintext({scale});
	
	/* 
	 * Encrypt the encoded vectors 
	 */

//////////////////////// The values for the first iteration //////////////////////// 
	auto ctxt_rep_r = cc->Encrypt(keys.publicKey, ptxt_rep_r);
	auto ctxt_rep_yini = cc->Encrypt(keys.publicKey, ptxt_rep_yini);
	auto ctxt_rep_uini = cc->Encrypt(keys.publicKey, ptxt_rep_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i){
		ctxt_dKur[i] = cc->Encrypt(keys.publicKey, ptxt_dKur[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i){
		ctxt_dKyini[i] = cc->Encrypt(keys.publicKey, ptxt_dKyini[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i){
		ctxt_dKuini[i] = cc->Encrypt(keys.publicKey, ptxt_dKuini[i]);	
	}
//////////////////////// The values for the first iteration //////////////////////// 			

	auto ctxt_r = cc->Encrypt(keys.publicKey, ptxt_r);
	auto ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);
	auto ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHY(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ctxt_cHY[i] = cc->Encrypt(keys.publicKey, ptxt_cHY[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHU(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ctxt_cHU[i] = cc->Encrypt(keys.publicKey, ptxt_cHU[i]);

	std::vector<std::vector<Ciphertext<DCRTPoly>>> ctxt_M_1(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i) // symmetric matrix
	{
		ctxt_M_1[i] = std::vector<Ciphertext<DCRTPoly>>(S-i); 
		for (size_t j = 0; j < S-i; ++j)
			ctxt_M_1[i][j] = cc->Encrypt(keys.publicKey, ptxt_M_1[i][j]);				
	}

	std::vector<std::vector<Ciphertext<DCRTPoly>>> ctxt_Uf;
	// Get Uf as elements to use it in the computation of u* without adding an extra masking, 
	// necessary if we work with column representation
	Matrix<complex<double>> Uf = HU.ExtractRows(plant->pu, Tu-1);		
	std::vector<std::vector<Plaintext>> ptxt_Uf(plant->fu);
	for (size_t i = 0; i < plant->fu; ++i) 
	{
		ptxt_Uf[i] = std::vector<Plaintext>(S); 
		for (size_t j = 0; j < S; ++j)
			ptxt_Uf[i][j] = cc->MakeCKKSPackedPlaintext({Uf(i,j)});	
	}		

	ctxt_Uf.resize(plant->fu);
# pragma omp parallel for 
	for (size_t i = 0; i < plant->fu; ++i)
	{
		ctxt_Uf[i] = std::vector<Ciphertext<DCRTPoly>>(S); 
		for (size_t j = 0; j < S; ++j)
			ctxt_Uf[i][j] = cc->Encrypt(keys.publicKey, ptxt_Uf[i][j]);				
	}


	Ciphertext<DCRTPoly> ctxt_1 = cc->Encrypt(keys.publicKey, ptxt_1);	
	Ciphertext<DCRTPoly> ctxt_scale = cc->Encrypt(keys.publicKey, ptxt_scale);		
	

	timeInit = TOC(t);
	cout << "Time for offline key generation, encoding and encryption: " << timeInit << " ms" << endl;

	// Step 4: Evaluation

	TIC(t);

	Ciphertext<DCRTPoly> ctxt_y, ctxt_u;
	Ciphertext<DCRTPoly> ctxt_mSchur, ctxt_mSchur_1, ctxt_scaled_mSchur_1;
	std::vector<Ciphertext<DCRTPoly>> ctxt_mVec(S), ctxt_mVec_s(S); 
	std::vector<Ciphertext<DCRTPoly>> ctxt_M_1mVec(S), ctxt_M_1mVec_s(S);

	// Start online computations
	for (size_t t = 0; t < T; t ++)
	{
		cout << "t = " << t << endl << endl;

		TIC(t2);

		TIC(t1);

		if (t == 0) 
		{

			// Matrix-vector multiplication for Kur*r
			Ciphertext<DCRTPoly> result_r;
			if ( rowsKur >= colsKur ) // tall
				result_r = EvalMatVMultTall(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
			else // wide
				if ( rowsKur % colsKur == 0) // wide Ef
				{
					result_r = EvalMatVMultWideEf(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
				}
				else // plain wide
					result_r = EvalMatVMultWide(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);			

			// Matrix-vector multiplication for Kyini*yini; 
			Ciphertext<DCRTPoly> result_y;
			if ( rowsKyini >= colsKyini ) // tall
				result_y = EvalMatVMultTall(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
			else // wide
				if ( rowsKyini % colsKyini == 0) // wide Ef
					result_y = EvalMatVMultWideEf(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
				else // plain wide
					result_y = EvalMatVMultWide(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	

			// Matrix-vector multiplication for Kuini*uini; matVMultTall
			auto result_u = EvalMatVMultTall(ctxt_dKuini, ctxt_rep_uini, *EvalRotKeys, rowsKuini, colsKuini, BsGs_u, dim1_u);		

			// Add the components
			ctxt_u = cc->EvalAdd ( result_u, result_y );
			ctxt_u = cc->EvalAdd ( ctxt_u, result_r );
			
		}

		else // t>0
		{
			if (t < Tstop)
			{
				ctxt_mSchur = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[S] ), ptxt_lamQ ),\
					cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[S] ), ptxt_lamR ) ), max(Ty,Tu) );
				ctxt_mSchur = cc->EvalAdd( ctxt_mSchur, ptxt_lamg );

	# pragma omp parallel for
				for (size_t i = 0; i < S; i ++){
					ctxt_mVec[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_lamR  ) ), max(Ty,Tu) );
				}

	# pragma omp parallel for
				for (size_t i=0; i < S; i++)
				{
					ctxt_mVec[i] = cc->Rescale(ctxt_mVec[i]); ctxt_mVec[i] = cc->Rescale(ctxt_mVec[i]);		
				}

				// scaled copy
	# pragma omp parallel for
				for (size_t i = 0; i < S; i ++)
				{
					ctxt_mVec_s[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_sclamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_sclamR  ) ), max(Ty,Tu) );
				}

	# pragma omp parallel for
				for (size_t i=0; i < S; i++)
				{
					ctxt_mVec_s[i] = cc->Rescale(ctxt_mVec_s[i]); ctxt_mVec_s[i] = cc->Rescale(ctxt_mVec_s[i]);						
				}

				ctxt_mSchur = cc->Rescale(ctxt_mSchur); ctxt_mSchur = cc->Rescale(ctxt_mSchur);	

				Ciphertext<DCRTPoly> ctxt_tempSum = ctxt_mSchur;
				for (size_t i = 0; i < S; ++i) 
				{
					for (size_t j = 1; j < S-i; ++j)
					{
						if (i == 0 && j == 1)
							ctxt_tempSum = cc->EvalMult( ctxt_M_1[0][1], cc->Rescale(cc->EvalMult( ctxt_mVec[0], ctxt_mVec_s[1] )) ) ;
						else
							ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i], ctxt_mVec_s[j+i] )) ) );
					}
				}		
				ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, ctxt_tempSum); // to account for the fact that M_1 is symmetric and we performed only half of the operations

				for (size_t i = 0; i < S; ++i) 
					ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, cc->EvalMult( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_mVec[i], ctxt_mVec_s[i] ) )) );				

				ctxt_mSchur = cc->EvalSub( ctxt_mSchur, cc->Rescale(ctxt_tempSum) );				

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec[i] = cc->EvalMult( ctxt_mVec[i], ctxt_M_1[i][0] );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
					{
						ctxt_M_1mVec[i] = cc->EvalAdd( ctxt_M_1mVec[i], cc->EvalMult( ctxt_mVec[i+j], ctxt_M_1[i][j] ) );
						ctxt_M_1mVec[i+j] = cc->EvalAdd( ctxt_M_1mVec[i+j], cc->EvalMult( ctxt_mVec[i], ctxt_M_1[i][j] ) );	
					}	


	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec[i] = cc->Rescale(ctxt_M_1mVec[i]);	

				// scaled copy
	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec_s[i] = cc->EvalMult( ctxt_mVec_s[i], ctxt_M_1[i][0] );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
					{
						ctxt_M_1mVec_s[i] = cc->EvalAdd( ctxt_M_1mVec_s[i], cc->EvalMult( ctxt_mVec_s[i+j], ctxt_M_1[i][j] ) );
						ctxt_M_1mVec_s[i+j] = cc->EvalAdd( ctxt_M_1mVec_s[i+j], cc->EvalMult( ctxt_mVec_s[i], ctxt_M_1[i][j] ) );	
					}	

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec_s[i] = cc->Rescale(ctxt_M_1mVec_s[i]);	

				// Resize ctxt_M_1
				ctxt_M_1.resize(S+1);
	# pragma omp parallel for 
				for (size_t i = 0; i < S+1; ++i)
					ctxt_M_1[i].resize(S+1-i);
			
				ctxt_M_1[S][0] = ctxt_scale; // encryption of scale of the last element on the diagonal	

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i) // add -M_1mVecT as the last column of M_1
					ctxt_M_1[i][S-i] = -ctxt_M_1mVec[i]; 
					

				// Copy matrix M_1 to recompute it after the clients provides 1/mSchur 
				auto ctxt_M_1_copy = ctxt_M_1;

				for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
					ctxt_M_1_copy[i][0] = cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_M_1_copy[i][0], ctxt_mSchur )), cc->Rescale(cc->EvalMult( ctxt_M_1mVec[i], ctxt_M_1mVec_s[i] )) );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
						ctxt_M_1_copy[i][j] = cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_M_1_copy[i][j], ctxt_mSchur )), cc->Rescale(cc->EvalMult( ctxt_M_1mVec[i], ctxt_M_1mVec_s[i+j] )) ) ;

				// bring to the same number of levels
				size_t M_1_levels = ctxt_M_1_copy[0][0]->GetLevel();

	# pragma omp parallel for 
				for (size_t i = 0; i <= S; ++i) // add -M_1mVecT as the last column of M_1
					ctxt_M_1_copy[i][S-i] = cc->LevelReduce(ctxt_M_1_copy[i][S-i], nullptr, M_1_levels - ctxt_M_1_copy[i][S-i]->GetLevel()); 				


				// Compute u*, with Uf represented as elements

				// Add the last column of Uf (last part of the last column of HU) as elements in ctxt_Uf
				// If we want to send u* as one ciphertext back to the client, we need to make sure the elements of u* are followed by zeros to not require masking at the result.
				// This means that e.g., Z should have zeros trailing

				auto HUSPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_cHU[S] );			

				for (size_t i = 0; i < plant->fu; i ++)
				{
					ctxt_Uf[i].resize(S+1);			
					ctxt_Uf[i][S] = cc->GetEncryptionAlgorithm()->EvalFastRotation( ctxt_cHU[S], plant->pu + i, cyclOrder, HUSPrecomp, *EvalRotKeys ); 		
				}				

				timeServer = TOC(t1);
				cout << "Time for computations without uini and yini at the server at step " << t << ": " << timeServer << " ms" << endl;	

				TIC(t1);					

				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S+1);
	# pragma omp parallel for
				for (size_t i = 0; i < S + 1; i ++)
				{ 
					ctxt_Z[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR  ) ), max(Tu,Ty) );

					ctxt_Z[i] = cc->EvalMult( ctxt_Z[i], ptxt_1 ); 
					// rescale to get it to the needed depth
					ctxt_Z[i] = cc->Rescale(ctxt_Z[i]); ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	
				}									
				

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1_copy[0][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][0], ctxt_Z[0] )) );
					for (size_t i = 1; i < S+1; ++i) 
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1_copy[i][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i] ) )) );

					for (size_t i = 0; i < S+1; ++i) 
					{
						for (size_t j = 1; j < S+1-i; ++j)
						{
							ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1_copy[i][j], \
								cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i+j] )), cc->Rescale(cc->EvalMult( ctxt_Uf[k][i+j], ctxt_Z[i] ) )) ) );
						}
					}	
				}							

				ctxt_u = ctxt_uel[0];		
				for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
					ctxt_u = cc->EvalAdd( ctxt_u, cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i, *EvalRotKeys));

			}
			else // t >= Tstop
			{
				TIC(t1);					

				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S);
	# pragma omp parallel for
				for (size_t i = 0; i < S ; i ++)
				{ 
					ctxt_Z[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR  ) ), max(Tu,Ty) );

					ctxt_Z[i] = cc->EvalMult( ctxt_Z[i], ptxt_1 ); 
					// rescale to get it to the needed depth
					ctxt_Z[i] = cc->Rescale(ctxt_Z[i]); ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	
				}						
		

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1[0][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][0], ctxt_Z[0] )) );
					for (size_t i = 1; i < S; ++i) 			
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i] ) )) );

					for (size_t i = 0; i < S; ++i) 
					{
						for (size_t j = 1; j < S-i; ++j)
						{			
							ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1[i][j], \
								cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i+j] )), cc->Rescale(cc->EvalMult( ctxt_Uf[k][i+j], ctxt_Z[i] ) )) ) );
						}
					}	
				}							

				ctxt_u = ctxt_uel[0];		
				for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_u = cc->EvalAdd( ctxt_u, cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i, *EvalRotKeys));
				}				
			}

		}

		cout << "\n# levels of ctxt_u at time " << t << ": " << ctxt_u->GetLevel() << ", depth: " << ctxt_u->GetDepth() <<\
		", # towers: " << ctxt_u->GetElements()[0].GetParams()->GetParams().size() << endl << endl;		

		timeServer = TOC(t1);
		cout << "Time for computing the control action at the server at step " << t << ": " << timeServer << " ms" << endl;	

		TIC(t1);
		Plaintext ptxt_Schur;
		complex<double> mSchur_1;

		if ( t > 0 && t < Tstop)
		{
			cc->Decrypt(keys.secretKey, ctxt_mSchur, &ptxt_Schur);
			ptxt_Schur->SetLength(1);
			mSchur_1 = double(1)/(ptxt_Schur->GetCKKSPackedValue()[0]);

		}

		Plaintext result_u_t;
		cout.precision(8);
		cc->Decrypt(keys.secretKey, ctxt_u, &result_u_t);
		if ( (t == 0) || (t > 0) )
			result_u_t->SetLength(plant->m);
		else
			result_u_t->SetLength(Tu);

		auto u = result_u_t->GetCKKSPackedValue(); // Make sure to make the imaginary parts to be zero s.t. error does not accumulate

		if (t > 0 && t < Tstop)
			for (size_t i = 0; i < plant->m; i ++)
					u[i] *= mSchur_1/scale;			

		if (t >= Tstop)
			for (size_t i = 0; i < plant->m; i ++)
				u[i] /= scale;		

		for (size_t i = 0; i < plant->m; i ++)
			u[i].imag(0);

		timeClientDec = TOC(t1);
		cout << "Time for decrypting the control action at the client at step " << t << ": " << timeClientDec << " ms" << endl;	

		TIC(t1);

		// Update plant
		plant->onlineUpdatex(u);
		plant->onlineLQR();
		if (plant->M == 1)
		{
			uini = u;
			mat2Vec(plant->gety(),yini);
		}
		else
		{
			Rotate(uini, plant->m);
			std::copy(u.begin(),u.begin()+plant->m,uini.begin()+plant->pu-plant->m);
			Rotate(yini, plant->p);
			std::vector<complex<double>> y(plant->p);
			mat2Vec(plant->gety(),y);
			std::copy(y.begin(),y.begin()+plant->p,yini.begin()+plant->py-plant->p);			
		}

		// plant->printYU(); // if you want to print inputs and outputs at every time step

		plant->setyini(yini);
		plant->setuini(uini);

		timeClientUpdate = TOC(t1);
		cout << "Time for updating the plant at step " << t << ": " << timeClientUpdate << " ms" << endl;		

		if (t < T-1) // we don't need to compute anything else after this
		{
			TIC(t1);

			// Re-encrypt variables 
			// Make sure to cut the number of towers

			if (t > 0 && t < Tstop)
			{	
				ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()),1,0);
				ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
				ctxt_y = cc->LevelReduce(ctxt_y, nullptr, 2*t);	
				ptxt_u = cc->MakeCKKSPackedPlaintext(u,1,0);
				ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);	
				ctxt_u = cc->LevelReduce(ctxt_u, nullptr, 2*t);	
				ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1},1,0));
				ctxt_mSchur_1 = cc->LevelReduce(ctxt_mSchur_1,nullptr,2*t);			
				ctxt_scaled_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({scale*mSchur_1},1,0));
				ctxt_scaled_mSchur_1 = cc->LevelReduce(ctxt_scaled_mSchur_1,nullptr,2*t);					
			}
			else
			{
				if (t < Tstop)
				{
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()));
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y);
					ptxt_u = cc->MakeCKKSPackedPlaintext(u);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);			
					ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1}));
					ctxt_scaled_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1*scale}));
				}
				else // t >= Tstop
				{
					int32_t dropLevels = ctxt_y->GetLevel(); // previous ctxt_y
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()));
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y);
					ptxt_u = cc->MakeCKKSPackedPlaintext(u);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);				
					ctxt_y = cc->LevelReduce(ctxt_y, nullptr, dropLevels);	
					ctxt_u = cc->LevelReduce(ctxt_u, nullptr, dropLevels);								
				}				

			}		

			timeClientEnc = TOC(t1);
			cout << "Time for encoding and encrypting at the client at time " << t+1 << ": " << timeClientEnc << " ms" << endl;	

			Plaintext ptxt_interm;
			cout.precision(8);

			////////////// Back to the server.
			TIC(t1);

			if (t > 0)
			{
				if (t < Tstop)
				{
					ctxt_M_1[S][0] = ctxt_scaled_mSchur_1; // encryption of 1/mSchur of the last element on the diagonal	
					
					// We compute M_1 with fewer levels by multiplying 1/mSchur to mVecT first for t > 1.		
		 # pragma omp parallel for 
					for (size_t i = 0; i < S; ++i) // add -m_Schur*M_1*mVecT as the last column of M_1
						ctxt_M_1[i][S-i] = cc->EvalMult( ctxt_M_1[i][0], cc->Rescale( cc->EvalMult( ctxt_mVec[i], -ctxt_mSchur_1 ) ) ); 									

					for (size_t i = 0; i < S; ++i)
						for (size_t j = 1; j < S-i; ++j)
						{	
							ctxt_M_1[i][S-i] = cc->EvalAdd( ctxt_M_1[i][S-i], cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i+j], -ctxt_mSchur_1 )) ));
							ctxt_M_1[i+j][S-i-j] = cc->EvalAdd( ctxt_M_1[i+j][S-i-j], cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i], -ctxt_mSchur_1 )) ));					
						}

	# pragma omp parallel for 
					for (size_t i = 0; i < S; ++i)
						ctxt_M_1[i][S-i] = cc->Rescale(ctxt_M_1[i][S-i]);	

					for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
						ctxt_M_1[i][0] = cc->EvalSub( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_M_1[i][S-i], ctxt_M_1mVec_s[i] )) );


					for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
						for (size_t j = 1; j < S-i; ++j)
							ctxt_M_1[i][j] = cc->EvalSub( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_M_1[i][S-i], ctxt_M_1mVec_s[i+j] ))  );

					S += 1;
				}
			}

			if (t < Tstop)
			{
				ctxt_cHY.resize(S+1); ctxt_cHU.resize(S+1);	

				if ( t > 0 )
				{
					ctxt_cHY[S-1] = cc->LevelReduce(ctxt_cHY[S-1], nullptr, 2);
					ctxt_cHU[S-1] = cc->LevelReduce(ctxt_cHU[S-1], nullptr, 2);
					ctxt_yini = cc->LevelReduce(ctxt_yini, nullptr, 2);
					ctxt_uini = cc->LevelReduce(ctxt_uini, nullptr, 2);
					ctxt_r = cc->LevelReduce(ctxt_r, nullptr, 2);
					CompressEvalKeys(*EvalRotKeys, 2);
				}	


				ctxt_cHY[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHY[S-1], plant->p, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, -Ty+plant->p, *EvalRotKeys) );
				ctxt_cHU[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHU[S-1], plant->m, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -Tu+plant->m, *EvalRotKeys) );	

				ctxt_mVec.resize(S+1); ctxt_M_1mVec.resize(S+1);
				ctxt_mVec_s.resize(S+1); ctxt_M_1mVec_s.resize(S+1);						
			}

			if (plant->M == 1)
			{
				ctxt_yini = ctxt_y;
				ctxt_uini = ctxt_u;
			}
			else 
			{
				ctxt_yini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_yini, plant->p, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, -plant->py+plant->p, *EvalRotKeys) );
				ctxt_uini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -plant->pu+plant->m, *EvalRotKeys) );
			}
		

			timeServerUpdate = TOC(t1);
			cout << "Time for updating the M_1, Hankel matrices, uini and yini at the server at time " << t+1 << ": " << timeServerUpdate << " ms" << endl;	

			timeStep = TOC(t2);		
			cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	

			cout << "S = " << S << endl;
		}
		else
		{
			timeStep = TOC(t2);		
			cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	
		}					


	}	

	timeEval = TOC(t);
	cout << "Total time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	timeEval = TOC(t0);
	cout << "Total offline+online time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	plant->printYU(); // print all inputs and outputs at the end of the simulation

}


void OnlineFeedbackRefreshStop()
{
	TimeVar t0,t,t1,t2,t3;
	TIC(t0);
	TIC(t);
	double timeInit(0.0), timeEval(0.0), timeStep(0.0);
	double timeClientUpdate(0.0), timeClientDec(0.0), timeClientEnc(0.0);
	double timeServer(0.0), timeServerUpdate(0.0), timeServerRefresh(0.0);

	/*
	 * Simulation parameters
	 */
	uint32_t T = 10;

	uint32_t Trefresh = 3; // Make sure Trefresh <= Tstop - 2.

	uint32_t Tstop = 5;

	/* 
	 * Initialize the plant
	 */
	// Plant<complex<double>>* plant = plantInitStableSys(); // M = 1, N = 3, T = 10
	Plant<complex<double>>* plant = plantInitRoom(); // M = 4, N = 10, T = 40

	// Scale M_1 up by a factor of scale (after t == 0)
	std::complex<double> scale = 1000;	

//////////////////////// Get inputs: r, yini, uini ////////////////////////
	std::vector<complex<double>> r(plant->getr().GetRows()); // The setpoint
	mat2Vec(plant->getr(), r);
	std::vector<complex<double>> yini(plant->getyini().GetRows()); // This is the initial yini; from now on, we compute it encrypted
	mat2Vec(plant->getyini(), yini);
	std::vector<complex<double>> uini(plant->getuini().GetRows()); // This is the initial uini; from now on, we compute it encrypted
	mat2Vec(plant->getuini(), uini);
//////////////////////// Got inputs: r, yini, uini ////////////////////////	

//////////////////////// These are necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 
	// Which diagonals to extract depend on the relationship between the 
	// # of rows and the # of columns of the matrices

	Matrix<complex<double>> Kur = plant->getKur(); // if m < p it will be wide, otherwise tall
	Matrix<complex<double>> Kyini = plant->getKyini(); // if mN > pM it will be tall, otherwise wide
	Matrix<complex<double>> Kuini = plant->getKuini(); // in general it is tall

	size_t colsKur = Kur.GetCols(); size_t rowsKur = Kur.GetRows();
	size_t colsKyini = Kyini.GetCols(); size_t rowsKyini = Kyini.GetRows(); 
	size_t colsKuini = Kuini.GetCols(); size_t rowsKuini = Kuini.GetRows();

	// Baby step giant step choices
	int32_t BsGs_r = 0;
	int32_t dim1_r, dim2_r;
	if (rowsKur >= colsKur) // tall matrix
	{
		dim1_r = std::ceil(std::sqrt(colsKur));
		dim2_r = std::ceil((double)colsKur/dim1_r);		
	}
	else // wide matrix
	{
		if ( rowsKur % colsKur == 0 ) // wide Ef
		{
			dim1_r = std::ceil(std::sqrt(rowsKur)); 
			dim2_r = std::ceil((double)rowsKur/dim1_r);	
		}
		else //plain wide
		{
			dim1_r = std::ceil(std::sqrt(colsKur));
			dim2_r = std::ceil((double)colsKur/dim1_r);		
		}
	}
	int32_t BsGs_y = 0;
	int32_t dim1_y, dim2_y;
	if (rowsKyini >= colsKyini) // tall matrix
	{
		dim1_y = std::ceil(std::sqrt(colsKyini));
		dim2_y = std::ceil((double)colsKyini/dim1_y);		
	}
	else // wide matrix
	{
		if ( rowsKyini % colsKyini == 0 ) // wide Ef
		{
			dim1_y = std::ceil(std::sqrt(rowsKyini)); 
			dim2_y = std::ceil((double)rowsKyini/dim1_y);	
		}
		else // plain wide
		{
			dim1_y = std::ceil(std::sqrt(colsKyini));
			dim2_y = std::ceil((double)colsKyini/dim1_y);		
		}
	}

	int32_t BsGs_u = 0;
	int32_t dim1_u = std::ceil(std::sqrt(colsKuini));
	int32_t dim2_u = std::ceil((double)colsKuini/dim1_u);		

	std::vector<std::vector<complex<double>>> dKur;
	if (rowsKur >= colsKur) // tall
	{
		dKur.resize(colsKur);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKur; i++)
		 	dKur[i] = std::vector<complex<double>>(rowsKur);

		mat2HybridDiags(Kur, dKur);	
		if (BsGs_r == 1)
		{
	# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_r; j ++)
				for (int32_t i = 0; i < dim1_r; i ++)
					if (dim1_r*j + i < colsKur)
					{
						Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, rowsKur));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKur % colsKur == 0) // wideEf
	 	{
			dKur.resize(rowsKur);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);

			mat2HybridDiags(Kur, dKur);	
			if (BsGs_r == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < rowsKur)
						{
							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKur.resize(colsKur);
			for (size_t i = 0; i < colsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);		 	

			mat2Diags(Kur, dKur);

			if (BsGs_r == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < colsKur)
						{
							std::vector<complex<double>> temp1 = dKur[dim1_r*j+i];
							std::vector<complex<double>> temp2 = dKur[dim1_r*j+i];
							for (int32_t k = 1; k < colsKur/rowsKur; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKur%rowsKur,back_inserter(temp1));	
							dKur[dim1_r*j+i] = temp1;

 							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKyini;
	if (rowsKyini >= colsKyini) // tall
	{
		dKyini.resize(colsKyini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKyini; i++)
		 	dKyini[i] = std::vector<complex<double>>(rowsKyini);

		mat2HybridDiags(Kyini, dKyini);	
		if (BsGs_y == 1)
		{
	# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_y; j ++)
				for (int32_t i = 0; i < dim1_y; i ++)
					if (dim1_y*j + i < colsKyini)
					{
						Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, rowsKyini));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKyini % colsKyini == 0) // wideEf
	 	{
			dKyini.resize(rowsKyini);
	#pragma omp parallel for	
			for (size_t i = 0; i < rowsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);

			mat2HybridDiags(Kyini, dKyini);	
			if (BsGs_y == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < rowsKyini)
						{
							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKyini.resize(colsKyini);
			for (size_t i = 0; i < colsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);		 	

			mat2Diags(Kyini, dKyini);

			if (BsGs_y == 1)
			{
	# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < colsKyini)
						{
							std::vector<complex<double>> temp1 = dKyini[dim1_y*j+i];
							std::vector<complex<double>> temp2 = dKyini[dim1_y*j+i];
							for (int32_t k = 1; k < colsKyini/rowsKyini; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKyini%rowsKyini,back_inserter(temp1));	
							dKyini[dim1_y*j+i] = temp1;

 							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKuini(colsKuini);
	#pragma omp parallel for	
	for (size_t i = 0; i < colsKuini; i++)
	 	dKuini[i] = std::vector<complex<double>>(rowsKuini);

	mat2HybridDiags(Kuini, dKuini);	
	if (BsGs_u == 1)
	{
	# pragma omp parallel for 
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < colsKuini)
				{
					Rotate(dKuini[dim1_u*j+i],ReduceRotation(-dim1_u*j, rowsKuini));
				}
	}	
//////////////////////// These were necessary only for the beginning of the algorithm: Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Get inputs: M_1, HY, HU ////////////////////////
	Matrix<complex<double>> M_1 = plant->getM_1(); // The initial inverse matrix = Yf'*Q*Yf + Uf'*R*Uf + lamy*Yp'*Yp + lamu*Up'*Up + lamg*I
	Matrix<complex<double>> HY = plant->getHY(); // The matrix of Hankel matrix for output measurements used for past and future prediction
	Matrix<complex<double>> HU = plant->getHU(); // The matrix of Hankel matrix for input measurements used for past and future prediction

	size_t S = M_1.GetRows();
	size_t Ty = HY.GetRows();
	size_t Tu = HU.GetRows(); 

	// Transform HY and HU into column representation
	std::vector<std::vector<complex<double>>> cHY(S);
	std::vector<std::vector<complex<double>>> cHU(S);		
	for (size_t i = 0; i < S; i ++ )
	{
		cHY[i] = std::vector<complex<double>>(Ty);
		cHU[i] = std::vector<complex<double>>(Tu);
	}

	mat2Cols(HY, cHY); 
	mat2Cols(HU, cHU); 

//////////////////////// Got inputs: M_1, HY, HU ////////////////////////

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////
	Costs<complex<double>> costs = plant->getCosts();
	std::vector<complex<double>> lamQ(plant->py, costs.lamy);
	lamQ.insert(std::end(lamQ), std::begin(costs.diagQ), std::end(costs.diagQ));
	std::vector<complex<double>> lamR(plant->pu, costs.lamu);
	lamR.insert(std::end(lamR), std::begin(costs.diagR), std::end(costs.diagR));	

	std::vector<complex<double>> lamQ_sc = lamQ;
	std::vector<complex<double>> lamR_sc = lamR;
	for (size_t i = 0; i < lamQ_sc.size(); i ++)
		lamQ_sc[i] /= scale;
	for (size_t i = 0; i < lamR_sc.size(); i ++)
		lamR_sc[i] /= scale;	

//////////////////////// Get costs: Q, R, lamg, lamy, lamu ////////////////////////		


	// Step 1: Setup CryptoContext

	// A. Specify main parameters
	/* A1) Multiplicative depth:
	 * The CKKS scheme we setup here will work for any computation
	 * that has a multiplicative depth equal to 'multDepth'.
	 * This is the maximum possible depth of a given multiplication,
	 * but not the total number of multiplications supported by the
	 * scheme.
	 *
	 * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
	 * a multiplicative depth of 1, but requires a total of 3 multiplications.
	 * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
	 * either as a computation of multiplicative depth 3 as
	 * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
	 * as g(x_i) = (x1*x2)*(x3*x4).
	 *
	 * For performance reasons, it's generally preferable to perform operations
	 * in the shortest multiplicative depth possible.
	 */

	/* For the online model-free control, we need a multDepth = 2t + 6 to compute
	 * the control action at time t. In this case, we assume that the client 
	 * transmits uini and yini at each time step.
	 */

	uint32_t multDepth = 2*(Trefresh-1) + 7;

	cout << " # of time steps = " << T << ", refresh at time = " << Trefresh << "*k, " <<\
	"stop collecting at time = " << Tstop <<", total circuit depth = " << multDepth << endl << endl;

	/* A2) Bit-length of scaling factor.
	 * CKKS works for real numbers, but these numbers are encoded as integers.
	 * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
	 * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
	 * integer). Say the result of a computation based on m' is 130, then at
	 * decryption, the scaling factor is removed so the user is presented with
	 * the real number result of 0.13.
	 *
	 * Parameter 'scaleFactorBits' determines the bit-length of the scaling
	 * factor D, but not the scaling factor itself. The latter is implementation
	 * specific, and it may also vary between ciphertexts in certain versions of
	 * CKKS (e.g., in EXACTRESCALE).
	 *
	 * Choosing 'scaleFactorBits' depends on the desired accuracy of the
	 * computation, as well as the remaining parameters like multDepth or security
	 * standard. This is because the remaining parameters determine how much noise
	 * will be incurred during the computation (remember CKKS is an approximate
	 * scheme that incurs small amounts of noise with every operation). The scaling
	 * factor should be large enough to both accommodate this noise and support results
	 * that match the desired accuracy.
	 */
	uint32_t scaleFactorBits = 50;

	/* A3) Number of plaintext slots used in the ciphertext.
	 * CKKS packs multiple plaintext values in each ciphertext.
	 * The maximum number of slots depends on a security parameter called ring
	 * dimension. In this instance, we don't specify the ring dimension directly,
	 * but let the library choose it for us, based on the security level we choose,
	 * the multiplicative depth we want to support, and the scaling factor size.
	 *
	 * Please use method GetRingDimension() to find out the exact ring dimension
	 * being used for these parameters. Give ring dimension N, the maximum batch
	 * size is N/2, because of the way CKKS works.
	 */
	// In the one-shot model-free control, we need the batch size to be the 
	// whole N/2 because we need to pack vectors repeatedly, without trailing zeros.
	uint32_t batchSize = max(Tu,Ty); // what to display and for EvalSum.

	// Has to take into consideration not only security, but dimensions of matrices and how much trailing zeros are neeeded
	uint32_t slots = 1024; 

	/* A4) Desired security level based on FHE standards.
	 * This parameter can take four values. Three of the possible values correspond
	 * to 128-bit, 192-bit, and 256-bit security, and the fourth value corresponds
	 * to "NotSet", which means that the user is responsible for choosing security
	 * parameters. Naturally, "NotSet" should be used only in non-production
	 * environments, or by experts who understand the security implications of their
	 * choices.
	 *
	 * If a given security level is selected, the library will consult the current
	 * security parameter tables defined by the FHE standards consortium
	 * (https://homomorphicencryption.org/introduction/) to automatically
	 * select the security parameters. Please see "TABLES of RECOMMENDED PARAMETERS"
	 * in  the following reference for more details:
	 * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
	 */
	// SecurityLevel securityLevel = HEStd_128_classic;
	SecurityLevel securityLevel = HEStd_NotSet;

	RescalingTechnique rsTech = APPROXRESCALE; 
	KeySwitchTechnique ksTech = HYBRID;	

	uint32_t dnum = 0;
	uint32_t maxDepth = 3;
	// This is the size of the first modulus
	uint32_t firstModSize = 60;	
	uint32_t relinWin = 10;
	MODE mode = OPTIMIZED; // Using ternary distribution

	/* 
	 * The following call creates a CKKS crypto context based on the arguments defined above.
	 */
	CryptoContext<DCRTPoly> cc =
			CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			   multDepth,
			   scaleFactorBits,
			   batchSize,
			   securityLevel,
			   slots*4, // set this to zero when security level = HEStd_128_classic
			   rsTech,
			   ksTech,
			   dnum,
			   maxDepth,
			   firstModSize,
			   relinWin,
			   mode);

	uint32_t RD = cc->GetRingDimension();
	cout << "CKKS scheme is using ring dimension " << RD << endl;
	uint32_t cyclOrder = RD*2;
	cout << "CKKS scheme is using the cyclotomic order " << cyclOrder << endl << endl;


	// Enable the features that you wish to use
	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
	cc->Enable(LEVELEDSHE);

	// B. Step 2: Key Generation
	/* B1) Generate encryption keys.
	 * These are used for encryption/decryption, as well as in generating different
	 * kinds of keys.
	 */
	auto keys = cc->KeyGen();

	/* B2) Generate the relinearization key
	 * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
	 * we get a result with some components that are valid under key s, and
	 * with an additional component that's valid under key s^2.
	 *
	 * In most cases, we want to perform relinearization of the multiplicaiton result,
	 * i.e., we want to transform the s^2 component of the ciphertext so it becomes valid
	 * under original key s. To do so, we need to create what we call a relinearization
	 * key with the following line.
	 */
	cc->EvalMultKeyGen(keys.secretKey);

	/* B3) Generate the rotation keys
	 * CKKS supports rotating the contents of a packed ciphertext, but to do so, we
	 * need to create what we call a rotation key. This is done with the following call,
	 * which takes as input a vector with indices that correspond to the rotation offset
	 * we want to support. Negative indices correspond to right shift and positive to left
	 * shift. Look at the output of this demo for an illustration of this.
	 *
	 * Keep in mind that rotations work on the entire ring dimension, not the specified
	 * batch size. This means that, if ring dimension is 8 and batch size is 4, then an
	 * input (1,2,3,4,0,0,0,0) rotated by 2 will become (3,4,0,0,0,0,1,2) and not
	 * (3,4,1,2,0,0,0,0). Also, as someone can observe in the output of this demo, since
	 * CKKS is approximate, zeros are not exact - they're just very small numbers.
	 */

	/* 
	 * Find rotation indices
	 */
//////////////////////// These are necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 		
	size_t maxNoRot = max(max(r.size(),yini.size()),uini.size());
	std::vector<int> indexVec(maxNoRot-1);
	std::iota (std::begin(indexVec), std::end(indexVec), 1);
	if (BsGs_r == 1)
	{
		if (dim1_r > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_r; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_r;
		}
		for (int32_t j = 0; j < dim2_r; j ++)
			for (int32_t i = 0; i < dim1_r; i ++)
				if (dim1_r*j + i < dKur.size())
				{
					if (rowsKur >= colsKur) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_r*j, rowsKur)); 
						indexVec.push_back(ReduceRotation(dim1_r*j, rowsKur));	
					}
					else // wide ef
						if (rowsKur % colsKur == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));	
						}
				}
	}
	if (BsGs_y == 1)
	{
		if (dim1_y > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_y; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_y;
		}
		for (int32_t j = 0; j < dim2_y; j ++)
			for (int32_t i = 0; i < dim1_y; i ++)
				if (dim1_y*j + i < dKyini.size())
				{
					if (rowsKyini >= colsKyini) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_y*j, rowsKyini)); 
						indexVec.push_back(ReduceRotation(dim1_y*j, rowsKyini));	
					}
					else // wide
						if (rowsKyini % colsKyini == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));							
						}
				}
	}
	if (BsGs_u == 1)
	{
		if (dim1_u > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_u; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_u;
		}
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < dKuini.size())
				{
					indexVec.push_back(ReduceRotation(-dim1_u*j, rowsKuini));
					indexVec.push_back(ReduceRotation(dim1_u*j, rowsKuini));
				}
	}	
//////////////////////// These were necessary only for the beginning of the algorithm: rotations for Kur, Kyini, Kuini //////////////////////// 	

//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 
	for (size_t i = 0; i < plant->fu; i ++) // rotations to compute the elements of Uf
		indexVec.push_back(plant->pu + i);	
	for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
		indexVec.push_back(-i);
//////////////////////// Rotations for u = U*M_1*Z //////////////////////// 

//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 	
	indexVec.push_back(plant->p); indexVec.push_back(plant->m);
	indexVec.push_back(-Ty+plant->p); indexVec.push_back(-Tu+plant->m);
	indexVec.push_back(-plant->py+plant->p); indexVec.push_back(-plant->pu+plant->m);	
//////////////////////// Rotations for constructing the new yini and uini and the last columns of HY and HU //////////////////////// 		

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	// change this to drop levels
	// cc->EvalAtIndexKeyGen(keys.secretKey, indexVec);	
	auto EvalRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr, keys.secretKey, indexVec);	

//////////////////////// Rotations for refreshing M_1 ////////////////////////	
	indexVec.clear();
	for (int32_t k = 1; k <= int((Tstop-2)/Trefresh); k ++)
	{
		for (int32_t i = 0; i < S + k*Trefresh; i ++)
		{
			for (int32_t j = 0; j < S + k*Trefresh - i; j ++)
			{
				indexVec.push_back( -(int)(i * (S + k*Trefresh) - int(i*(i-1)/2) + j ) );
			}
		}
	}

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalPackKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr, keys.secretKey, indexVec);		
	CompressEvalKeys(*EvalPackKeys, (2*(Trefresh-1)+5));

	indexVec.clear();
	for (int32_t k = 1; k <= int((Tstop-2)/Trefresh); k ++)
	{
		for (int32_t i = 0; i < S + k*Trefresh; i ++)
		{
			for (int32_t j = 0; j < S + k*Trefresh - i; j ++)
			{
				indexVec.push_back( i * (S + k*Trefresh) - int(i*(i-1)/2) + j );
			}
		}
	}

	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalUnpackKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);		

//////////////////////// Rotations for refreshing M_1 ////////////////////////					

	/* 
	 * B4) Generate keys for summing up the packed values in a ciphertext, needed for inner product
	 */
	cc->EvalSumKeyGen(keys.secretKey);

	// Step 3: Encoding and encryption of inputs

	/* 
	 * Encoding as plaintexts
	 */

	// Vectors r, yini and uini need to be repeated in the packed plaintext for the first time step and with zero encryptions for the following time steps
	std::vector<std::complex<double>> rep_r(Fill(r,slots));
	Plaintext ptxt_rep_r = cc->MakeCKKSPackedPlaintext(rep_r);
	std::vector<std::complex<double>> zero_r(Ty); 
	std::copy(r.begin(),r.end(),zero_r.begin()+plant->py);
	Plaintext ptxt_r = cc->MakeCKKSPackedPlaintext(zero_r);

	std::vector<std::complex<double>> rep_yini(Fill(yini,slots));	
	Plaintext ptxt_rep_yini = cc->MakeCKKSPackedPlaintext(rep_yini);
	Plaintext ptxt_yini = cc->MakeCKKSPackedPlaintext(yini);

	std::vector<std::complex<double>> rep_uini(Fill(uini,slots));	
	Plaintext ptxt_rep_uini = cc->MakeCKKSPackedPlaintext(rep_uini);
	Plaintext ptxt_uini = cc->MakeCKKSPackedPlaintext(uini);

	Plaintext ptxt_u, ptxt_y;

//////////////////////// These are necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 
	std::vector<Plaintext> ptxt_dKur(dKur.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
	{
		if (BsGs_r == 1) 
		{
			std::vector<std::complex<double>> rep_dKur(Fill(dKur[i],slots)); 
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(rep_dKur);
		}
		else
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(dKur[i]);
	}

	std::vector<Plaintext> ptxt_dKuini(dKuini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
	{
		if (BsGs_u == 1)
		{
			std::vector<std::complex<double>> rep_dKuini(Fill(dKuini[i],slots)); // if we want to use baby step giant step
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(rep_dKuini);
		}
		else
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(dKuini[i]);
	}

	std::vector<Plaintext> ptxt_dKyini(dKyini.size());
	# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
	{
		if (BsGs_y == 1) 
		{
			std::vector<std::complex<double>> rep_dKyini(Fill(dKyini[i],slots)); // if we want to use baby step giant step
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(rep_dKyini);
		}
		else		
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(dKyini[i]);
	}
//////////////////////// These were necessary only for the beginning of the algorithm: encryptions for Kur, Kyini, Kuini //////////////////////// 

	std::vector<Plaintext> ptxt_cHY(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ptxt_cHY[i] = cc->MakeCKKSPackedPlaintext(cHY[i]);

	std::vector<Plaintext> ptxt_cHU(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ptxt_cHU[i] = cc->MakeCKKSPackedPlaintext(cHU[i]);	

	std::vector<std::vector<Plaintext>> ptxt_M_1(S);
	for (size_t i = 0; i < S; ++i) // symmetric matrix
	{
		ptxt_M_1[i] = std::vector<Plaintext>(S-i); 
		for (size_t j = 0; j < S-i; ++j)
			ptxt_M_1[i][j] = cc->MakeCKKSPackedPlaintext({M_1(i,j+i)*scale});	
	}

	Plaintext ptxt_lamQ = cc->MakeCKKSPackedPlaintext(lamQ);
	Plaintext ptxt_lamR = cc->MakeCKKSPackedPlaintext(lamR);
	Plaintext ptxt_lamg = cc->MakeCKKSPackedPlaintext({costs.lamg});

	Plaintext ptxt_1 = cc->MakeCKKSPackedPlaintext({1});
	std::vector<complex<double>> complex_1 = {1};

	Plaintext ptxt_sclamQ = cc->MakeCKKSPackedPlaintext(lamQ_sc);
	Plaintext ptxt_sclamR = cc->MakeCKKSPackedPlaintext(lamR_sc);

	Plaintext ptxt_scale = cc->MakeCKKSPackedPlaintext({scale});
	
	/* 
	 * Encrypt the encoded vectors
	 */

//////////////////////// The values for the first iteration //////////////////////// 
	auto ctxt_rep_r = cc->Encrypt(keys.publicKey, ptxt_rep_r);
	auto ctxt_rep_yini = cc->Encrypt(keys.publicKey, ptxt_rep_yini);
	auto ctxt_rep_uini = cc->Encrypt(keys.publicKey, ptxt_rep_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i){
		ctxt_dKur[i] = cc->Encrypt(keys.publicKey, ptxt_dKur[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i){
		ctxt_dKyini[i] = cc->Encrypt(keys.publicKey, ptxt_dKyini[i]);
	}

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i){
		ctxt_dKuini[i] = cc->Encrypt(keys.publicKey, ptxt_dKuini[i]);	
	}
//////////////////////// The values for the first iteration //////////////////////// 			

	auto ctxt_r = cc->Encrypt(keys.publicKey, ptxt_r);
	auto ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);
	auto ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHY(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ctxt_cHY[i] = cc->Encrypt(keys.publicKey, ptxt_cHY[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_cHU(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i)
		ctxt_cHU[i] = cc->Encrypt(keys.publicKey, ptxt_cHU[i]);

	std::vector<std::vector<Ciphertext<DCRTPoly>>> ctxt_M_1(S);
# pragma omp parallel for 
	for (size_t i = 0; i < S; ++i) // symmetric matrix
	{
		ctxt_M_1[i] = std::vector<Ciphertext<DCRTPoly>>(S-i); 
		for (size_t j = 0; j < S-i; ++j)
			ctxt_M_1[i][j] = cc->Encrypt(keys.publicKey, ptxt_M_1[i][j]);				
	}

	std::vector<std::vector<Ciphertext<DCRTPoly>>> ctxt_Uf;
	// Get Uf as elements to use it in the computation of u* without adding an extra masking, necessary if we work with column representation
	Matrix<complex<double>> Uf = HU.ExtractRows(plant->pu, Tu-1);		
	std::vector<std::vector<Plaintext>> ptxt_Uf(plant->fu);
	for (size_t i = 0; i < plant->fu; ++i) 
	{
		ptxt_Uf[i] = std::vector<Plaintext>(S); 
		for (size_t j = 0; j < S; ++j)
			ptxt_Uf[i][j] = cc->MakeCKKSPackedPlaintext({Uf(i,j)});	
	}		

	ctxt_Uf.resize(plant->fu);
# pragma omp parallel for 
	for (size_t i = 0; i < plant->fu; ++i)
	{
		ctxt_Uf[i] = std::vector<Ciphertext<DCRTPoly>>(S); 
		for (size_t j = 0; j < S; ++j)
			ctxt_Uf[i][j] = cc->Encrypt(keys.publicKey, ptxt_Uf[i][j]);				
	}

	Ciphertext<DCRTPoly> ctxt_1 = cc->Encrypt(keys.publicKey, ptxt_1);	
	Ciphertext<DCRTPoly> ctxt_scale = cc->Encrypt(keys.publicKey, ptxt_scale);		
	
	timeInit = TOC(t);
	cout << "Time for offline key generation, encoding and encryption: " << timeInit << " ms" << endl;

	// Step 4: Evaluation

	TIC(t);

	Ciphertext<DCRTPoly> ctxt_y, ctxt_u;
	Ciphertext<DCRTPoly> ctxt_mSchur, ctxt_mSchur_1, ctxt_scaled_mSchur_1;
	std::vector<Ciphertext<DCRTPoly>> ctxt_mVec(S), ctxt_mVec_s(S); 
	std::vector<Ciphertext<DCRTPoly>> ctxt_M_1mVec(S), ctxt_M_1mVec_s(S);
	Ciphertext<DCRTPoly> ctxt_M_1_packed; // ciphertexts that packs M_1

	// Start online computations
	for (size_t t = 0; t < T; t ++)
	{
		cout << "t = " << t << endl << endl;

		TIC(t2);

		TIC(t1);

		if (t == 0) 
		{

			// Matrix-vector multiplication for Kur*r
			Ciphertext<DCRTPoly> result_r;
			if ( rowsKur >= colsKur ) // tall
				result_r = EvalMatVMultTall(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
			else // wide
				if ( rowsKur % colsKur == 0) // wide Ef
				{
					result_r = EvalMatVMultWideEf(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
				}
				else // plain wide
					result_r = EvalMatVMultWide(ctxt_dKur, ctxt_rep_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	

			// Matrix-vector multiplication for Kyini*yini; 
			Ciphertext<DCRTPoly> result_y;
			if ( rowsKyini >= colsKyini ) // tall
				result_y = EvalMatVMultTall(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
			else // wide
				if ( rowsKyini % colsKyini == 0) // wide Ef
					result_y = EvalMatVMultWideEf(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
				else // plain wide
					result_y = EvalMatVMultWide(ctxt_dKyini, ctxt_rep_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	

			// Matrix-vector multiplication for Kuini*uini; matVMultTall
			auto result_u = EvalMatVMultTall(ctxt_dKuini, ctxt_rep_uini, *EvalRotKeys, rowsKuini, colsKuini, BsGs_u, dim1_u);			

			// Add the components
			ctxt_u = cc->EvalAdd ( result_u, result_y );
			ctxt_u = cc->EvalAdd ( ctxt_u, result_r );
		
		}

		else // t>0
		{
			if (t < Tstop)
			{
				ctxt_mSchur = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[S] ), ptxt_lamQ ),\
					cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[S] ), ptxt_lamR ) ), max(Ty,Tu) );
				ctxt_mSchur = cc->EvalAdd( ctxt_mSchur, ptxt_lamg );

	# pragma omp parallel for
				for (size_t i = 0; i < S; i ++){
					ctxt_mVec[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_lamR  ) ), max(Ty,Tu) );
				}

	# pragma omp parallel for
				for (size_t i=0; i < S; i++)
				{
					ctxt_mVec[i] = cc->Rescale(ctxt_mVec[i]); ctxt_mVec[i] = cc->Rescale(ctxt_mVec[i]);		
				}

				// scaled copy
	# pragma omp parallel for
				for (size_t i = 0; i < S; i ++){
					ctxt_mVec_s[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[S], ctxt_cHY[i] ), ptxt_sclamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[S], ctxt_cHU[i] ), ptxt_sclamR  ) ), max(Ty,Tu) );
				}

	# pragma omp parallel for
				for (size_t i=0; i < S; i++)
				{
					ctxt_mVec_s[i] = cc->Rescale(ctxt_mVec_s[i]); ctxt_mVec_s[i] = cc->Rescale(ctxt_mVec_s[i]);							
				}					

				ctxt_mSchur = cc->Rescale(ctxt_mSchur); ctxt_mSchur = cc->Rescale(ctxt_mSchur);	

				Ciphertext<DCRTPoly> ctxt_tempSum = ctxt_mSchur;
				for (size_t i = 0; i < S; ++i) 
				{
					for (size_t j = 1; j < S-i; ++j)
					{
						if (i == 0 && j == 1)
							ctxt_tempSum = cc->EvalMult( ctxt_M_1[0][1], cc->Rescale(cc->EvalMult( ctxt_mVec[0], ctxt_mVec_s[1] )) ) ;
						else
							ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i], ctxt_mVec_s[j+i] )) ) );
					}
				}		
				ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, ctxt_tempSum); // to account for the fact that M_1 is symmetric and we performed only half of the operations

				for (size_t i = 0; i < S; ++i) 
					ctxt_tempSum = cc->EvalAdd( ctxt_tempSum, cc->EvalMult( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_mVec[i], ctxt_mVec_s[i] ) )) );				

				ctxt_mSchur = cc->EvalSub( ctxt_mSchur, cc->Rescale(ctxt_tempSum) );			

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec[i] = cc->EvalMult( ctxt_mVec[i], ctxt_M_1[i][0] );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
					{
						ctxt_M_1mVec[i] = cc->EvalAdd( ctxt_M_1mVec[i], cc->EvalMult( ctxt_mVec[i+j], ctxt_M_1[i][j] ) );
						ctxt_M_1mVec[i+j] = cc->EvalAdd( ctxt_M_1mVec[i+j], cc->EvalMult( ctxt_mVec[i], ctxt_M_1[i][j] ) );	
					}	

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec[i] = cc->Rescale(ctxt_M_1mVec[i]);	

				// scaled copy
	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec_s[i] = cc->EvalMult( ctxt_mVec_s[i], ctxt_M_1[i][0] );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
					{
						ctxt_M_1mVec_s[i] = cc->EvalAdd( ctxt_M_1mVec_s[i], cc->EvalMult( ctxt_mVec_s[i+j], ctxt_M_1[i][j] ) );
						ctxt_M_1mVec_s[i+j] = cc->EvalAdd( ctxt_M_1mVec_s[i+j], cc->EvalMult( ctxt_mVec_s[i], ctxt_M_1[i][j] ) );	
					}	

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i)
					ctxt_M_1mVec_s[i] = cc->Rescale(ctxt_M_1mVec_s[i]);		

				// Resize ctxt_M_1
				ctxt_M_1.resize(S+1);
	# pragma omp parallel for 
				for (size_t i = 0; i < S+1; ++i)
					ctxt_M_1[i].resize(S+1-i);
			
				ctxt_M_1[S][0] = ctxt_scale; // encryption of scale of the last element on the diagonal	

	# pragma omp parallel for 
				for (size_t i = 0; i < S; ++i) // add -M_1mVecT as the last column of M_1
					ctxt_M_1[i][S-i] = -ctxt_M_1mVec[i]; 
					

				// Copy matrix M_1 to recompute it after the clients provides 1/mSchur 
				auto ctxt_M_1_copy = ctxt_M_1;	

				for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
					ctxt_M_1_copy[i][0] = cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_M_1_copy[i][0], ctxt_mSchur )), cc->Rescale(cc->EvalMult( ctxt_M_1mVec[i], ctxt_M_1mVec_s[i] )) );

				for (size_t i = 0; i < S; ++i)
					for (size_t j = 1; j < S-i; ++j)
						ctxt_M_1_copy[i][j] = cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_M_1_copy[i][j], ctxt_mSchur )), cc->Rescale(cc->EvalMult( ctxt_M_1mVec[i], ctxt_M_1mVec_s[i+j] )) ) ;

				// bring to the same number of levels
				size_t M_1_levels = ctxt_M_1_copy[0][0]->GetLevel();

	# pragma omp parallel for 
				for (size_t i = 0; i <= S; ++i) // add -M_1mVecT as the last column of M_1
					ctxt_M_1_copy[i][S-i] = cc->LevelReduce(ctxt_M_1_copy[i][S-i], nullptr, M_1_levels - ctxt_M_1_copy[i][S-i]->GetLevel()); 				

				// When u_t reaches the maximum allowed multiplicative depth, the server packs M_1 into a vector (if S^2 < RD/2 and into multiple vectors otherwise) 
				// and sends it to the client, that refreshes a single ciphertexts 
				if ( t % Trefresh == 0 && t != Tstop - 1) // in the last case, we don't need to refresh M_1
				{
					TIC(t3);

					for (int32_t i = 0; i < S+1; i ++)
					{
						if (i == 0)
							ctxt_M_1_packed = cc->EvalMult( ctxt_M_1_copy[i][0], ptxt_1 );
						else
							ctxt_M_1_packed = cc->EvalAdd(ctxt_M_1_packed, cc->GetEncryptionAlgorithm()->EvalAtIndex(\
								cc->EvalMult( ctxt_M_1_copy[i][0], ptxt_1 ), -(int)(i*(S+1) - int(i*(i-1)/2)), *EvalPackKeys ) );
						for (int32_t j = 1; j < S+1-i; j ++)
						{
							ctxt_M_1_packed = cc->EvalAdd( ctxt_M_1_packed, cc->GetEncryptionAlgorithm()->EvalAtIndex(\
								cc->EvalMult( ctxt_M_1_copy[i][j], ptxt_1 ), -(int)(i*(S+1) - int(i*(i-1)/2) + j ), *EvalPackKeys) );
						}
										
					}
					// clear keys if this was the last refresh
					if (t == int((Tstop-2)/Trefresh)*Trefresh)
					{
						EvalPackKeys->clear();
					}	

					timeServerRefresh = TOC(t3);	
					cout << "Time for packing inv(M) at the server at step " << t << ": " << timeServerRefresh << " ms" << endl;					

				}				

				// Compute u*, with Uf represented as elements

				// Add the last column of Uf (last part of the last column of HU) as elements in ctxt_Uf
				// If we want to send u* as one ciphertext back to the client, we need to make sure the elements of u* are followed by zeros to not require masking at the result.
				// This means that e.g., Z should have zeros trailing

				auto HUSPrecomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_cHU[S] );			

				for (size_t i = 0; i < plant->fu; i ++)
				{
					ctxt_Uf[i].resize(S+1);			
					ctxt_Uf[i][S] = cc->GetEncryptionAlgorithm()->EvalFastRotation( ctxt_cHU[S], plant->pu + i, cyclOrder, HUSPrecomp, *EvalRotKeys ); 		
				}				

				timeServer = TOC(t1);
				cout << "Time for computations without uini and yini at the server at step " << t << ": " << timeServer << " ms" << endl;	

				TIC(t1);					

				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S+1);
	# pragma omp parallel for
				for (size_t i = 0; i < S + 1; i ++)
				{ 
					ctxt_Z[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR  ) ), max(Tu,Ty) );

					ctxt_Z[i] = cc->EvalMult( ctxt_Z[i], ptxt_1 ); 
					// rescale to get it to the needed depth
					ctxt_Z[i] = cc->Rescale(ctxt_Z[i]); ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	
				}						

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1_copy[0][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][0], ctxt_Z[0] )) );
					for (size_t i = 1; i < S+1; ++i) 		
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1_copy[i][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i] ) )) );

					for (size_t i = 0; i < S+1; ++i) 
						for (size_t j = 1; j < S+1-i; ++j)
						{				
							ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1_copy[i][j], \
								cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i+j] )), cc->Rescale(cc->EvalMult( ctxt_Uf[k][i+j], ctxt_Z[i] ) )) ) );
						}
				}							

				ctxt_u = ctxt_uel[0];		
				for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_u = cc->EvalAdd( ctxt_u, cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i, *EvalRotKeys));
				}

			}
			else // t >= Tstop
			{
				TIC(t1);					

				std::vector<Ciphertext<DCRTPoly>> ctxt_Z(S);
	# pragma omp parallel for
				for (size_t i = 0; i < S ; i ++)
				{ 
					ctxt_Z[i] = cc->EvalSum( cc->EvalAdd (cc->EvalMult( cc->EvalMult( ctxt_cHY[i], cc->EvalAdd( ctxt_yini, ctxt_r ) ), ptxt_lamQ ),\
						cc->EvalMult( cc->EvalMult( ctxt_cHU[i], ctxt_uini ), ptxt_lamR  ) ), max(Tu,Ty) );

					ctxt_Z[i] = cc->EvalMult( ctxt_Z[i], ptxt_1 ); 
					// rescale to get it to the needed depth
					ctxt_Z[i] = cc->Rescale(ctxt_Z[i]); ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	ctxt_Z[i] = cc->Rescale(ctxt_Z[i]);	
				}						
		

				std::vector<Ciphertext<DCRTPoly>> ctxt_uel(plant->m);
				for (size_t k = 0; k < plant->m; k ++)
				{
					ctxt_uel[k] = cc->EvalMult ( ctxt_M_1[0][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][0], ctxt_Z[0] )) );
					for (size_t i = 1; i < S; ++i) 				
						ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i] ) )) );

					for (size_t i = 0; i < S; ++i) 
						for (size_t j = 1; j < S-i; ++j)
						{				
							ctxt_uel[k] = cc->EvalAdd( ctxt_uel[k], cc->EvalMult ( ctxt_M_1[i][j], \
								cc->EvalAdd( cc->Rescale(cc->EvalMult( ctxt_Uf[k][i], ctxt_Z[i+j] )), cc->Rescale(cc->EvalMult( ctxt_Uf[k][i+j], ctxt_Z[i] ) )) ) );
						}
				}							

				ctxt_u = ctxt_uel[0];		
				for(int32_t i = 1; i < plant->m; i ++)	// in this case, we can only send the relevant elements
				{
					ctxt_u = cc->EvalAdd( ctxt_u, cc->GetEncryptionAlgorithm()->EvalAtIndex( ctxt_uel[i], -i, *EvalRotKeys));
				}				
			}

		}

		cout << "\n# levels of ctxt_u at time " << t << ": " << ctxt_u->GetLevel() << ", depth: " << ctxt_u->GetDepth() <<\
		", # towers: " << ctxt_u->GetElements()[0].GetParams()->GetParams().size() << endl << endl;		

		timeServer = TOC(t1);
		cout << "Time for computing the control action at the server at step " << t << ": " << timeServer << " ms" << endl;	

		TIC(t1);
		Plaintext ptxt_Schur;
		complex<double> mSchur_1;

		if ( t > 0 && t < Tstop )
		{
			cc->Decrypt(keys.secretKey, ctxt_mSchur, &ptxt_Schur);
			ptxt_Schur->SetLength(1);
			mSchur_1 = double(1)/(ptxt_Schur->GetCKKSPackedValue()[0]);
		}

		Plaintext result_u_t;
		cout.precision(8);
		cc->Decrypt(keys.secretKey, ctxt_u, &result_u_t);
		if ( (t == 0) || (t > 0) )
			result_u_t->SetLength(plant->m);
		else
			result_u_t->SetLength(Tu);

		auto u = result_u_t->GetCKKSPackedValue(); // Make sure to make the imaginary parts to be zero s.t. error does not accumulate

		if (t > 0 && t < Tstop)
			for (size_t i = 0; i < plant->m; i ++)
					u[i] *= mSchur_1/scale;			

		if (t >= Tstop)
			for (size_t i = 0; i < plant->m; i ++)
					u[i] /= scale;			

		for (size_t i = 0; i < plant->m; i ++)
			u[i].imag(0);


		timeClientDec = TOC(t1);
		cout << "Time for decrypting the control action at the client at step " << t << ": " << timeClientDec << " ms" << endl;	

		TIC(t1);

		// Update plant
		plant->onlineUpdatex(u);
		plant->onlineLQR();
		if (plant->M == 1)
		{
			uini = u;
			mat2Vec(plant->gety(),yini);
		}
		else
		{
			Rotate(uini, plant->m);
			std::copy(u.begin(),u.begin()+plant->m,uini.begin()+plant->pu-plant->m);
			Rotate(yini, plant->p);
			std::vector<complex<double>> y(plant->p);
			mat2Vec(plant->gety(),y);
			std::copy(y.begin(),y.begin()+plant->p,yini.begin()+plant->py-plant->p);			
		}

		// plant->printYU(); // if you want to print inputs and outputs at every time step

		plant->setyini(yini);
		plant->setuini(uini);

		timeClientUpdate = TOC(t1);
		cout << "Time for updating the plant at step " << t << ": " << timeClientUpdate << " ms" << endl;		


		if (t < T-1) // we don't need to compute anything else after this
		{
			TIC(t1);

			// Re-encrypt variables 
			// Make sure to cut the number of towers

			if ((t > 0) && (t > (int)((Tstop-2)/Trefresh)*Trefresh ) && (t < Tstop)) 
			{	
				if ( t%Trefresh != 0)
				{
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()),1,0);
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
					ctxt_y = cc->LevelReduce(ctxt_y, nullptr, 2*(t%Trefresh));	
					ptxt_u = cc->MakeCKKSPackedPlaintext(u,1,0);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);	
					ctxt_u = cc->LevelReduce(ctxt_u, nullptr, 2*(t%Trefresh));	
					ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1},1,0));
					ctxt_mSchur_1 = cc->LevelReduce(ctxt_mSchur_1,nullptr,2*(t%Trefresh));			
					ctxt_scaled_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({scale*mSchur_1},1,0));
					ctxt_scaled_mSchur_1 = cc->LevelReduce(ctxt_scaled_mSchur_1,nullptr,2*(t%Trefresh));	
				}	
				else // need a corner case because of the stop
				{
					int32_t tred = t - (int)((Tstop-2)/Trefresh)*Trefresh;
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()),1,0);
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
					ctxt_y = cc->LevelReduce(ctxt_y, nullptr, 2*tred);	
					ptxt_u = cc->MakeCKKSPackedPlaintext(u,1,0);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);	
					ctxt_u = cc->LevelReduce(ctxt_u, nullptr, 2*tred);	
					ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1},1,0));
					ctxt_mSchur_1 = cc->LevelReduce(ctxt_mSchur_1,nullptr,2*tred);			
					ctxt_scaled_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({scale*mSchur_1},1,0));
					ctxt_scaled_mSchur_1 = cc->LevelReduce(ctxt_scaled_mSchur_1,nullptr,2*tred);						
				}			
			}
			else
			{
				if (t < Tstop)
				{
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()));
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y);
					ptxt_u = cc->MakeCKKSPackedPlaintext(u);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);			
					if (t > 0)
					{
						ctxt_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1}));
						ctxt_scaled_mSchur_1 = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext({mSchur_1*scale}));
					}

					if ((t > 0) && (t%Trefresh == 0) ) 
					{
						Plaintext result_M_1;
						cc->Decrypt(keys.secretKey, ctxt_M_1_packed, &result_M_1);		
						result_M_1->SetLength(int((S+1)*(S+2)/2));
						std::vector<complex<double>> M_1_packed(int((S+1)*(S+2)/2));
						for (size_t i = 0; i < (S+1)*(S+2)/2; i ++)
						{
							M_1_packed[i] = mSchur_1 * result_M_1->GetCKKSPackedValue()[i];	
							M_1_packed[i].imag(0);
						}

						ctxt_M_1_packed = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(M_1_packed)); 	
					} 
				}
				else // t >= Tstop
				{
					int32_t dropLevels = ctxt_y->GetLevel(); // previous ctxt_y
					ptxt_y = cc->MakeCKKSPackedPlaintext(mat2Vec(plant->gety()),1,0);
					ctxt_y = cc->Encrypt(keys.publicKey, ptxt_y); 		
					ptxt_u = cc->MakeCKKSPackedPlaintext(u,1,0);
					ctxt_u = cc->Encrypt(keys.publicKey, ptxt_u);			
					ctxt_y = cc->LevelReduce(ctxt_y, nullptr, dropLevels);	
					ctxt_u = cc->LevelReduce(ctxt_u, nullptr, dropLevels);							
				}

			}		

			timeClientEnc = TOC(t1);
			cout << "Time for encoding and encrypting at the client at time " << t+1 << ": " << timeClientEnc << " ms" << endl;	

			Plaintext ptxt_interm;
			cout.precision(8);

			////////////// Back to the server.
			TIC(t1);

			if (t > 0)
			{
				if (t < Tstop)
				{
					if ( (t%Trefresh)!= 0 || (t > int((Tstop-2)/Trefresh)*Trefresh) )
					{
						ctxt_M_1[S][0] = ctxt_scaled_mSchur_1; // encryption of 1/mSchur of the last element on the diagonal	
						
						// We compute M_1 with fewer levels by multiplying 1/mSchur to mVecT first for t > 1.		
			 # pragma omp parallel for 
						for (size_t i = 0; i < S; ++i) // add -m_Schur*M_1*mVecT as the last column of M_1
							ctxt_M_1[i][S-i] = cc->EvalMult( ctxt_M_1[i][0], cc->Rescale( cc->EvalMult( ctxt_mVec[i], -ctxt_mSchur_1 ) ) ); 

						for (size_t i = 0; i < S; ++i)
							for (size_t j = 1; j < S-i; ++j)
							{	
								ctxt_M_1[i][S-i] = cc->EvalAdd( ctxt_M_1[i][S-i], cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i+j], -ctxt_mSchur_1 )) ));
								ctxt_M_1[i+j][S-i-j] = cc->EvalAdd( ctxt_M_1[i+j][S-i-j], cc->EvalMult( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_mVec[i], -ctxt_mSchur_1 )) ));					
							}

		# pragma omp parallel for 
						for (size_t i = 0; i < S; ++i)
							ctxt_M_1[i][S-i] = cc->Rescale(ctxt_M_1[i][S-i]);	
	
						for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
							ctxt_M_1[i][0] = cc->EvalSub( ctxt_M_1[i][0], cc->Rescale(cc->EvalMult( ctxt_M_1[i][S-i], ctxt_M_1mVec_s[i] )) );


						for (size_t i = 0; i < S; ++i) // change the first S-1 columns of M_1 (taking into account the symmetry)
							for (size_t j = 1; j < S-i; ++j)
								ctxt_M_1[i][j] = cc->EvalSub( ctxt_M_1[i][j], cc->Rescale(cc->EvalMult( ctxt_M_1[i][S-i], ctxt_M_1mVec_s[i+j] ))  );
						
					}
					else //(t%Trefresh)== 0
					{
						if (t < Tstop-1) // in this last case, we don't need to refresh M_1
						{
							// compute digits for fast rotations 
							auto M_1Precomp = cc->GetEncryptionAlgorithm()->EvalFastRotationPrecompute( ctxt_M_1_packed );
							for (int32_t i = 0; i < S+1; i ++)
							{
								for (int32_t j = 0; j < S+1-i; j ++)
								{
									if (i*(S+1) - int(i*(i-1)/2) + j == 0)
										ctxt_M_1[i][j] = cc->Rescale( cc->EvalMult( ctxt_M_1_packed, ptxt_1 ) );	
									else
										ctxt_M_1[i][j] = cc->Rescale( cc->EvalMult( cc->GetEncryptionAlgorithm()->EvalFastRotation(\
											ctxt_M_1_packed, i*(S+1) - int(i*(i-1)/2) + j, cyclOrder, M_1Precomp, *EvalUnpackKeys ), ptxt_1 ) );				
								}
							}	

							timeServerRefresh = TOC(t3);	
							cout << "Time for unpacking inv(M) at the server at step " << t << ": " << timeServerRefresh << " ms" << endl;																
						}

						// clear keys if this was the last refresh
						if (t == int((Tstop-2)/Trefresh)*Trefresh)
						{
							EvalUnpackKeys->clear();
						}									
					}			

					S += 1;
				}
			}

			if (t < Tstop)
			{
				ctxt_cHY.resize(S+1); ctxt_cHU.resize(S+1);	

				if ( (t > 0) && (t > (int)((Tstop-2)/Trefresh)*Trefresh ) )
				{
					ctxt_cHY[S-1] = cc->LevelReduce(ctxt_cHY[S-1], nullptr, 2);
					ctxt_cHU[S-1] = cc->LevelReduce(ctxt_cHU[S-1], nullptr, 2);
					ctxt_yini = cc->LevelReduce(ctxt_yini, nullptr, 2);
					ctxt_uini = cc->LevelReduce(ctxt_uini, nullptr, 2);
					ctxt_r = cc->LevelReduce(ctxt_r, nullptr, 2);
					CompressEvalKeys(*EvalRotKeys, 2);
				}	

				ctxt_cHY[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHY[S-1], plant->p, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, -Ty+plant->p, *EvalRotKeys) );
				ctxt_cHU[S] = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_cHU[S-1], plant->m, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -Tu+plant->m, *EvalRotKeys) );	

				ctxt_mVec.resize(S+1); ctxt_M_1mVec.resize(S+1);
				ctxt_mVec_s.resize(S+1); ctxt_M_1mVec_s.resize(S+1);					
			}	

			if (plant->M == 1)
			{
				ctxt_yini = ctxt_y;
				ctxt_uini = ctxt_u;
			}
			else 
			{
				ctxt_yini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_yini, plant->p, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_y, -plant->py+plant->p, *EvalRotKeys) );
				ctxt_uini = cc->EvalAdd( cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys),\
					cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -plant->pu+plant->m, *EvalRotKeys) );
			}

			timeServerUpdate = TOC(t1);
			cout << "Time for updating the M_1, Hankel matrices, uini and yini at the server at time " << t+1 << ": " << timeServerUpdate << " ms" << endl;	

			timeStep = TOC(t2);		
			cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	

			cout << "S = " << S << endl;
		}


	}	

	timeEval = TOC(t);
	cout << "Total time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	timeEval = TOC(t0);
	cout << "Total offline+online time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	plant->printYU(); // print all inputs and outputs at the end of the simulation


}

void OfflineFeedback()
{
	TimeVar t,t1,t2;
	TIC(t);
	double timeInit(0.0), timeEval(0.0), timeStep(0.0);
	double timeClientUpdate(0.0), timeClientDec(0.0), timeClientEnc(0.0), timeServer0(0.0), timeServer(0.0);

	/*
	 * Simulation parameters
	 */	
	uint32_t T = 5;

	int32_t flagSendU = 1;	

	uint32_t Trefresh = 3; // if flagSendU = 1, this parameter is automatically set to 1

	/* 
	 * Initialize the plant
	 */
	// Plant<complex<double>>* plant = plantInitStableSys(); // M = 1, N = 3, T = 10
	Plant<complex<double>>* plant = plantInitRoom(); // M = 4, N = 10, T = 40

	// Inputs
	std::vector<complex<double>> r(plant->getr().GetRows());
	mat2Vec(plant->getr(), r);
	std::vector<complex<double>> yini(plant->getyini().GetRows());
	mat2Vec(plant->getyini(), yini);
	std::vector<complex<double>> uini(plant->getuini().GetRows()); 
	mat2Vec(plant->getuini(), uini);

	// Which diagonals to extract depend on the relationship between the 
	// # of rows and the # of columns of the matrices

	Matrix<complex<double>> Kur = plant->getKur(); // if m < p it will be wide, otherwise tall
	Matrix<complex<double>> Kyini = plant->getKyini(); // if mN > pM it will be tall, otherwise wide
	Matrix<complex<double>> Kuini = plant->getKuini(); // in general it is tall

	size_t colsKur = Kur.GetCols(); size_t rowsKur = Kur.GetRows();
	size_t colsKyini = Kyini.GetCols(); size_t rowsKyini = Kyini.GetRows(); 
	size_t colsKuini = Kuini.GetCols(); size_t rowsKuini = Kuini.GetRows();

	// Baby step giant step choices
	int32_t BsGs_r = 0;
	int32_t dim1_r, dim2_r;
	if (rowsKur >= colsKur) // tall matrix
	{
		dim1_r = std::ceil(std::sqrt(colsKur));
		dim2_r = std::ceil((double)colsKur/dim1_r);		
	}
	else // wide matrix
	{
		if ( rowsKur % colsKur == 0 ) // wide Ef
		{
			dim1_r = std::ceil(std::sqrt(rowsKur)); 
			dim2_r = std::ceil((double)rowsKur/dim1_r);	
		}
		else //plain wide
		{
			dim1_r = std::ceil(std::sqrt(colsKur));
			dim2_r = std::ceil((double)colsKur/dim1_r);		
		}
	}
	int32_t BsGs_y = 0;
	int32_t dim1_y, dim2_y;
	if (rowsKyini >= colsKyini) // tall matrix
	{
		dim1_y = std::ceil(std::sqrt(colsKyini));
		dim2_y = std::ceil((double)colsKyini/dim1_y);		
	}
	else // wide matrix
	{
		if ( rowsKyini % colsKyini == 0 ) // wide Ef
		{
			dim1_y = std::ceil(std::sqrt(rowsKyini)); 
			dim2_y = std::ceil((double)rowsKyini/dim1_y);	
		}
		else // plain wide
		{
			dim1_y = std::ceil(std::sqrt(colsKyini));
			dim2_y = std::ceil((double)colsKyini/dim1_y);		
		}
	}

	int32_t BsGs_u = 0;
	int32_t dim1_u = std::ceil(std::sqrt(colsKuini));
	int32_t dim2_u = std::ceil((double)colsKuini/dim1_u);		

	int32_t BsGs_refresh = 0;
	int32_t dim1_refresh, dim2_refresh, dim_tot;
	if (BsGs_u == 0)
	{
		dim1_refresh = std::ceil(std::sqrt(colsKuini));
		dim2_refresh = std::ceil((double)colsKuini/dim1_refresh);	
		dim_tot = colsKuini;
	}		
	else
	{
		dim1_refresh = std::ceil(std::sqrt(dim1_u));
		dim2_refresh = std::ceil((double)dim1_u/dim1_refresh);	
		dim_tot = dim1_u;
	}

	std::vector<std::vector<complex<double>>> dKur;
	if (rowsKur >= colsKur) // tall
	{
		dKur.resize(colsKur);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKur; i++)
		 	dKur[i] = std::vector<complex<double>>(rowsKur);

		mat2HybridDiags(Kur, dKur);	
		if (BsGs_r == 1)
		{
# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_r; j ++)
				for (int32_t i = 0; i < dim1_r; i ++)
					if (dim1_r*j + i < colsKur)
					{
						Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, rowsKur));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKur % colsKur == 0) // wideEf
	 	{
			dKur.resize(rowsKur);
#pragma omp parallel for	
			for (size_t i = 0; i < rowsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);

			mat2HybridDiags(Kur, dKur);	
			if (BsGs_r == 1)
			{
# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < rowsKur)
						{
							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKur.resize(colsKur);
			for (size_t i = 0; i < colsKur; i ++)
			 	dKur[i] = std::vector<complex<double>>(colsKur);		 	

			mat2Diags(Kur, dKur);

			if (BsGs_r == 1)
			{
# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_r; j ++)
					for (int32_t i = 0; i < dim1_r; i ++)
						if (dim1_r*j + i < colsKur)
						{
							std::vector<complex<double>> temp1 = dKur[dim1_r*j+i];
							std::vector<complex<double>> temp2 = dKur[dim1_r*j+i];
							for (int32_t k = 1; k < colsKur/rowsKur; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKur%rowsKur,back_inserter(temp1));	
							dKur[dim1_r*j+i] = temp1;

 							Rotate(dKur[dim1_r*j+i],ReduceRotation(-dim1_r*j, colsKur));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKyini;
	if (rowsKyini >= colsKyini) // tall
	{
		dKyini.resize(colsKyini);
	#pragma omp parallel for	
		for (size_t i = 0; i < colsKyini; i++)
		 	dKyini[i] = std::vector<complex<double>>(rowsKyini);

		mat2HybridDiags(Kyini, dKyini);	
		if (BsGs_y == 1)
		{
# pragma omp parallel for 
			for (int32_t j = 0; j < dim2_y; j ++)
				for (int32_t i = 0; i < dim1_y; i ++)
					if (dim1_y*j + i < colsKyini)
					{
						Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, rowsKyini));
					}
	 	}	
	 }
	 else // wide
	 	if (rowsKyini % colsKyini == 0) // wideEf
	 	{
			dKyini.resize(rowsKyini);
#pragma omp parallel for	
			for (size_t i = 0; i < rowsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);

			mat2HybridDiags(Kyini, dKyini);	
			if (BsGs_y == 1)
			{
# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < rowsKyini)
						{
							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 
		 }	
		 else // plain wide
		 {
			dKyini.resize(colsKyini);
			for (size_t i = 0; i < colsKyini; i ++)
			 	dKyini[i] = std::vector<complex<double>>(colsKyini);		 	

			mat2Diags(Kyini, dKyini);

			if (BsGs_y == 1)
			{
# pragma omp parallel for 
				for (int32_t j = 0; j < dim2_y; j ++)
					for (int32_t i = 0; i < dim1_y; i ++)
						if (dim1_y*j + i < colsKyini)
						{
							std::vector<complex<double>> temp1 = dKyini[dim1_y*j+i];
							std::vector<complex<double>> temp2 = dKyini[dim1_y*j+i];
							for (int32_t k = 1; k < colsKyini/rowsKyini; k++)
								std::copy(temp2.begin(),temp2.end(),back_inserter(temp1));
							std::copy(temp2.begin(),temp2.begin()+colsKyini%rowsKyini,back_inserter(temp1));	
							dKyini[dim1_y*j+i] = temp1;

 							Rotate(dKyini[dim1_y*j+i],ReduceRotation(-dim1_y*j, colsKyini));
						}
		 	} 					
		 }

	std::vector<std::vector<complex<double>>> dKuini(colsKuini);
#pragma omp parallel for	
	for (size_t i = 0; i < colsKuini; i++)
	 	dKuini[i] = std::vector<complex<double>>(rowsKuini);

	mat2HybridDiags(Kuini, dKuini);	
	if (BsGs_u == 1)
	{
# pragma omp parallel for 
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < colsKuini)
				{
					Rotate(dKuini[dim1_u*j+i],ReduceRotation(-dim1_u*j, rowsKuini));
				}
	}	

	// Step 1: Setup CryptoContext

	// A. Specify main parameters
	/* A1) Multiplicative depth:
	 * The CKKS scheme we setup here will work for any computation
	 * that has a multiplicative depth equal to 'multDepth'.
	 * This is the maximum possible depth of a given multiplication,
	 * but not the total number of multiplications supported by the
	 * scheme.
	 *
	 * For example, computation f(x, y) = x^2 + x*y + y^2 + x + y has
	 * a multiplicative depth of 1, but requires a total of 3 multiplications.
	 * On the other hand, computation g(x_i) = x1*x2*x3*x4 can be implemented
	 * either as a computation of multiplicative depth 3 as
	 * g(x_i) = ((x1*x2)*x3)*x4, or as a computation of multiplicative depth 2
	 * as g(x_i) = (x1*x2)*(x3*x4).
	 *
	 * For performance reasons, it's generally preferable to perform operations
	 * in the shortest multiplicative depth possible.
	 */
	/* For the one-shot model-free control, we need a multDepth = 2, if the client sends back uini and yini,
	 * and a multDepth = 2t - 1 if we use whatever we want but need to mask and rotate the 
	 * result u* to construct uini.
	*/

	uint32_t multDepth; 
	if (flagSendU == 1)
		multDepth = 2;
	else
		multDepth = 2*Trefresh;

	/* A2) Bit-length of scaling factor.
	 * CKKS works for real numbers, but these numbers are encoded as integers.
	 * For instance, real number m=0.01 is encoded as m'=round(m*D), where D is
	 * a scheme parameter called scaling factor. Suppose D=1000, then m' is 10 (an
	 * integer). Say the result of a computation based on m' is 130, then at
	 * decryption, the scaling factor is removed so the user is presented with
	 * the real number result of 0.13.
	 *
	 * Parameter 'scaleFactorBits' determines the bit-length of the scaling
	 * factor D, but not the scaling factor itself. The latter is implementation
	 * specific, and it may also vary between ciphertexts in certain versions of
	 * CKKS (e.g., in EXACTRESCALE).
	 *
	 * Choosing 'scaleFactorBits' depends on the desired accuracy of the
	 * computation, as well as the remaining parameters like multDepth or security
	 * standard. This is because the remaining parameters determine how much noise
	 * will be incurred during the computation (remember CKKS is an approximate
	 * scheme that incurs small amounts of noise with every operation). The scaling
	 * factor should be large enough to both accommodate this noise and support results
	 * that match the desired accuracy.
	 */
	uint32_t scaleFactorBits = 50;

	/* A3) Number of plaintext slots used in the ciphertext.
	 * CKKS packs multiple plaintext values in each ciphertext.
	 * The maximum number of slots depends on a security parameter called ring
	 * dimension. In this instance, we don't specify the ring dimension directly,
	 * but let the library choose it for us, based on the security level we choose,
	 * the multiplicative depth we want to support, and the scaling factor size.
	 *
	 * Please use method GetRingDimension() to find out the exact ring dimension
	 * being used for these parameters. Give ring dimension N, the maximum batch
	 * size is N/2, because of the way CKKS works.
	 */
	// In the one-shot model-free control, we need the batch size to be the 
	// whole N/2 because we need to pack vectors repeatedly, without trailing 
	// zeros.
	uint32_t batchSize = max(plant->p, plant->m); // what to display
	uint32_t slots = 1024; // has to take into consideration not only security, but dimensions of matrices and how much trailing zeros are neeeded

	/* A4) Desired security level based on FHE standards.
	 * This parameter can take four values. Three of the possible values correspond
	 * to 128-bit, 192-bit, and 256-bit security, and the fourth value corresponds
	 * to "NotSet", which means that the user is responsible for choosing security
	 * parameters. Naturally, "NotSet" should be used only in non-production
	 * environments, or by experts who understand the security implications of their
	 * choices.
	 *
	 * If a given security level is selected, the library will consult the current
	 * security parameter tables defined by the FHE standards consortium
	 * (https://homomorphicencryption.org/introduction/) to automatically
	 * select the security parameters. Please see "TABLES of RECOMMENDED PARAMETERS"
	 * in  the following reference for more details:
	 * http://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf
	 */
	// SecurityLevel securityLevel = HEStd_128_classic;
	SecurityLevel securityLevel = HEStd_NotSet;

	RescalingTechnique rsTech = APPROXRESCALE; 
	// RescalingTechnique rsTech = EXACTRESCALE;
	KeySwitchTechnique ksTech = HYBRID;	// BV is so much slower!

	uint32_t dnum = 0;
	uint32_t maxDepth = 3;
	// This is the size of the first modulus
	uint32_t firstModSize = 60;	
	uint32_t relinWin = 10;
	MODE mode = OPTIMIZED; // Using ternary distribution	

	/* 
	 * The following call creates a CKKS crypto context based on the arguments defined above.
	 */
	CryptoContext<DCRTPoly> cc =
			CryptoContextFactory<DCRTPoly>::genCryptoContextCKKS(
			   multDepth,
			   scaleFactorBits,
			   batchSize,
			   securityLevel,
			   slots*4, 
			   rsTech,
			   ksTech,
			   dnum,
			   maxDepth,
			   firstModSize,
			   relinWin,
			   mode);

	uint32_t RD = cc->GetRingDimension();
	cout << "CKKS scheme is using ring dimension " << RD << endl;
	uint32_t cyclOrder = RD*2;
	cout << "CKKS scheme is using the cyclotomic order " << cyclOrder << endl << endl;


	// Enable the features that you wish to use
	cc->Enable(ENCRYPTION);
	cc->Enable(SHE);
	cc->Enable(LEVELEDSHE);

	// B. Step 2: Key Generation
	/* B1) Generate encryption keys.
	 * These are used for encryption/decryption, as well as in generating different
	 * kinds of keys.
	 */
	auto keys = cc->KeyGen();

	/* B2) Generate the relinearization key
	 * In CKKS, whenever someone multiplies two ciphertexts encrypted with key s,
	 * we get a result with some components that are valid under key s, and
	 * with an additional component that's valid under key s^2.
	 *
	 * In most cases, we want to perform relinearization of the multiplicaiton result,
	 * i.e., we want to transform the s^2 component of the ciphertext so it becomes valid
	 * under original key s. To do so, we need to create what we call a relinearization
	 * key with the following line.
	 */
	cc->EvalMultKeyGen(keys.secretKey);

	/* B3) Generate the rotation keys
	 * CKKS supports rotating the contents of a packed ciphertext, but to do so, we
	 * need to create what we call a rotation key. This is done with the following call,
	 * which takes as input a vector with indices that correspond to the rotation offset
	 * we want to support. Negative indices correspond to right shift and positive to left
	 * shift. Look at the output of this demo for an illustration of this.
	 *
	 * Keep in mind that rotations work on the entire ring dimension, not the specified
	 * batch size. This means that, if ring dimension is 8 and batch size is 4, then an
	 * input (1,2,3,4,0,0,0,0) rotated by 2 will become (3,4,0,0,0,0,1,2) and not
	 * (3,4,1,2,0,0,0,0). Also, as someone can observe in the output of this demo, since
	 * CKKS is approximate, zeros are not exact - they're just very small numbers.
	 */

	/* 
	 * Find rotation indices
	 */
	size_t maxNoRot = max(max(r.size(),yini.size()),uini.size());
	std::vector<int> indexVec(maxNoRot-1);
	std::iota (std::begin(indexVec), std::end(indexVec), 1);
	if (BsGs_r == 1)
	{
		if (dim1_r > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_r; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_r;
		}
		for (int32_t j = 0; j < dim2_r; j ++)
			for (int32_t i = 0; i < dim1_r; i ++)
				if (dim1_r*j + i < dKur.size())
				{
					if (rowsKur >= colsKur) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_r*j, rowsKur)); 
						indexVec.push_back(ReduceRotation(dim1_r*j, rowsKur));	
					}
					else // wide ef
						if (rowsKur % colsKur == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_r*j, colsKur)); 
							indexVec.push_back(ReduceRotation(dim1_r*j, colsKur));	
						}
				}
	}
	if (BsGs_y == 1)
	{
		if (dim1_y > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_y; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_y;
		}
		for (int32_t j = 0; j < dim2_y; j ++)
			for (int32_t i = 0; i < dim1_y; i ++)
				if (dim1_y*j + i < dKyini.size())
				{
					if (rowsKyini >= colsKyini) // tall
					{
						indexVec.push_back(ReduceRotation(-dim1_y*j, rowsKyini)); 
						indexVec.push_back(ReduceRotation(dim1_y*j, rowsKyini));	
					}
					else // wide
						if (rowsKyini % colsKyini == 0) // wide Ef
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));
						}
						else // plain wide
						{
							indexVec.push_back(ReduceRotation(-dim1_y*j, colsKyini)); 
							indexVec.push_back(ReduceRotation(dim1_y*j, colsKyini));							
						}
				}
	}
	if (BsGs_u == 1)
	{
		if (dim1_u > maxNoRot)
		{
			for (int32_t i = maxNoRot; i < dim1_u; i ++)
				indexVec.push_back(i);
			maxNoRot = dim1_u;
		}
		for (int32_t j = 0; j < dim2_u; j ++)
			for (int32_t i = 0; i < dim1_u; i ++)
				if (dim1_u*j + i < dKuini.size())
				{
					indexVec.push_back(ReduceRotation(-dim1_u*j, rowsKuini));
					indexVec.push_back(ReduceRotation(dim1_u*j, rowsKuini));
				}
	}	
	if (flagSendU != 1) // to compute new uini at the server
	{
		indexVec.push_back(plant->m);
		indexVec.push_back(-(plant->pu-plant->m));
		if (BsGs_u == 0)
		{
			// for (size_t i = 1; i < colsKuini; i ++ ) // for less efficient computations
			// 	indexVec.push_back(-i*(int)plant->pu);
			for (size_t i = 1; i <= std::floor(std::log2(colsKuini)); i ++ )
				indexVec.push_back(-pow(2,i-1)*(int)plant->pu);
			for (size_t i = pow(2,std::floor(std::log2(colsKuini))); i < colsKuini; i ++ )
				indexVec.push_back(-i*(int)plant->pu);
		}
		else
		{
			// for (size_t i = 1; i < dim1_u; i ++ ) // for less efficient computations
			// 	indexVec.push_back(-i*(int)plant->pu);
			for (size_t i = 1; i <= std::floor(std::log2(dim1_u)); i ++ )
				indexVec.push_back(-pow(2,i-1)*(int)plant->pu);
			for (size_t i = pow(2,std::floor(std::log2(dim1_u))); i < colsKuini; i ++ )
				indexVec.push_back(-i*(int)plant->pu);			
		}	

		if (BsGs_refresh == 1)
		{
			for (int32_t i = 1; i < dim1_refresh; i ++)
				indexVec.push_back(-i*(int)plant->pu);
			for (int32_t j = 0; j < dim2_refresh; j ++)
				for (int32_t i = 0; i < dim1_refresh; i ++)
					if (dim1_refresh*j + i < dim_tot)
					{
						indexVec.push_back(-(int)(dim1_refresh*j*plant->pu)); // check
					}	
		}
	}


	// remove any duplicate indices to avoid the generation of extra automorphism keys
	sort( indexVec.begin(), indexVec.end() );
	indexVec.erase( std::unique( indexVec.begin(), indexVec.end() ), indexVec.end() );
	//remove automorphisms corresponding to 0
	indexVec.erase(std::remove(indexVec.begin(), indexVec.end(), 0), indexVec.end());	

	auto EvalRotKeys = cc->GetEncryptionAlgorithm()->EvalAtIndexKeyGen(nullptr,keys.secretKey, indexVec);	

	/* 
	 * B4) Generate keys for summing up the packed values in a ciphertext
	 */

	// Step 3: Encoding and encryption of inputs

	// Encoding as plaintexts

	// Vectors r, yini and uini need to be repeated in the packed plaintext - bring issues with the rotations in baby step giant step?
	std::vector<std::complex<double>> rep_r(Fill(r,slots));
	Plaintext ptxt_r = cc->MakeCKKSPackedPlaintext(rep_r);

	std::vector<std::complex<double>> rep_yini(Fill(yini,slots));	
	Plaintext ptxt_yini = cc->MakeCKKSPackedPlaintext(rep_yini);

	std::vector<std::complex<double>> rep_uini(Fill(uini,slots));	
	Plaintext ptxt_uini = cc->MakeCKKSPackedPlaintext(rep_uini);

	std::vector<Plaintext> ptxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
	{
		if (BsGs_r == 1) 
		{
			std::vector<std::complex<double>> rep_dKur(Fill(dKur[i],slots)); 
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(rep_dKur);
		}
		else
			ptxt_dKur[i] = cc->MakeCKKSPackedPlaintext(dKur[i]);
	}

	std::vector<Plaintext> ptxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
	{
		if (BsGs_u == 1) 
		{
			std::vector<std::complex<double>> rep_dKuini(Fill(dKuini[i],slots)); // if we want to use baby step giant step
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(rep_dKuini);
		}
		else
			ptxt_dKuini[i] = cc->MakeCKKSPackedPlaintext(dKuini[i]);
	}

	std::vector<Plaintext> ptxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
	{
		if (BsGs_y == 1) 
		{
			std::vector<std::complex<double>> rep_dKyini(Fill(dKyini[i],slots)); // if we want to use baby step giant step
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(rep_dKyini);
		}
		else		
			ptxt_dKyini[i] = cc->MakeCKKSPackedPlaintext(dKyini[i]);
	}

	/* 
	 * Encrypt the encoded vectors
	 */
	auto ctxt_r = cc->Encrypt(keys.publicKey, ptxt_r);
	auto ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);
	auto ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKur(dKur.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKur.size(); ++i)
		ctxt_dKur[i] = cc->Encrypt(keys.publicKey, ptxt_dKur[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKyini(dKyini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKyini.size(); ++i)
		ctxt_dKyini[i] = cc->Encrypt(keys.publicKey, ptxt_dKyini[i]);

	std::vector<Ciphertext<DCRTPoly>> ctxt_dKuini(dKuini.size());
# pragma omp parallel for 
	for (size_t i = 0; i < dKuini.size(); ++i)
		ctxt_dKuini[i] = cc->Encrypt(keys.publicKey, ptxt_dKuini[i]);			


	timeInit = TOC(t);
	cout << "Time for offline key generation, encoding and encryption: " << timeInit << " ms" << endl;

	// Step 4: Evaluation

	TIC(t);

	TIC(t1);

	// Kur * r needs to only be performed when r changes! 
	// If the client agrees to send a signal when r changes, then we can save computations.
	// Otherwise, we compute Kur * r at every time step.

	// Matrix-vector multiplication for Kur*r
	Ciphertext<DCRTPoly> result_r;
	if ( rowsKur >= colsKur ) // tall
		result_r = EvalMatVMultTall(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
	else // wide
		if ( rowsKur % colsKur == 0) // wide Ef
		{
			result_r = EvalMatVMultWideEf(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	
		}
		else // plain wide
			result_r = EvalMatVMultWide(ctxt_dKur, ctxt_r, *EvalRotKeys, rowsKur, colsKur, BsGs_r, dim1_r);	

	timeServer0 = TOC(t1);
	cout << "Time for computing the constant values at the server at step 0: " << timeServer0 << " ms" << endl;		

	Ciphertext<DCRTPoly> ctxt_u;

	// Start online computations
	for (size_t t = 0; t < T; t ++)
	{
		cout << "t = " << t << endl << endl;

		TIC(t2);

		TIC(t1);

		// Matrix-vector multiplication for Kyini*yini; 
		Ciphertext<DCRTPoly> result_y;
		if ( rowsKyini >= colsKyini ) // tall
			result_y = EvalMatVMultTall(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
		else // wide
			if ( rowsKyini % colsKyini == 0) // wide Ef
				result_y = EvalMatVMultWideEf(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	
			else // plain wide
				result_y = EvalMatVMultWide(ctxt_dKyini, ctxt_yini, *EvalRotKeys, rowsKyini, colsKyini, BsGs_y, dim1_y);	

		// Matrix-vector multiplication for Kuini*uini; matVMultTall
		auto result_u = EvalMatVMultTall(ctxt_dKuini, ctxt_uini, *EvalRotKeys, rowsKuini, colsKuini, BsGs_u, dim1_u);	
	

		// Add the components
		ctxt_u = cc->EvalAdd ( result_u, result_y);
		ctxt_u = cc->EvalAdd ( ctxt_u, result_r );			


		timeServer = TOC(t1);
		cout << "Time for computing the control action at the server at step " << t << ": " << timeServer << " ms" << endl;	

		// If the client will not send uini, then the server has to update it
		if (flagSendU != 1 && ( (t+1) % Trefresh != 0 ) ) 
		{
			TIC(t1);

			std::vector<complex<double>> mask(slots);
			for (size_t i = plant->m; i < plant->pu; i ++)
				mask[i] = 1;
			Plaintext mask_ptxt = cc->MakeCKKSPackedPlaintext(mask);
			ctxt_uini = cc->EvalMult(ctxt_uini,mask_ptxt);							

			std::vector<complex<double>> mask_u(slots);
			for (size_t i = 0; i < plant->m; i ++)
				mask_u[i] = 1;		
			mask_ptxt = cc->MakeCKKSPackedPlaintext(mask_u);
			ctxt_u = cc->EvalMult(ctxt_u,mask_ptxt);		

			ctxt_u = cc->Rescale(ctxt_u);		

			if (ReduceRotation(-(plant->pu-plant->m),slots) != 0)
				ctxt_uini = cc->EvalAdd(cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys), cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_u, -(plant->pu-plant->m), *EvalRotKeys));	
			else
				ctxt_uini = cc->EvalAdd(cc->GetEncryptionAlgorithm()->EvalAtIndex(ctxt_uini, plant->m, *EvalRotKeys), ctxt_u);			

			// Only need to repeat uini the amount of times required by the diagonal method in the matrix vector multiplication

			ctxt_uini = EvalSumRot( ctxt_uini, *EvalRotKeys, dim_tot, plant->pu, BsGs_refresh, dim1_refresh );		

			ctxt_uini = cc->Rescale(ctxt_uini);

			timeServer = TOC(t1);
			cout << "Time for updating uini at the server at time " << t << " for the next time step: " << timeServer << " ms" << endl;		
		}

		TIC(t1);

		Plaintext result_u_t;
		cout.precision(8);
		cc->Decrypt(keys.secretKey, ctxt_u, &result_u_t);
		result_u_t->SetLength(plant->m);

		auto u = result_u_t->GetCKKSPackedValue(); // Make sure to make the imaginary parts to be zero s.t. error does not accumulate
		for (size_t i = 0; i < plant->m; i ++)
			u[i].imag(0);

		timeClientDec = TOC(t1);
		cout << "Time for decrypting the control action at the client at step " << t << ": " << timeClientDec << " ms" << endl;	

		TIC(t1);

		// Update plant
		plant->updatex(u);
		if (plant->M == 1)
		{
			uini = u;
			mat2Vec(plant->gety(),yini);
		}
		else
		{
			Rotate(uini, plant->m);
			std::copy(u.begin(),u.begin()+plant->m,uini.begin()+plant->pu-plant->m);
			Rotate(yini, plant->p);
			std::vector<complex<double>> y(plant->p);
			mat2Vec(plant->gety(),y);
			std::copy(y.begin(),y.begin()+plant->p,yini.begin()+plant->py-plant->p);			
		}

		// plant->printYU(); // if you want to print inputs and outputs at every time step
		plant->setyini(yini);
		plant->setuini(uini);


		timeClientUpdate = TOC(t1);
		cout << "Time for updating the plant at step " << t << ": " << timeClientUpdate << " ms" << endl;		

		TIC(t1);

		// Re-encrypt variables // Make sure to cut the number of levels if necessary!
		rep_yini = Fill(yini,slots);	
		ptxt_yini = cc->MakeCKKSPackedPlaintext(rep_yini);
		ctxt_yini = cc->Encrypt(keys.publicKey, ptxt_yini);

		if (flagSendU == 1 || ( (t+1) % Trefresh == 0 ) )
		{
			rep_uini = Fill(uini,slots);	
			ptxt_uini = cc->MakeCKKSPackedPlaintext(rep_uini);		
			ctxt_uini = cc->Encrypt(keys.publicKey, ptxt_uini);	
		}

		timeClientEnc = TOC(t1);
		cout << "Time for encoding and encrypting at the client at time " << t+1 << ": " << timeClientEnc << " ms" << endl;	

		timeStep = TOC(t2);		
		cout << "\nTotal time for evaluation at time " << t << ": " << timeStep << " ms" << endl << endl;	

	}	

	timeEval = TOC(t);
	cout << "Total time for evaluation for " << T << " steps: " << timeEval << " ms" << endl;	

	plant->printYU(); // print all inputs and outputs at the end of the simulation


}

Plant<complex<double>>* plantInitStableSys()
{
	auto zeroAlloc = [=]() { return 0; };

	std::string SYSTEM = "stable_system_";
	std::string FILETYPE = ".txt";

	// Construct plant

	uint32_t n = 4, m = 2, p = 4; 
	// double W = 0.001, V = 0.01; // set noise parameters
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
	uint32_t Tini = 1;
	uint32_t Tfin = 3;
	uint32_t T = 10;
	plant->M = Tini; plant->N = Tfin;

	/* 
	 * We don't consider concatenation of trajectories in this system example 
	 */
	lbcrypto::Matrix<complex<double>> ud = lbcrypto::Matrix<complex<double>>(zeroAlloc, m, T);
	lbcrypto::Matrix<complex<double>> yd = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, T);
	readMatrix(ud, p, DATAFOLDER + SYSTEM + "ud" + FILETYPE);	
	readMatrix(yd, m, DATAFOLDER + SYSTEM + "yd" + FILETYPE);	

	uint32_t pu = m*Tini;
	uint32_t fu = m*Tfin;
	uint32_t py = p*Tini;
	uint32_t fy = p*Tfin;

	plant->precollect(ud, pu, fu, 1);
	plant->precollect(yd, py, fy, 0);

	lbcrypto::Matrix<complex<double>> uini = lbcrypto::Matrix<complex<double>>(zeroAlloc, pu, 1);
	lbcrypto::Matrix<complex<double>> yini = lbcrypto::Matrix<complex<double>>(zeroAlloc, py, 1);
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
	lbcrypto::Matrix<complex<double>> r = lbcrypto::Matrix<complex<double>>(zeroAlloc, fy, 1);
	readVector(r, DATAFOLDER + SYSTEM + "ry" + FILETYPE, 0);

	// plant->constLQR(); // the inversion is too slow so we read the inverse from file
	lbcrypto::Matrix<complex<double>> K = lbcrypto::Matrix<complex<double>>(zeroAlloc, plant->S, plant->S);
	readMatrix(K, fu, DATAFOLDER + SYSTEM + "K" + FILETYPE);
	
	plant->constLQR(K);

	plant->setr(r);
	plant->setyini(yini);
	plant->setuini(uini);

	/* 
	 * Uncomment to see the plaintext behavior
	 */
	// uint32_t N = 10; 

	// for(int i = 0; i < N; i ++)
	// {
	// 	cout << "i = " << i << endl;

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

	return plant;
}



Plant<complex<double>>* plantInitRoom()
{
	auto zeroAlloc = [=]() { return 0; };

	std::string SYSTEM = "room_";
	std::string FILETYPE = ".txt";

	// Construct plant

	uint32_t n = 4, m = 1, p = 1; 
	// double W = 0.001, V = 0.01; // set noise parameters
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

	/* This does not work for trajectory concatenation
	// lbcrypto::Matrix<complex<double>> ud = lbcrypto::Matrix<complex<double>>(zeroAlloc, m, T);
	// lbcrypto::Matrix<complex<double>> yd = lbcrypto::Matrix<complex<double>>(zeroAlloc, p, T);
	// readMatrix(yd, p, DATAFOLDER + SYSTEM + "yd" + FILETYPE);	
	// readMatrix(ud, m, DATAFOLDER + SYSTEM + "ud" + FILETYPE);	
	// plant->precollect(ud, pu, fu, 1);
	// plant->precollect(yd, py, fy, 0);
	*/

	/* 
	 * This takes directly the Hankel matrices for the concatenated trajectory values
	 */
	lbcrypto::Matrix<complex<double>> HU = lbcrypto::Matrix<complex<double>>(zeroAlloc, pu+fu, T-(Tini+Tfin)+1);
	lbcrypto::Matrix<complex<double>> HY = lbcrypto::Matrix<complex<double>>(zeroAlloc, py+fy, T-(Tini+Tfin)+1);
	readMatrix(HU, pu+fu, DATAFOLDER + SYSTEM + "HU" + FILETYPE);	
	readMatrix(HY, py+fy, DATAFOLDER + SYSTEM + "HY" + FILETYPE);	

	plant->precollectH(HU, pu, fu, 1);
	plant->precollectH(HY, py, fy, 0);

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

	// plant->constLQR(); // the inversion is too slow so read the inverse from file as well
	lbcrypto::Matrix<complex<double>> K = lbcrypto::Matrix<complex<double>>(zeroAlloc, plant->S, plant->S);
	readMatrix(K, fu, DATAFOLDER + SYSTEM + "K" + FILETYPE);

	plant->constLQR(K);

	plant->setr(r);
	plant->setyini(yini);
	plant->setuini(uini);

	cout.precision(8);

	/* 
	 * Uncomment to see the plaintext behavior
	 */
	// uint32_t N = 5; 

	// for(int i = 0; i < N; i ++)
	// {
	// 	cout << "i = " << i << endl;

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

	// 	plant->printYU();

	// 	if (i < N - 1)
	// 		plant->onlineLQR();
	// }

	// plant->setyini(yini);
	// plant->setuini(uini);

	// // Without updating: "offline" feedback

	// uint32_t Nnon = 10;
	// for(int i = 0; i < Nnon; i ++)
	// {
	// 	cout << "i = " << i << endl;

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

	// 	plant->printYU();

	// 	K = plant->getM_1();

	// 	plant->constLQR(K);		

	// }	

	return plant;
}

