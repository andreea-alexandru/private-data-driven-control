/*
 * Helper functions. 
 *
 */
#ifndef __HELPERCONTROL_H
#define __HELPERCONTROL_H

#include <vector>
#include "../../core/lib/math/matrix.cpp"
#include "omp.h"
// #include <fstream>
// #include <iostream>
// #include <sstream>
// #include <iterator> // back_inserter

size_t ReduceRotation(size_t index, size_t size);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRot(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t n, const size_t size, const size_t BsGs=0, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRotBsGs(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t n, const size_t size, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTall(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, const size_t BsGs=0, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTallBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWide(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, const size_t BsGs=0, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEf(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, const size_t BsGs=0, size_t dim1=0 );

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEfBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
		const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
		const size_t rows, const size_t cols, size_t dim1=0 );

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a matrix object.
 */
template<class templElement>
lbcrypto::Matrix<templElement> extractHybridDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	auto zeroAlloc = [=]() { return templElement(0); };
	lbcrypto::Matrix<templElement> result = lbcrypto::Matrix<templElement>(zeroAlloc, max(rows,cols), 1);
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 *
 */
template<class templElement>
void extractHybridDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d, lbcrypto::Matrix<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a std::vector object.
 */
template<class templElement>
std::vector<templElement> extractHybridDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	std::vector<templElement> result(max(rows,cols));
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the extended diagonal d of a possibly rectangular matrix.
 * There will be min(rows,cols) possible diagonals with max(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 *
 */
template<class templElement>
void extractHybridDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d, std::vector<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= min(rows,cols)) {
        throw std::invalid_argument("There are fewer hybrid diagonals than specified. Maybe you want to use extractDiag?");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
}


/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a matrix object.
 */
template<class templElement>
lbcrypto::Matrix<templElement> extractDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	auto zeroAlloc = [=]() { return templElement(0); };
	lbcrypto::Matrix<templElement> result = lbcrypto::Matrix<templElement>(zeroAlloc, min(rows,cols), 1);
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 */
template<class templElement>
void extractDiag(lbcrypto::Matrix<templElement> const& M, uint32_t d, lbcrypto::Matrix<templElement>& result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result(row,0) = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result(row,0) = M(row%rows,(row+d)%cols);
	}
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 *
 * @return the resulting diagonal as a std::vector object.
 */
template<class templElement>
std::vector<templElement> extractDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	std::vector<templElement> result(min(rows,cols));
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
	return result;
}

/** 
 * Extract the reduced diagonal d of a possibly rectangular matrix.
 * There will be max(rows,cols) possible diagonals with min(rows,cols) elements.
 *
 * @param &M matrix
 * @param d the diagonal to extract
 * @param &result the reference of the result
 */
template<class templElement>
void extractDiag2Vec(lbcrypto::Matrix<templElement> const& M, uint32_t d, std::vector<templElement> &result)
{
	size_t rows = M.GetRows();
	size_t cols = M.GetCols();
	if (d >= max(rows,cols)) {
        throw std::invalid_argument("There are fewer diagonals than specified.");
    }	
	if (rows >= cols)
	{
#pragma omp parallel for
		for (size_t row = 0; row < cols; row++)
			result[row] = M((row-int((row+d)/cols)*cols%rows)%rows,(row+d)%cols);
	}
	else
	{
#pragma omp parallel for		
		for (size_t row = 0; row < rows; row++)
			result[row] = M(row%rows,(row+d)%cols);
	}
}

/** 
 * Transform a row matrix or a column matrix into an std::vector object.
 *
 * @param &M matrix
 * @param &result the reference of the result
 */
template<class templElement>
void mat2Vec(lbcrypto::Matrix<templElement> const& M, std::vector<templElement>& vec)
{
	if ( M.GetRows()!=1 && M.GetCols()!=1 )
		throw std::invalid_argument("This function is designed for matrices that are one row or one column.");

	if ( M.GetRows() == 1 )
	{
#pragma omp parallel for
		for (size_t col = 0; col < M.GetCols(); col++)
			vec[col] = M(0,col);		
	}
	else if ( M.GetCols() == 1 )
	{
#pragma omp parallel for
		for (size_t row = 0; row < M.GetRows(); row++)
			vec[row] = M(row,0);
	}		
}

/** 
 * Transform a row matrix or a column matrix into an std::vector object.
 *
 * @param &M matrix
 * @param &result the reference of the result
 */
template<class templElement>
std::vector<templElement> mat2Vec(lbcrypto::Matrix<templElement> const& M)
{
	if ( M.GetRows()!=1 && M.GetCols()!=1 )
		throw std::invalid_argument("This function is designed for matrices that are one row or one column.");

	std::vector<templElement> vec;

	if ( M.GetRows() == 1 )
	{
		vec.resize(M.GetCols());
#pragma omp parallel for
		for (size_t col = 0; col < M.GetCols(); col++)
			vec[col] = M(0,col);		
	}
	else if ( M.GetCols() == 1 )
	{
		vec.resize(M.GetRows());
#pragma omp parallel for
		for (size_t row = 0; row < M.GetRows(); row++)
			vec[row] = M(row,0);
	}		

	return vec;
}

/** 
 * Transform a matrix into a vector of column vectors.
 *
 * @param &M matrix
 * @param &Cols the reference of the result
 */
template<class templElement>
void mat2Cols(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& Cols)
{
#pragma omp parallel for
	for (size_t col = 0; col < M.GetCols(); col++){
		mat2Vec(M.ExtractCol(col),Cols[col]);
	}
}

/** 
 * Transform a matrix into a vector of vectors that represent the reduced diagonals.
 *
 * @param &M matrix
 * @param &Diags the reference of the result
 */
template<class templElement>
void mat2Diags(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& Diags)
{
#pragma omp parallel for
	for (size_t i = 0; i < max(M.GetRows(),M.GetCols()); i++)
		Diags[i] = extractDiag2Vec(M, i);
}


/** 
 * Transform a matrix into a vector of vectors that represent the extented/hybrid diagonals.
 *
 * @param &M matrix
 * @param &HDiags the reference of the result
 */
template<class templElement>
void mat2HybridDiags(lbcrypto::Matrix<templElement> const& M, std::vector<std::vector<templElement>>& HDiags)
{
#pragma omp parallel for
	for (size_t i = 0; i < min(M.GetRows(),M.GetCols()); i++)
		HDiags[i] = extractHybridDiag2Vec(M, i);
}


/**
 * Rotates a vector by an index - left rotation; positive index = left rotation
 *
 * @param &vec input vector.
 * @param index rotation index.
 * @param &result the rotated vector
 */
template<class templElement>
void Rotate(std::vector<templElement> &vec, int32_t index) {

	int32_t size = vec.size();

	std::vector<templElement> copy = vec;

	if (index < 0 || index > size){
		index = ReduceRotation(index,size);
	}

	if (index != 0){
		// two cases: i+index <= slots and i+index > slots
		for(int32_t i = 0; i < size-index; i++)
			vec[i] = copy[i+index];
		for(int32_t i = size-index; i < size; i++)
			vec[i] = copy[i+index-size];
	}
}

template<class templElement>
std::vector<templElement> Fill(const std::vector<templElement> &vec, int slots) {

	int vecSize = vec.size();

	std::vector<templElement> result(slots);

	for (int i = 0; i < slots; i++)
		result[i] = vec[i % vecSize];

	return result;
}

/**
 * Performs the inner rotate and multiply in a matrix-vector multiplication in an optimized way, using the baby step giant step method.
 *
 * @param & diags the vector of diagonals that represent the matrix
 * @param &v input vector
 * @param &result the resulting vector
 * @param dim1 is the giant step
 */
template<class templElement>
void BsGsMult(std::vector<std::vector<templElement>> const& diags, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t dim1=0)
{
	int32_t N = diags.size();
	if (dim1 == 0)
		dim1 = std::ceil(std::sqrt(N));
	int32_t dim2 = std::ceil((double)N/dim1);

	std::vector<templElement> rotDiags(diags[0].size());
	std::vector<templElement> rotV(v.size());

	std::vector<templElement> temp(diags[0].size());	

	result = Fill(temp,result.size()); // make sure there are zeros inside

	std::vector<templElement> sum(rotDiags);	

	for (int32_t j = 0; j < dim2; j ++)
	{
		std::fill(sum.begin(), sum.end(), templElement(0));

		for (int32_t i = 0; i < dim1; i ++)
		{
			if (dim1*j + i < N)
			{
				std::fill(temp.begin(), temp.end(), templElement(0));
				rotV = v;
				if (ReduceRotation(i, v.size())!=0)
					Rotate(rotV,ReduceRotation(i, v.size()));

				rotDiags = diags[dim1*j+i];
				if (ReduceRotation(-dim1*j, rotDiags.size()) != 0) 
					Rotate(rotDiags, ReduceRotation(-dim1*j, rotDiags.size()));

		 		// Perform element-wise multiplication, with the result stored in result
				std::transform( rotDiags.begin(), rotDiags.end(), rotV.begin(), temp.begin(), std::multiplies<templElement>() );	

				std::transform( sum.begin(), sum.end(), temp.begin(), sum.begin(), std::plus<templElement>() );
					
			}
		}

		Rotate(sum, ReduceRotation(dim1*j, sum.size()));		

		std::transform( result.begin(), result.end(), sum.begin(), result.begin(),std::plus<templElement>() );
	
	}

}

/**
 * Performs the inner rotate and multiply in a matrix-vector multiplication in an optimized way, using the baby step giant step method.
 *
 * @param & diags the vector of diagonals that represent the matrix
 * @param &v input vector
 * @param &result the resulting vector
 * @param dim1 is the giant step
 */
template<class templElement>
void BsGsMultWide(std::vector<std::vector<templElement>> const& diags, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t dim1=0)
{
	int32_t N = diags.size();
	if (dim1 == 0)
		dim1 = std::ceil(std::sqrt(N));
	int32_t M = diags[0].size();
	int32_t dim2 = std::ceil((double)N/dim1);

	std::vector<templElement> rotDiags(N);
	std::vector<templElement> rotV(N);

	std::vector<templElement> temp(N);	

	result = Fill(temp,result.size()); // make sure there are zeros inside

	std::vector<templElement> sum(N);	

	for (int32_t j = 0; j < dim2; j ++)
	{
		std::fill(sum.begin(), sum.end(), templElement(0));

		for (int32_t i = 0; i < dim1; i ++)
		{
			if (dim1*j + i < N)
			{
				std::fill(temp.begin(), temp.end(), templElement(0));
				rotV = v;
				if (ReduceRotation(i, N)!=0)
					Rotate(rotV,ReduceRotation(i, N));

				rotDiags = diags[dim1*j+i];
				for (size_t k = 1; k < N/M ; k++)
					std::copy(diags[dim1*j+i].begin(),diags[dim1*j+i].end(),back_inserter(rotDiags));
				std::copy(diags[dim1*j+i].begin(),diags[dim1*j+i].begin()+N%M,back_inserter(rotDiags));					

				if (ReduceRotation(-dim1*j, N) != 0) 
					Rotate(rotDiags, ReduceRotation(-dim1*j, N));

		 		// Perform element-wise multiplication, with the result stored in result
				std::transform( rotDiags.begin(), rotDiags.end(), rotV.begin(), temp.begin(), std::multiplies<templElement>() );	

				std::transform( sum.begin(), sum.end(), temp.begin(), sum.begin(), std::plus<templElement>() );
					
			}
		}

		Rotate(sum, ReduceRotation(dim1*j, N));	

		std::transform( result.begin(), result.end(), sum.begin(), result.begin(),std::plus<templElement>() );

	}

}

/**
 * Multiplies a wide matrix (rows < cols) by a vector, using short diagonals.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultWide(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t BsGs = 0, int32_t dim1 = 0)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the reduced diagonal representation from M	
	std::vector<std::vector<templElement>> shortDiags(cols);
#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	 	shortDiags[i] = std::vector<templElement>(rows);

	mat2Diags(M, shortDiags);

	if (BsGs == 0) // don't use Baby step Giant step method
	{
		// Obtain the rotated versions of v
		std::vector<std::vector<templElement>> rotV(cols);
#pragma omp parallel for	
		for (size_t i = 0; i < cols; i++)
		{
		 	rotV[i] = v;
		 	Rotate(rotV[i],i);
		 	// Perform element-wise multiplication, with the result stored in shortDiags
			std::transform( shortDiags[i].begin(), shortDiags[i].end(), rotV[i].begin(), shortDiags[i].begin(),std::multiplies<templElement>() );	
		}


		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
			for (int32_t i = 0; i < std::ceil(cols/pow(2,h)); i++){
				if (i + std::ceil(cols/pow(2,h)) < std::ceil(cols/pow(2,h-1))){
					std::transform( shortDiags[i].begin(), shortDiags[i].end(), shortDiags[i+std::ceil(cols/pow(2,h))].begin(), shortDiags[i].begin(),std::plus<templElement>() );
				}
			}
		}
		result = shortDiags[0];
	}
	else // use Baby step Giant step method
	{
		BsGsMultWide(shortDiags, v, result, dim1);
	}


}

/**
 * Multiplies a tall matrix (cols < rows) by a vector, using extended diagonals.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultTall(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t BsGs = 0, int32_t dim1 = 0)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the extended diagonal representation from M	
	std::vector<std::vector<templElement>> extDiags(cols);
#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	 	extDiags[i] = std::vector<templElement>(rows);

	mat2HybridDiags(M, extDiags);

	// Obtain the rotated versions of v, which now should be v repeated until the closest multiple of the number of cols > the number of rows is reached
	std::vector<std::vector<templElement>> rotV(std::ceil((double)rows/cols)*cols);
	std::vector<templElement> extv = v;
	for (size_t i = 0; i < rows/cols ; i++)
		std::copy(v.begin(),v.end(),back_inserter(extv));


	if (BsGs == 0)
	{
// #pragma omp parallel for	
		for (size_t i = 0; i < cols; i++)
		{
		 	rotV[i] = extv;

		 	Rotate(rotV[i],i);

			std::transform( extDiags[i].begin(), extDiags[i].end(), rotV[i].begin(), extDiags[i].begin(),std::multiplies<templElement>() );	 	
		}


		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
			for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
				if (i + std::ceil((double)cols/pow(2,h)) < std::ceil(cols/pow(2,h-1))){
					std::transform( extDiags[i].begin(), extDiags[i].end(), extDiags[i+std::ceil(double(cols)/pow(2,h))].begin(), extDiags[i].begin(),std::plus<templElement>() );
				}
			}
		}	
		result = extDiags[0];	
	}
	else // use Baby step Giant step method
	{
		BsGsMult(extDiags, extv, result, dim1);
	}	

}

/**
 * Multiplies a wide matrix (rows < cols) by a vector, using extended diagonals. 
 * This is possible when rows divides cols.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultWideEf(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t BsGs = 0, int32_t dim1 = 0)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Obtain the extended diagonal representation from M	
	std::vector<std::vector<templElement>> extDiags(rows);
#pragma omp parallel for	
	for (size_t i = 0; i < rows; i++)
	 	extDiags[i] = std::vector<templElement>(cols);

	mat2HybridDiags(M, extDiags);

	std::vector<templElement> temp(cols);


	if (BsGs == 0)
	{
		std::vector<std::vector<templElement>> rotV(cols);

#pragma omp parallel for	
		for (size_t i = 0; i < rows; i++)
		{
		 	rotV[i] = v;
		 	Rotate(rotV[i],i);
		 	// Perform element-wise multiplication, with the result stored in extDiags
			std::transform( extDiags[i].begin(), extDiags[i].end(), rotV[i].begin(), extDiags[i].begin(),std::multiplies<templElement>() );	 	
		}

		// Perform binary tree addition
		for (int32_t h = 1; h <= std::ceil(std::log2(rows)); h++){
			for (int32_t i = 0; i < std::ceil(double(rows)/pow(2,h)); i++){
				if (i + std::ceil(double(rows)/pow(2,h)) < std::ceil(double(rows)/pow(2,h-1)))
					std::transform( extDiags[i].begin(), extDiags[i].end(), extDiags[i+std::ceil(double(rows)/pow(2,h))].begin(), 
						extDiags[i].begin(),std::plus<templElement>() );
			}
		}

		temp = extDiags[0];

	}

	else
	{
		BsGsMult(extDiags, v, temp, dim1);
	}

	double q = double(cols)/double(rows);
	double logq = std::ceil(std::log2(q));
	double aq = pow(2,logq);

	// Expand the vector by zeros up to the closest power of 2 from #rows*ceil(log(#columns/#rows))
	std::vector<templElement> res(rows*aq);	
	std::copy(temp.begin(), temp.end(), res.begin());		

	// Perform binary tree rotations
	for (int32_t h = 1; h <= logq; h ++){
		std::transform( res.begin(), res.begin()+rows*(aq/pow(2,h)), 
			res.begin()+rows*(aq/pow(2,h)), res.begin(),std::plus<templElement>() );		
	}

	std::copy(res.begin(), res.begin()+rows, result.begin());	
}

/**
 * Multiplies a matrix by a vector.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMult(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result, int32_t BsGs = 0, int32_t dim1 = 0)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");
	if (M.GetRows() <= M.GetCols()) // wide
		if (M.GetCols() % M.GetRows())
			matVMultWide(M, v, result, BsGs, dim1);
		else // less storage, more rotations
			matVMultWideEf(M, v, result, BsGs, dim1);
	else // tall
		matVMultTall(M, v, result, BsGs, dim1);
}

/**
 * Multiplies a matrix by a vector using the column method.
 *
 * @param & M input matrix
 * @param &v input vector.
 * @param &result the resulting vector
 */
template<class templElement>
void matVMultCol(lbcrypto::Matrix<templElement> const& M, std::vector<templElement> const& v, std::vector<templElement>& result)
{
	if ( M.GetCols() != v.size())
		throw std::invalid_argument("# of columns of the matrix does not match the size of the vector.");

	size_t rows = M.GetRows();
	size_t cols = M.GetCols();

	// Extract the columns of the matrix M
	std::vector<std::vector<templElement>> Cols(cols);
	for (size_t i = 0; i < cols; i++)
		Cols[i] = std::vector<templElement>(rows);
	mat2Cols(M, Cols);

#pragma omp parallel for	
	for (size_t i = 0; i < cols; i++)
	{
		for (size_t j = 0; j < rows; j++)
		{
	 	// Perform multiplication between a column and its corresponding element in v, with the result stored in Cols
			Cols[i][j] *= v[i];
		}
	}	

	// Perform binary tree addition
	for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
		for (int32_t i = 0; i < std::ceil(cols/pow(2,h)); i++){
			if (i + std::ceil(cols/pow(2,h)) < std::ceil(cols/pow(2,h-1)))
				std::transform( Cols[i].begin(), Cols[i].end(), Cols[i+std::ceil(cols/pow(2,h))].begin(), Cols[i].begin(),std::plus<templElement>() );
		}
	}
	result = Cols[0];
}

// reads matrix in std::vector<std::vector>>
template<class templElement>
void readMatrix(std::vector<std::vector<templElement>>& matrix, size_t rows, const std::string filename)
{

    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        
        size_t row = 0;
        while (std::getline(in, line)) {
            std::istringstream is(line);
            // matrix[row] should not be initialized for the back_inserter to work
            std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(matrix[row]));
            row++;
        }
    } 
    in.close();
}

// reads matrix in lbcrypto::Matrix
template<class templElement>
void readMatrix(lbcrypto::Matrix<templElement>& matrix, size_t rows, const std::string filename)
{

    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        
        size_t row = 0;
        while (std::getline(in, line)) {
            std::istringstream is(line);
            std::vector<templElement> v;
            // matrix[row] should not be initialized for the back_inserter to work
            std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(v));
            matrix.SetRow(row, v);
            row++;
        }
    } 
    in.close();
}

// reads vector in std::vector
template<class templElement>
void readVector(std::vector<templElement>& vec, const std::string filename)
{
    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        // vec should not be initialized for the back_inserter to work
        std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(vec));
    } 
    in.close();
}


// reads vector in lbcrypto::Matrix, as a row if row=0 and as a column otherwise
template<class templElement>
void readVector(lbcrypto::Matrix<templElement>& vec, const std::string filename, int32_t row)
{
    std::ifstream in(filename, std::ifstream::in|std::ios::binary);

    if (in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        std::vector<templElement> v;
        // vec should not be initialized for the back_inserter to work
        std::copy(std::istream_iterator<templElement>(is), std::istream_iterator<templElement>(), std::back_inserter(v));
        if (row == 1)
        	vec.SetRow(0, v);
        else
        	vec.SetCol(0, v);
    } 
    in.close();
}

//////////////////////// Functions for diagonal matrix-vector multiplication and Rotate and Sum //////////////////////// 
	/**
	* EvalMatVMultTall - Computes the product between a tall matrix and a vector using tree additions
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen.
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTall(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, const size_t BsGs, size_t dim1 ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		if (BsGs == 0)
		{
			std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
			ctxt_vRot[0] = ctxt_v; 
		
		// Element-wise multiplication of the extended diagonals with the rotated vectors
		ctxt_vRot[0] = cc->EvalMult( ctxt_diags[0], ctxt_vRot[0] );		
	#pragma omp parallel for	
			for (size_t i = 1; i < cols; i++){
				ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);
				ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );		
			}

			// Perform binary tree addition
			for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
				for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
					if (i + std::ceil((double)cols/pow(2,h)) < std::ceil((double)cols/pow(2,h-1))){
						ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)cols/pow(2,h))] );

					}					
				}
			}
			result = ctxt_vRot[0];
		} 
		else // baby step giant step
		{
			result = EvalMatVMultTallBsGs( ctxt_diags, ctxt_v, evalKeys, rows, cols, dim1 );
		}

		return result;
	}		

	/**
	* EvalMatVMultTallBsGs - Computes the product between a tall matrix and a vector using the baby step giant step optimization
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultTallBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, size_t dim1 ) {

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);		

		if (dim1 == 0) // but need to somehow make sure that the automorphism keys are generated for this value
			dim1 = std::ceil(std::sqrt(cols));
		int32_t dim2 = std::ceil((double)cols/dim1);		

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));		

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(dim1-1);

		// hoisted automorphisms
	#pragma omp parallel for
		for (uint32_t i = 1; i < dim1; i++)
			ctxt_vRot[i-1] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);

		for (int32_t j = 0; j < dim2; j ++)
		{

			auto sum = cc->EvalMult(ctxt_diags[dim1*j],ctxt_v);

			for (int32_t i = 1; i < dim1; i ++)
			{
				if (dim1*j + i < cols)
					sum = cc->EvalAdd(sum,cc->EvalMult(ctxt_diags[dim1*j+i],ctxt_vRot[i-1]));		
			}

			if (j == 0)
				result = sum;
			else
				result = cc->EvalAdd(result,cc->GetEncryptionAlgorithm()->EvalAtIndex(sum,ReduceRotation(dim1*j,rows), evalKeys));								

		}

			return result;
	}	

	/**
	* EvalMatVMultWide - Computes the product between a wide matrix and a vector using tree additions
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen.
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWide(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, const size_t BsGs, size_t dim1 ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		if (BsGs == 0)
		{
			std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
			ctxt_vRot[0] = ctxt_v; 
		
		// Element-wise multiplication of the extended diagonals with the rotated vectors
		ctxt_vRot[0] = cc->EvalMult( ctxt_diags[0], ctxt_vRot[0] );		
	#pragma omp parallel for	
			for (size_t i = 1; i < cols; i++){
				ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);
				ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );		
			}

			// Perform binary tree addition
			for (int32_t h = 1; h <= std::ceil(std::log2(cols)); h++){
				for (int32_t i = 0; i < std::ceil((double)cols/pow(2,h)); i++){
					if (i + std::ceil((double)cols/pow(2,h)) < std::ceil((double)cols/pow(2,h-1))){
						ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)cols/pow(2,h))] );

					}					
				}
			}
			result = ctxt_vRot[0];
		} 
		else // baby step giant step
		{
			result = EvalMatVMultWideBsGs( ctxt_diags, ctxt_v, evalKeys, rows, cols, dim1 );
		}

		return result;
	}		

	/**
	* EvalMatVMultWideBsGs - Computes the product between a wide matrix and a vector using the baby step giant step optimization
	* @param diags - the matrix is represented as a vector of extended diagonals
	* @param v - the vector to multiply with
	* @param &evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, size_t dim1 ) {

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);		

		if (dim1 == 0) // but need to somehow make sure that the automorphism keys are generated for this value
			dim1 = std::ceil(std::sqrt(cols));
		int32_t dim2 = std::ceil((double)cols/dim1);		

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));		

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(dim1-1);

		// hoisted automorphisms
	#pragma omp parallel for
		for (uint32_t i = 1; i < dim1; i++)
			ctxt_vRot[i-1] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);

		for (int32_t j = 0; j < dim2; j ++)
		{

			auto sum = cc->EvalMult(ctxt_diags[dim1*j],ctxt_v);

			for (int32_t i = 1; i < dim1; i ++)
			{
				if (dim1*j + i < cols)
					sum = cc->EvalAdd(sum,cc->EvalMult(ctxt_diags[dim1*j+i],ctxt_vRot[i-1]));		
			}

			if (j == 0)
				result = sum;
			else
				result = cc->EvalAdd(result,cc->GetEncryptionAlgorithm()->EvalAtIndex(sum, ReduceRotation(dim1*j,cols), evalKeys));								
		}


		return result;
	}		

	/**
	* EvalMatVMultWideEf - Computes the product between a wide matrix (satisfying rows % cols = 0) 
	* and a vector using tree addition. This is more storage efficient than EvalMatWide.
	* @param ctxt_diags - the matrix is represented as a vector of extended diagonals
	* @param ctxt_v - the vector to multiply with
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEf(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, const size_t BsGs, size_t dim1 ) {

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// // Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));				

		if (BsGs == 0)
		{
			std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(cols);
			ctxt_vRot[0] = ctxt_v; 

	# pragma omp parallel for 
			for (size_t i = 1; i < cols; ++i)
				ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);

		// Element-wise multiplication of the extended diagonals with the rotated vectors
	#pragma omp parallel for	
			for (size_t i = 0; i < rows; i++)
					ctxt_vRot[i] = cc->EvalMult( ctxt_diags[i], ctxt_vRot[i] );


			// Perform binary tree addition
			for (int32_t h = 1; h <= std::ceil(std::log2(rows)); h++){
				for (int32_t i = 0; i < std::ceil((double)rows/pow(2,h)); i++){
					if (i + std::ceil((double)rows/pow(2,h)) < std::ceil((double)rows/pow(2,h-1)))
						ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)rows/pow(2,h))] );
				}
			}
			result = ctxt_vRot[0];
		}
		else // BsGs
		{
			result = EvalMatVMultWideEfBsGs( ctxt_diags, ctxt_v, evalKeys, rows, cols, dim1 );
		}

		double logq = std::ceil(std::log2(double(cols)/double(rows)));
		double aq = pow(2,logq);
		// Assumes that the number of zeros trailing in ctxt_vRot[0] are at least rows*aq - cols 
		// (need zeros up to the closest power of 2 from #rows*ceil(log(#columns/#rows)))

		// Perform binary tree rotations
		for (int32_t h = 1; h <= logq; h++){
			result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex( result, rows*(aq/pow(2,h)), evalKeys ) );
		}

		return result;
	}

	/**
	* EvalMatVMultWideEfBsGs - Computes the product between a wide matrix (satisfying rows % cols = 0) 
	* and a vector using baby step giant step optimization. This is more storage efficient than EvalMatWideBsGs.
	* @param ctxt_diags - the matrix is represented as a vector of extended diagonals
	* @param ctxt_v - the vector to multiply with
	* @param rows - the number of rows of the matrix
	* @param cols - the number of columns of the matrix
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the product
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalMatVMultWideEfBsGs(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& ctxt_diags,
			const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t rows, const size_t cols, size_t dim1 ) {

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);		

		if (dim1 == 0) // but need to somehow make sure that the automorphism keys are generated for this value
			dim1 = std::ceil(std::sqrt(cols));
		int32_t dim2 = std::ceil((double)cols/dim1);		

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));		

		std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(dim1-1);

		// hoisted automorphisms
	#pragma omp parallel for
		for (uint32_t i = 1; i < dim1; i++)
			ctxt_vRot[i-1] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, i, m, vPrecomp, evalKeys);

		for (int32_t j = 0; j < dim2; j ++)
		{
			auto sum = cc->EvalMult(ctxt_diags[dim1*j],ctxt_v);

			for (int32_t i = 1; i < dim1; i ++)
			{
				if (dim1*j + i < rows)
					sum = cc->EvalAdd(sum,cc->EvalMult(ctxt_diags[dim1*j+i],ctxt_vRot[i-1]));
			}
			if (j == 0)
				result = sum;
			else
				result = cc->EvalAdd(result,cc->GetEncryptionAlgorithm()->EvalAtIndex(sum,ReduceRotation(dim1*j,cols),evalKeys));	
		}		

		return result;
	}		

	/**
	* EvalSumRot - Computes the rotate and sum procedure for a vector using tree addition 
	* @param ctxt_v - the vector to rotate and sum
	* @param evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param n - the number of times to rotate
	* @param size - size of the vector
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the sum
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRot(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t n, const size_t size, const size_t BsGs, size_t dim1 ) {
		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);	

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));			

		if (BsGs == 0)
		{
			/* 
			 * No optimization, make sure the correct rotations are computed before calling this
			 */
			// for (int32_t i = 1; i < n; i ++ ) 
			// { 
			// 	result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys));					
			// }			

			/*
			 * Improve by using tree addition with all fast rotations, make sure the correct rotations are computed before calling this
			 */
			// std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> ctxt_vRot(n);
			// ctxt_vRot[0] = ctxt_v; 					

			// for (size_t i = 1; i < n; i++)
			// 	ctxt_vRot[i] = cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys);
			// for (int32_t h = 1; h <= std::ceil(std::log2(n)); h++)
			// {
			// 	for (int32_t i = 0; i < std::ceil((double)n/pow(2,h)); i++)
			// 	{
			// 		if (i + std::ceil((double)n/pow(2,h)) < std::ceil((double)n/pow(2,h-1)))
			// 			ctxt_vRot[i] = cc->EvalAdd( ctxt_vRot[i], ctxt_vRot[i+std::ceil((double)n/pow(2,h))] );			
			// 	}
			// }
			// result = ctxt_vRot[0];

			/* 
			 * Improve by using tree rotate and sum with regular rotations			
			 */	
			for (int32_t h = 1; h <= std::floor(std::log2(n)); h++)
			{
				result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalAtIndex(result, -(int)pow(2,h-1)*size, evalKeys) );			
			}
			for (int32_t i = pow(2,std::floor(std::log2(n))); i < n; i ++ ) // for tall matrix
			{ 
				result = cc->EvalAdd( result, cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys));					
			}				
		} 
		else // baby step giant step
		{
			result = EvalSumRotBsGs( ctxt_v, evalKeys, n, size, dim1 );
		}

		return result;
	}

	/**
	* EvalSumRotBsGs - Computes the rotate and sum procedure for a vector using baby step giant step optimization
	* @param ctxt_v - the vector to rotate and sum
	* @param evalKeys - reference to the map of evaluation keys generated by EvalAutomorphismKeyGen
	* @param n - the number of times to rotate
	* @param size - size of the vector
	* @param BsGs - flag to specify if the baby step giant step method should be used, default = 0
	* @param dim1 - the giant step dimension, default = 0
	* @return a vector containing the sum
	*/
	lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EvalSumRotBsGs(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ctxt_v, const std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &evalKeys, 
			const size_t n, const size_t size, size_t dim1 ) {	

		// Get crypto context
		auto cc = ctxt_v->GetCryptoContext();
		// Get cyclotomic order
		uint32_t m = cc->GetCyclotomicOrder();			
		// Homomorphic fast rotations with precomputations
		auto vPrecomp = cc->EvalFastRotationPrecompute(ctxt_v);		

		if (dim1 == 0) // but need to somehow make sure that the automorphism keys are generated for this value
			dim1 = std::ceil(std::sqrt(n));
		int32_t dim2 = std::ceil((double)n/dim1);		

		lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));		

		for (int32_t j = 0; j < dim2; j ++)
		{

			lbcrypto::Ciphertext<lbcrypto::DCRTPoly> sum(new lbcrypto::CiphertextImpl<lbcrypto::DCRTPoly>(*(ctxt_v)));	

			for (int32_t i = 1; i < dim1; i ++)
			{
				if (dim1*j + i < n)
					sum = cc->EvalAdd(sum,cc->GetEncryptionAlgorithm()->EvalFastRotation(ctxt_v, -i*(int)size, m, vPrecomp, evalKeys));		
			}

			if (j == 0)
				result = sum;
			else
				result = cc->EvalAdd(result,cc->GetEncryptionAlgorithm()->EvalAtIndex(sum,-(int)(dim1*j*size),evalKeys));								

		}						

		return result;
	}				
//////////////////////// Functions for diagonal matrix-vector multiplication and Rotate and Sum //////////////////////// 


void CompressEvalKeys(std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>> &ek, size_t level) {

	const shared_ptr<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>> cryptoParams =
			std::dynamic_pointer_cast<lbcrypto::LPCryptoParametersCKKS<lbcrypto::DCRTPoly>>(ek.begin()->second->GetCryptoParameters());

	if (cryptoParams->GetKeySwitchTechnique() == lbcrypto::BV) {

		std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>>::iterator it;

		for ( it = ek.begin(); it != ek.end(); it++ )
		{
			std::vector<lbcrypto::DCRTPoly> b = it->second->GetBVector();
			std::vector<lbcrypto::DCRTPoly> a = it->second->GetAVector();

			for (size_t k = 0; k < a.size(); k++) {
				a[k].DropLastElements(level);
				b[k].DropLastElements(level);
			}

			it->second->ClearKeys();

			it->second->SetAVector(std::move(a));
			it->second->SetBVector(std::move(b));
		}
	} else if (cryptoParams->GetKeySwitchTechnique() == lbcrypto::HYBRID) {

		size_t curCtxtLevel = ek.begin()->second->GetBVector()[0].GetParams()->GetParams().size() -
				cryptoParams->GetAuxElementParams()->GetParams().size();

		// current number of levels after compression
		uint32_t newLevels = curCtxtLevel - level;

		std::map<usint, lbcrypto::LPEvalKey<lbcrypto::DCRTPoly>>::iterator it;

		for ( it = ek.begin(); it != ek.end(); it++ )
		{
			std::vector<lbcrypto::DCRTPoly> b = it->second->GetBVector();
			std::vector<lbcrypto::DCRTPoly> a = it->second->GetAVector();

			for (size_t k = 0; k < a.size(); k++) {

				auto elementsA = a[k].GetAllElements();
				auto elementsB = b[k].GetAllElements();
				for(size_t i = newLevels; i < curCtxtLevel; i++) {
					elementsA.erase(elementsA.begin() + newLevels);
					elementsB.erase(elementsB.begin() + newLevels);
				}

				a[k] = lbcrypto::DCRTPoly(elementsA);
				b[k] = lbcrypto::DCRTPoly(elementsB);

			}

			it->second->ClearKeys();

			it->second->SetAVector(std::move(a));
			it->second->SetBVector(std::move(b));
		}

	} 
	else {
		PALISADE_THROW(lbcrypto::not_available_error, "Compressed evaluation keys are not currently supported for GHS keyswitching.");
	}


}

/**
 * Ensures that the index for rotation is positive and between 1 and the size of the vector.
 *
 * @param index signed rotation amount.
 * @param size size of vector that is rotated.
 */
size_t ReduceRotation(size_t index, size_t size){

	int32_t isize = int32_t(size);

	// if size is a power of 2
	if ((size & (size - 1)) == 0){
		int32_t n = log2(size);
		if (index >= 0)
			return index - ((index >> n) << n);
		else
			return index+isize + ((int32_t(fabs(index)) >> n) << n);
	}
	else
		return (isize+index%isize)%isize;
}


#endif