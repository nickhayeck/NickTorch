#pragma once
#include <vector>
#include <iostream>
#include <cmath>


// minimal floating-point matrix implementation
struct Matrix {
	int rows;
	int cols;
	std::vector<double> val;
	bool is_transposed;
	// scalar init
	Matrix(double d) : rows(1), cols(1), val({d}), is_transposed(false) {}
	//array init
	template<int L>
	Matrix(double (&arr)[L], int N, int K) : 
	rows(N), cols(K), val(arr, arr + L), is_transposed(false) {
		assert(L == N*K);
	}
	// zero init
	Matrix(int N, int K) : rows(N), cols(K), val(N*K), is_transposed(false) {}
	// default constructor
	Matrix() : rows(1), cols(1), val(1), is_transposed(false) {}
	
    // transpose
	Matrix t() const;

	int current_rows() const;

	int current_cols() const;

	inline const Matrix like_ones() {
        Matrix out = *this;
        for (double& entry : out.val) {
            entry = 1;
        }

        return out;
    }

	void set(int i, int j, double value);
	double get(int i, int j);
	// matrix-matrix operations
	// add
	Matrix operator+(Matrix& b);
	// sub
	Matrix operator-(Matrix& b);
	// mul
	Matrix operator*(Matrix& b);
	// div
	Matrix operator/(Matrix& b);
	// dot
	Matrix dot(Matrix& other);
	// exp
	Matrix exp();

};

// utility operations
// print
void operator<<(std::ostream& o, Matrix& b);
// matrix-scalar operations
// mul
Matrix operator*(double d, Matrix b);
// div
Matrix operator/(double d, Matrix b);