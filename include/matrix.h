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
	
    // init configurations
    // sets to all one value
    Matrix all_scalar(double d);
    // sets diagonal to one. *Warning*: will not change the off-diagnoal elements
    Matrix eye();
    // sets to all one value
    Matrix ones();
    
    // transpose
	Matrix t();
    // checks if matrix is a one by one matrix, holding a single scalar
    bool is_scalar() const;

	inline const Matrix like_ones() {
        Matrix out = *this;
        for (double& entry : out.val) {
            entry = 1;
        }

        return out;
    }

	void set(int i, int j, double value);
	double get(int i, int j);
    // unary matrix operations
    // negation
    Matrix operator-();
	// exp
	Matrix exp();
    // log: the natural log
	Matrix log();
    // dot
    Matrix dot(Matrix b);

};

// utility operations
// print
void operator<<(std::ostream& o, Matrix& b);
// optionally broadcast from scalar to matrix like b;
const Matrix broadcast_like(Matrix a, Matrix b);
// matrix-matrix operations
// add
Matrix operator+(Matrix a, Matrix b);
// sub
Matrix operator-(Matrix a, Matrix b);
// mul
Matrix operator*(Matrix a, Matrix b);
// div
Matrix operator/(Matrix a, Matrix b);
// matrix-scalar operations
// add
Matrix operator+(double a, Matrix b);
Matrix operator+(Matrix a, double b);
// sub
Matrix operator-(double a, Matrix b);
Matrix operator-(Matrix a, double b);
// mul
Matrix operator*(double d, Matrix b);
Matrix operator*(Matrix b, double d);
// div
Matrix operator/(double d, Matrix b);
Matrix operator/(Matrix b, double d);