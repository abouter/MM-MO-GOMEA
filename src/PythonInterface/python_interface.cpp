#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <Eigen/Dense>

#include "GPGOMEA/Evolution/EvolutionRun.h"
#include "GPGOMEA/Evolution/EvolutionState.h"
#include "GPGOMEA/Evolution/MOArchive.h"
#include "GPGOMEA/RunHandling/IMSHandler.h"

namespace py = pybind11; 
using namespace std;

typedef Eigen::ArrayXXf MatRef;
typedef Eigen::ArrayXf VecRef;
//typedef Eigen::Ref<Eigen::ArrayXXf> MatRef;
//typedef Eigen::Ref<Eigen::ArrayXf> VecRef;

arma::mat MatRefToArma(MatRef m){
	arma::mat out(m.rows(), m.cols());
	for(int i = 0; i < m.rows(); i++)
		for(int j = 0; j < m.cols(); j++)
			out(i,j) = m(i,j);
	return out;
}

arma::vec VecRefToArma(VecRef v){
	arma::vec out(v.rows());
	for(int i = 0; i < v.rows(); i++)
		out(i) = v(i);
	return out;
}

py::list evolve(string options, MatRef X, VecRef y) {
	// 1. SETUP
	auto opts = Utils::SplitStringByChar(options, ' ');
	int argc = opts.size()+1;
	char * argv[argc];
	string title = "mmmogpg";
	argv[0] = (char*) title.c_str();
	for (int i = 1; i < argc; i++) {
		argv[i] = (char*) opts[i-1].c_str();
	}

	EvolutionState * st = new EvolutionState();
	st->config->running_from_python = true;
	st->SetOptions(argc, argv);
	arma::mat aXtrain = MatRefToArma(X);
	arma::mat aXtest = MatRefToArma(X);
	arma::mat aytrain = VecRefToArma(y);
	arma::mat aytest = VecRefToArma(y);
	st->SetDataSetTraining(aXtrain, aytrain);
	st->SetDataSetTest(aXtest, aytest);

	IMSHandler * imsh = new IMSHandler(st);

	// 2. RUN
	imsh->Start();

	// 3. OUTPUT
	Node *elitist = imsh->GetFinalElitist();
	/*MOArchive out_archive;
	for (EvolutionRun * r : imsh->runs) {
		for( auto ind : r->mo_archive.mo_archive ){
			out_archive.UpdateMOArchive(ind);
		}
	}
	if (out_archive.mo_archive.empty()) {
		throw runtime_error("No models found, something went wrong");
	}*/

	py::list models;
	//for (auto sol : out_archive.mo_archive ){
		string model_repr = elitist->GetPythonExpression();//sol->GetPythonExpression();
		models.append(model_repr);
	//}

	// 4. CLEANUP
	delete imsh;

	return models;
}

PYBIND11_MODULE(_pb_mmmogpg, m) {
  m.doc() = "pybind11-based interface for MM-MO-GOMEA"; // optional module docstring
  m.def("evolve", &evolve, "Runs MM-MO-GOMEA evolution in C++");
}
