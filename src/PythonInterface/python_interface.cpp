#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
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

py::list evolve(MatRef Xtrain, VecRef ytrain, MatRef Xtest = MatRef(), VecRef ytest = VecRef(), string file = "", const py::kwargs &kwargs = py::dict()) {
	// 1. SETUP
	EvolutionState * st = new EvolutionState();
	st->config->running_from_python = true;
	
	if( file.size() > 0 )  
	{
		st->SetOptionsFromFile(file);
		if( kwargs )
			std::cout << "Warning: Reading parameter settings from file. Input parameters are ignored." << std::endl;
	}
	else if( kwargs )
	{
		int argc = 1;
		char * argv[kwargs.size()+1];
		string title = "mmmogpg";
		argv[0] = (char*) title.c_str();
		
		for( auto item: kwargs )
		{
			std::string arg_str = "--";
			arg_str += std::string(py::str(item.first));
			arg_str += "=";
			arg_str += std::string(py::str(item.second));
			argv[argc++] = (char*) arg_str.c_str();
			//std::cout << arg_str << std::endl;
		}
		st->SetOptions(argc, argv);
	}

	arma::mat aXtrain = MatRefToArma(Xtrain);
	arma::mat aytrain = VecRefToArma(ytrain);
	st->SetDataSetTraining(aXtrain, aytrain);
	if (Xtest.size() > 0 && ytest.size() > 0)
	{
		arma::mat aXtest = MatRefToArma(Xtest);
		arma::mat aytest = VecRefToArma(ytest);
		st->SetDataSetTest(aXtest, aytest);
	}
	else
	{
		arma::mat aXtest = MatRefToArma(Xtrain);
		arma::mat aytest = VecRefToArma(ytrain);
		st->SetDataSetTest(aXtest, aytest);
	}

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
  m.def("evolve", &evolve, "Runs MM-MO-GOMEA evolution in C++",
  	py::arg("Xtrain").none(true) = py::none(), py::arg("ytrain").none(true) = py::none(), py::arg_v("Xtest", MatRef(), "None"), py::arg_v("ytest", VecRef(), "None"), py::arg("file") );
}
